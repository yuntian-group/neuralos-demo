from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any, Optional
import asyncio
import json
import time
import os
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import aiohttp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SessionStatus(Enum):
    QUEUED = "queued"
    ACTIVE = "active"
    COMPLETED = "completed"
    TIMEOUT = "timeout"

@dataclass
class UserSession:
    session_id: str
    client_id: str
    websocket: WebSocket
    created_at: float
    status: SessionStatus
    worker_id: Optional[str] = None
    last_activity: Optional[float] = None
    max_session_time: Optional[float] = None
    user_has_interacted: bool = False

@dataclass
class WorkerInfo:
    worker_id: str
    gpu_id: int
    endpoint: str
    is_available: bool
    current_session: Optional[str] = None
    last_ping: float = 0

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, UserSession] = {}
        self.workers: Dict[str, WorkerInfo] = {}
        self.session_queue: List[str] = []
        self.active_sessions: Dict[str, str] = {}  # session_id -> worker_id
        
        # Configuration
        self.IDLE_TIMEOUT = 20.0  # When no queue
        self.QUEUE_WARNING_TIME = 10.0
        self.MAX_SESSION_TIME_WITH_QUEUE = 60.0  # When there's a queue
        self.QUEUE_SESSION_WARNING_TIME = 45.0  # 15 seconds before timeout
        self.GRACE_PERIOD = 10.0

    async def register_worker(self, worker_id: str, gpu_id: int, endpoint: str):
        """Register a new worker"""
        self.workers[worker_id] = WorkerInfo(
            worker_id=worker_id,
            gpu_id=gpu_id,
            endpoint=endpoint,
            is_available=True,
            last_ping=time.time()
        )
        logger.info(f"Registered worker {worker_id} on GPU {gpu_id} at {endpoint}")

    async def get_available_worker(self) -> Optional[WorkerInfo]:
        """Get an available worker"""
        for worker in self.workers.values():
            if worker.is_available and time.time() - worker.last_ping < 30:  # Worker ping timeout
                return worker
        return None

    async def add_session_to_queue(self, session: UserSession):
        """Add a session to the queue"""
        self.sessions[session.session_id] = session
        self.session_queue.append(session.session_id)
        session.status = SessionStatus.QUEUED
        logger.info(f"Added session {session.session_id} to queue. Queue size: {len(self.session_queue)}")

    async def process_queue(self):
        """Process the session queue"""
        while self.session_queue:
            session_id = self.session_queue[0]
            session = self.sessions.get(session_id)
            
            if not session or session.status != SessionStatus.QUEUED:
                self.session_queue.pop(0)
                continue
                
            worker = await self.get_available_worker()
            if not worker:
                break  # No available workers
                
            # Assign session to worker
            self.session_queue.pop(0)
            session.status = SessionStatus.ACTIVE
            session.worker_id = worker.worker_id
            session.last_activity = time.time()
            
            # Set session time limit based on queue status
            if len(self.session_queue) > 0:
                session.max_session_time = self.MAX_SESSION_TIME_WITH_QUEUE
            
            worker.is_available = False
            worker.current_session = session_id
            self.active_sessions[session_id] = worker.worker_id
            
            logger.info(f"Assigned session {session_id} to worker {worker.worker_id}")
            
            # Notify user that their session is starting
            await self.notify_session_start(session)
            
            # Start session monitoring
            asyncio.create_task(self.monitor_active_session(session_id))

    async def notify_session_start(self, session: UserSession):
        """Notify user that their session is starting"""
        try:
            await session.websocket.send_json({
                "type": "session_start",
                "worker_id": session.worker_id,
                "max_session_time": session.max_session_time
            })
        except Exception as e:
            logger.error(f"Failed to notify session start for {session.session_id}: {e}")

    async def monitor_active_session(self, session_id: str):
        """Monitor an active session for timeouts"""
        session = self.sessions.get(session_id)
        if not session:
            return
            
        try:
            while session.status == SessionStatus.ACTIVE:
                current_time = time.time()
                
                # Check if session has exceeded time limit
                if session.max_session_time:
                    elapsed = current_time - session.last_activity if session.last_activity else 0
                    remaining = session.max_session_time - elapsed
                    
                    # Send warning at 15 seconds before timeout
                    if remaining <= 15 and remaining > 10:
                        await session.websocket.send_json({
                            "type": "session_warning",
                            "time_remaining": remaining,
                            "queue_size": len(self.session_queue)
                        })
                    
                    # Grace period handling
                    elif remaining <= 10 and remaining > 0:
                        # Check if queue is empty - if so, extend session
                        if len(self.session_queue) == 0:
                            session.max_session_time = None  # Remove time limit
                            await session.websocket.send_json({
                                "type": "time_limit_removed",
                                "reason": "queue_empty"
                            })
                        else:
                            await session.websocket.send_json({
                                "type": "grace_period",
                                "time_remaining": remaining,
                                "queue_size": len(self.session_queue)
                            })
                    
                    # Timeout
                    elif remaining <= 0:
                        await self.end_session(session_id, SessionStatus.TIMEOUT)
                        return
                
                # Check idle timeout when no queue
                elif not session.max_session_time and session.last_activity:
                    idle_time = current_time - session.last_activity
                    if idle_time >= self.IDLE_TIMEOUT:
                        await self.end_session(session_id, SessionStatus.TIMEOUT)
                        return
                    elif idle_time >= self.QUEUE_WARNING_TIME:
                        await session.websocket.send_json({
                            "type": "idle_warning",
                            "time_remaining": self.IDLE_TIMEOUT - idle_time
                        })
                
                await asyncio.sleep(1)  # Check every second
                
        except Exception as e:
            logger.error(f"Error monitoring session {session_id}: {e}")
            await self.end_session(session_id, SessionStatus.COMPLETED)

    async def end_session(self, session_id: str, status: SessionStatus):
        """End a session and free up the worker"""
        session = self.sessions.get(session_id)
        if not session:
            return
            
        session.status = status
        
        # Free up the worker
        if session.worker_id and session.worker_id in self.workers:
            worker = self.workers[session.worker_id]
            worker.is_available = True
            worker.current_session = None
            
            # Notify worker to clean up
            try:
                async with aiohttp.ClientSession() as client_session:
                    await client_session.post(f"{worker.endpoint}/end_session", 
                                            json={"session_id": session_id})
            except Exception as e:
                logger.error(f"Failed to notify worker {worker.worker_id} of session end: {e}")
        
        # Remove from active sessions
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            
        logger.info(f"Ended session {session_id} with status {status}")
        
        # Process next in queue
        asyncio.create_task(self.process_queue())

    async def update_queue_info(self):
        """Send queue information to waiting users"""
        for i, session_id in enumerate(self.session_queue):
            session = self.sessions.get(session_id)
            if session and session.status == SessionStatus.QUEUED:
                try:
                    # Calculate estimated wait time
                    active_sessions_count = len(self.active_sessions)
                    avg_session_time = self.MAX_SESSION_TIME_WITH_QUEUE if active_sessions_count > 0 else 30.0
                    estimated_wait = (i + 1) * avg_session_time / max(len(self.workers), 1)
                    
                    await session.websocket.send_json({
                        "type": "queue_update",
                        "position": i + 1,
                        "total_waiting": len(self.session_queue),
                        "estimated_wait_minutes": estimated_wait / 60,
                        "active_sessions": active_sessions_count
                    })
                except Exception as e:
                    logger.error(f"Failed to send queue update to session {session_id}: {e}")

    async def handle_user_activity(self, session_id: str):
        """Update user activity timestamp"""
        session = self.sessions.get(session_id)
        if session:
            session.last_activity = time.time()
            if not session.user_has_interacted:
                session.user_has_interacted = True
                logger.info(f"User started interacting in session {session_id}")

    async def _forward_to_worker(self, worker: WorkerInfo, session_id: str, data: dict):
        """Forward input to worker asynchronously"""
        try:
            async with aiohttp.ClientSession() as client_session:
                async with client_session.post(
                    f"{worker.endpoint}/process_input",
                    json={
                        "session_id": session_id,
                        "data": data
                    }
                ) as response:
                    if response.status != 200:
                        logger.error(f"Worker returned status {response.status}")
                        # Optionally handle worker errors here
        except Exception as e:
            logger.error(f"Error forwarding to worker {worker.worker_id}: {e}")

# Global session manager
session_manager = SessionManager()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get():
    return HTMLResponse(open("static/index.html").read())

@app.post("/register_worker")
async def register_worker(worker_info: dict):
    """Endpoint for workers to register themselves"""
    await session_manager.register_worker(
        worker_info["worker_id"],
        worker_info["gpu_id"], 
        worker_info["endpoint"]
    )
    return {"status": "registered"}

@app.post("/worker_ping")
async def worker_ping(worker_info: dict):
    """Endpoint for workers to ping their availability"""
    worker_id = worker_info["worker_id"]
    if worker_id in session_manager.workers:
        session_manager.workers[worker_id].last_ping = time.time()
        session_manager.workers[worker_id].is_available = worker_info.get("is_available", True)
    return {"status": "ok"}

@app.post("/worker_result")
async def worker_result(result_data: dict):
    """Endpoint for workers to send back processing results"""
    session_id = result_data.get("session_id")
    worker_id = result_data.get("worker_id")
    result = result_data.get("result")
    
    if not session_id or not result:
        raise HTTPException(status_code=400, detail="Missing session_id or result")
    
    # Find the session and send result to the WebSocket
    session = session_manager.sessions.get(session_id)
    if session and session.status == SessionStatus.ACTIVE:
        try:
            await session.websocket.send_json(result)
            logger.info(f"Sent result to session {session_id}")
        except Exception as e:
            logger.error(f"Failed to send result to session {session_id}: {e}")
    else:
        logger.warning(f"Could not find active session {session_id} for result")
    
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Create session
    session_id = str(uuid.uuid4())
    client_id = f"{int(time.time())}_{session_id[:8]}"
    
    session = UserSession(
        session_id=session_id,
        client_id=client_id,
        websocket=websocket,
        created_at=time.time(),
        status=SessionStatus.QUEUED
    )
    
    logger.info(f"New WebSocket connection: {client_id}")
    
    try:
        # Add to queue
        await session_manager.add_session_to_queue(session)
        
        # Try to process queue immediately
        await session_manager.process_queue()
        
        # Send initial queue status
        if session.status == SessionStatus.QUEUED:
            await session_manager.update_queue_info()
        
        # Main message loop
        while True:
            try:
                data = await websocket.receive_json()
                
                # Update activity
                await session_manager.handle_user_activity(session_id)
                
                # Handle different message types
                if data.get("type") == "heartbeat":
                    await websocket.send_json({"type": "heartbeat_response"})
                    continue
                
                # If session is active, forward to worker
                if session.status == SessionStatus.ACTIVE and session.worker_id:
                    worker = session_manager.workers.get(session.worker_id)
                    if worker:
                        try:
                            # Forward message to worker (don't wait for response for regular inputs)
                            # The worker will send results back asynchronously via /worker_result
                            asyncio.create_task(session_manager._forward_to_worker(worker, session_id, data))
                        except Exception as e:
                            logger.error(f"Error forwarding to worker: {e}")
                
                # Handle control messages (these need synchronous responses)
                elif data.get("type") in ["reset", "update_sampling_steps", "update_use_rnn", "get_settings"]:
                    if session.status == SessionStatus.ACTIVE and session.worker_id:
                        worker = session_manager.workers.get(session.worker_id)
                        if worker:
                            try:
                                async with aiohttp.ClientSession() as client_session:
                                    async with client_session.post(
                                        f"{worker.endpoint}/process_input",
                                        json={
                                            "session_id": session_id,
                                            "data": data
                                        }
                                    ) as response:
                                        if response.status == 200:
                                            result = await response.json()
                                            await websocket.send_json(result)
                                        else:
                                            logger.error(f"Worker returned status {response.status}")
                            except Exception as e:
                                logger.error(f"Error forwarding control message: {e}")
                    else:
                        # Send appropriate response for queued users
                        await websocket.send_json({
                            "type": "error",
                            "message": "Session not active yet. Please wait in queue."
                        })
                        
            except asyncio.TimeoutError:
                logger.info("WebSocket connection timed out")
                break
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {client_id}")
                break
                
    except Exception as e:
        logger.error(f"Error in WebSocket connection {client_id}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up session
        if session_id in session_manager.sessions:
            await session_manager.end_session(session_id, SessionStatus.COMPLETED)
            del session_manager.sessions[session_id]
        
        logger.info(f"WebSocket connection closed: {client_id}")

# Background task to periodically update queue info
async def periodic_queue_update():
    while True:
        try:
            await session_manager.update_queue_info()
            await asyncio.sleep(5)  # Update every 5 seconds
        except Exception as e:
            logger.error(f"Error in periodic queue update: {e}")

@app.on_event("startup")
async def startup_event():
    # Start background tasks
    asyncio.create_task(periodic_queue_update())

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Dispatcher for Neural OS")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the dispatcher on")
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port) 