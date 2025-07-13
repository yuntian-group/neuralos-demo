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
from collections import defaultdict, deque
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Analytics and monitoring
class SystemAnalytics:
    def __init__(self):
        self.start_time = time.time()
        self.total_connections = 0
        self.active_connections = 0
        self.total_interactions = 0
        self.ip_addresses = defaultdict(int)  # IP -> connection count
        self.session_durations = deque(maxlen=100)  # Last 100 session durations
        self.waiting_times = deque(maxlen=100)  # Last 100 waiting times
        self.users_bypassed_queue = 0  # Users who got GPU immediately
        self.users_waited_in_queue = 0  # Users who had to wait
        self.gpu_utilization_samples = deque(maxlen=100)  # GPU utilization over time
        self.queue_size_samples = deque(maxlen=100)  # Queue size over time
        self.log_file = None
        self._init_log_file()
    
    def _init_log_file(self):
        """Initialize the system log file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"system_analytics_{timestamp}.log"
        self.log_file = log_filename
        self._write_log("="*80)
        self._write_log("NEURAL OS MULTI-GPU SYSTEM ANALYTICS")
        self._write_log("="*80)
        self._write_log(f"System started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_log("")
    
    def _write_log(self, message):
        """Write message to log file and console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")
    
    def log_new_connection(self, client_id: str, ip: str):
        """Log new connection"""
        self.total_connections += 1
        self.active_connections += 1
        self.ip_addresses[ip] += 1
        
        unique_ips = len(self.ip_addresses)
        self._write_log(f"🔗 NEW CONNECTION: {client_id} from {ip}")
        self._write_log(f"   📊 Total connections: {self.total_connections} | Active: {self.active_connections} | Unique IPs: {unique_ips}")
    
    def log_connection_closed(self, client_id: str, duration: float, interactions: int, reason: str = "normal"):
        """Log connection closed"""
        self.active_connections -= 1
        self.total_interactions += interactions
        self.session_durations.append(duration)
        
        avg_duration = sum(self.session_durations) / len(self.session_durations) if self.session_durations else 0
        
        self._write_log(f"🚪 CONNECTION CLOSED: {client_id}")
        self._write_log(f"   ⏱️  Duration: {duration:.1f}s | Interactions: {interactions} | Reason: {reason}")
        self._write_log(f"   📊 Active connections: {self.active_connections} | Avg session duration: {avg_duration:.1f}s")
    
    def log_queue_bypass(self, client_id: str):
        """Log when user bypasses queue (gets GPU immediately)"""
        self.users_bypassed_queue += 1
        bypass_rate = (self.users_bypassed_queue / self.total_connections) * 100 if self.total_connections > 0 else 0
        self._write_log(f"⚡ QUEUE BYPASS: {client_id} got GPU immediately")
        self._write_log(f"   📊 Bypass rate: {bypass_rate:.1f}% ({self.users_bypassed_queue}/{self.total_connections})")
    
    def log_queue_wait(self, client_id: str, wait_time: float, queue_position: int):
        """Log when user had to wait in queue"""
        self.users_waited_in_queue += 1
        self.waiting_times.append(wait_time)
        
        avg_wait = sum(self.waiting_times) / len(self.waiting_times) if self.waiting_times else 0
        wait_rate = (self.users_waited_in_queue / self.total_connections) * 100 if self.total_connections > 0 else 0
        
        self._write_log(f"⏳ QUEUE WAIT: {client_id} waited {wait_time:.1f}s (was #{queue_position})")
        self._write_log(f"   📊 Wait rate: {wait_rate:.1f}% | Avg wait time: {avg_wait:.1f}s")
    
    def log_gpu_status(self, total_gpus: int, active_gpus: int, available_gpus: int):
        """Log GPU utilization"""
        utilization = (active_gpus / total_gpus) * 100 if total_gpus > 0 else 0
        self.gpu_utilization_samples.append(utilization)
        
        avg_utilization = sum(self.gpu_utilization_samples) / len(self.gpu_utilization_samples) if self.gpu_utilization_samples else 0
        
        self._write_log(f"🖥️  GPU STATUS: {active_gpus}/{total_gpus} in use ({utilization:.1f}% utilization)")
        self._write_log(f"   📊 Available: {available_gpus} | Avg utilization: {avg_utilization:.1f}%")
    
    def log_worker_registered(self, worker_id: str, gpu_id: int, endpoint: str):
        """Log when a worker registers"""
        self._write_log(f"⚙️  WORKER REGISTERED: {worker_id} (GPU {gpu_id}) at {endpoint}")
    
    def log_worker_disconnected(self, worker_id: str, gpu_id: int):
        """Log when a worker disconnects"""
        self._write_log(f"⚙️  WORKER DISCONNECTED: {worker_id} (GPU {gpu_id})")
    
    def log_no_workers_available(self, queue_size: int):
        """Log critical situation when no workers are available"""
        self._write_log(f"⚠️  CRITICAL: No GPU workers available! {queue_size} users waiting")
        self._write_log("   Please check worker processes and GPU availability")
    
    def log_queue_status(self, queue_size: int, estimated_wait: float):
        """Log queue status"""
        self.queue_size_samples.append(queue_size)
        
        avg_queue_size = sum(self.queue_size_samples) / len(self.queue_size_samples) if self.queue_size_samples else 0
        
        if queue_size > 0:
            self._write_log(f"📝 QUEUE STATUS: {queue_size} users waiting | Est. wait: {estimated_wait:.1f}s")
            self._write_log(f"   📊 Avg queue size: {avg_queue_size:.1f}")
    
    def log_periodic_summary(self):
        """Log periodic system summary"""
        uptime = time.time() - self.start_time
        uptime_hours = uptime / 3600
        
        unique_ips = len(self.ip_addresses)
        avg_duration = sum(self.session_durations) / len(self.session_durations) if self.session_durations else 0
        avg_wait = sum(self.waiting_times) / len(self.waiting_times) if self.waiting_times else 0
        avg_utilization = sum(self.gpu_utilization_samples) / len(self.gpu_utilization_samples) if self.gpu_utilization_samples else 0
        avg_queue_size = sum(self.queue_size_samples) / len(self.queue_size_samples) if self.queue_size_samples else 0
        
        bypass_rate = (self.users_bypassed_queue / self.total_connections) * 100 if self.total_connections > 0 else 0
        
        self._write_log("")
        self._write_log("="*60)
        self._write_log("📊 SYSTEM SUMMARY")
        self._write_log("="*60)
        self._write_log(f"⏱️  Uptime: {uptime_hours:.1f} hours")
        self._write_log(f"🔗 Connections: {self.total_connections} total | {self.active_connections} active | {unique_ips} unique IPs")
        self._write_log(f"💬 Total interactions: {self.total_interactions}")
        self._write_log(f"⚡ Queue bypass rate: {bypass_rate:.1f}% ({self.users_bypassed_queue}/{self.total_connections})")
        self._write_log(f"⏳ Avg waiting time: {avg_wait:.1f}s")
        self._write_log(f"📝 Avg queue size: {avg_queue_size:.1f}")
        self._write_log(f"🖥️  Avg GPU utilization: {avg_utilization:.1f}%")
        self._write_log(f"⏱️  Avg session duration: {avg_duration:.1f}s")
        self._write_log("")
        self._write_log("🌍 TOP IP ADDRESSES:")
        for ip, count in sorted(self.ip_addresses.items(), key=lambda x: x[1], reverse=True)[:10]:
            self._write_log(f"   {ip}: {count} connections")
        self._write_log("="*60)
        self._write_log("")

# Initialize analytics
analytics = SystemAnalytics()

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
    ip_address: Optional[str] = None
    interaction_count: int = 0
    queue_start_time: Optional[float] = None
    idle_warning_sent: bool = False
    session_warning_sent: bool = False

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
        
        # Log worker registration
        analytics.log_worker_registered(worker_id, gpu_id, endpoint)
        
        # Log GPU status
        total_gpus = len(self.workers)
        active_gpus = len([w for w in self.workers.values() if not w.is_available])
        available_gpus = total_gpus - active_gpus
        analytics.log_gpu_status(total_gpus, active_gpus, available_gpus)

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
        session.queue_start_time = time.time()
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
                # Log critical situation if no workers are available
                if len(self.workers) == 0:
                    analytics.log_no_workers_available(len(self.session_queue))
                break  # No available workers
                
            # Calculate wait time
            wait_time = time.time() - session.queue_start_time if session.queue_start_time else 0
            queue_position = self.session_queue.index(session_id) + 1
            
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
            
            # Log analytics
            if wait_time > 0:
                analytics.log_queue_wait(session.client_id, wait_time, queue_position)
            else:
                analytics.log_queue_bypass(session.client_id)
            
            # Log GPU status
            total_gpus = len(self.workers)
            active_gpus = len([w for w in self.workers.values() if not w.is_available])
            available_gpus = total_gpus - active_gpus
            analytics.log_gpu_status(total_gpus, active_gpus, available_gpus)
            
            # Initialize session on worker with client_id for logging
            try:
                async with aiohttp.ClientSession() as client_session:
                    await client_session.post(f"{worker.endpoint}/init_session", json={
                        "session_id": session_id,
                        "client_id": session.client_id
                    })
            except Exception as e:
                logger.error(f"Failed to initialize session on worker {worker.worker_id}: {e}")
            
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
                    
                    # Send warning at 15 seconds before timeout (only once)
                    if remaining <= 15 and remaining > 10 and not session.session_warning_sent:
                        await session.websocket.send_json({
                            "type": "session_warning",
                            "time_remaining": remaining,
                            "queue_size": len(self.session_queue)
                        })
                        session.session_warning_sent = True
                        logger.info(f"Session warning sent to {session_id}, time remaining: {remaining:.1f}s")
                    
                    # Grace period handling
                    elif remaining <= 10 and remaining > 0:
                        # Check if queue is empty - if so, extend session
                        if len(self.session_queue) == 0:
                            session.max_session_time = None  # Remove time limit
                            session.session_warning_sent = False  # Reset warning since limit removed
                            await session.websocket.send_json({
                                "type": "time_limit_removed",
                                "reason": "queue_empty"
                            })
                            logger.info(f"Session time limit removed for {session_id} (queue empty)")
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
                    elif idle_time >= self.QUEUE_WARNING_TIME and not session.idle_warning_sent:
                        await session.websocket.send_json({
                            "type": "idle_warning",
                            "time_remaining": self.IDLE_TIMEOUT - idle_time
                        })
                        session.idle_warning_sent = True
                        logger.info(f"Idle warning sent to {session_id}, time remaining: {self.IDLE_TIMEOUT - idle_time:.1f}s")
                
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
        
        # Calculate session duration
        duration = time.time() - session.created_at
        
        # Log analytics
        reason = "timeout" if status == SessionStatus.TIMEOUT else "normal"
        analytics.log_connection_closed(session.client_id, duration, session.interaction_count, reason)
        
        # Free up the worker
        if session.worker_id and session.worker_id in self.workers:
            worker = self.workers[session.worker_id]
            worker.is_available = True
            worker.current_session = None
            
            # Log GPU status
            total_gpus = len(self.workers)
            active_gpus = len([w for w in self.workers.values() if not w.is_available])
            available_gpus = total_gpus - active_gpus
            analytics.log_gpu_status(total_gpus, active_gpus, available_gpus)
            
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
                    # Calculate dynamic estimated wait time
                    estimated_wait = self._calculate_dynamic_wait_time(i + 1)
                    
                    await session.websocket.send_json({
                        "type": "queue_update",
                        "position": i + 1,
                        "total_waiting": len(self.session_queue),
                        "estimated_wait_seconds": estimated_wait,
                        "active_sessions": len(self.active_sessions),
                        "available_workers": len([w for w in self.workers.values() if w.is_available])
                    })
                except Exception as e:
                    logger.error(f"Failed to send queue update to session {session_id}: {e}")
        
        # Log queue status if there's a queue
        if self.session_queue:
            estimated_wait = self._calculate_dynamic_wait_time(1)
            analytics.log_queue_status(len(self.session_queue), estimated_wait)

    def _calculate_dynamic_wait_time(self, position_in_queue: int) -> float:
        """Calculate dynamic estimated wait time based on current session progress"""
        current_time = time.time()
        available_workers = len([w for w in self.workers.values() if w.is_available])
        
        # If there are available workers, no wait time
        if available_workers > 0:
            return 0
        
        # Calculate remaining time for active sessions
        min_remaining_time = float('inf')
        active_session_times = []
        
        for session_id in self.active_sessions:
            session = self.sessions.get(session_id)
            if session and session.last_activity:
                if session.max_session_time:
                    # Session has time limit (queue exists)
                    elapsed = current_time - session.last_activity
                    remaining = session.max_session_time - elapsed
                    remaining = max(0, remaining)  # Don't go negative
                else:
                    # No time limit, estimate based on average usage
                    elapsed = current_time - session.last_activity
                    # Assume sessions without time limits will run for average of 45 seconds more
                    remaining = max(45 - elapsed, 10)  # Minimum 10 seconds
                
                active_session_times.append(remaining)
                min_remaining_time = min(min_remaining_time, remaining)
        
        # If no active sessions found, use default
        if not active_session_times:
            min_remaining_time = 30.0
        
        # Calculate estimated wait time based on position
        num_workers = len(self.workers)
        if num_workers == 0:
            return 999  # No workers available
        
        if position_in_queue <= num_workers:
            # User will get a worker as soon as current sessions end
            return min_remaining_time
        else:
            # User needs to wait for multiple session cycles
            cycles_to_wait = (position_in_queue - 1) // num_workers
            remaining_in_current_cycle = (position_in_queue - 1) % num_workers + 1
            
            # Time for complete cycles (use average session time)
            avg_session_time = self.MAX_SESSION_TIME_WITH_QUEUE if len(self.session_queue) > 0 else 45.0
            full_cycles_time = cycles_to_wait * avg_session_time
            
            # Time for current partial cycle
            if remaining_in_current_cycle <= len(active_session_times):
                # Sort session times to get when the Nth worker will be free
                sorted_times = sorted(active_session_times)
                current_cycle_time = sorted_times[remaining_in_current_cycle - 1]
            else:
                current_cycle_time = min_remaining_time
            
            return full_cycles_time + current_cycle_time

    async def handle_user_activity(self, session_id: str):
        """Update user activity timestamp and reset warning flags"""
        session = self.sessions.get(session_id)
        if session:
            old_time = session.last_activity
            session.last_activity = time.time()
            session.interaction_count += 1
            
            # Reset warning flags if user activity detected after warnings
            warning_reset = False
            if session.idle_warning_sent:
                logger.info(f"User activity detected, resetting idle warning for session {session_id}")
                session.idle_warning_sent = False
                warning_reset = True
            
            if session.session_warning_sent:
                logger.info(f"User activity detected, resetting session warning for session {session_id}")
                session.session_warning_sent = False
                warning_reset = True
            
            # Notify client that activity was detected and warnings are reset
            if warning_reset:
                try:
                    await session.websocket.send_json({"type": "activity_reset"})
                    logger.info(f"Activity reset message sent to session {session_id}")
                except Exception as e:
                    logger.error(f"Failed to send activity reset to session {session_id}: {e}")
            
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

# Global connection counter like in main.py
connection_counter = 0

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
    global connection_counter
    await websocket.accept()
    
    # Extract client IP address
    client_ip = "unknown"
    if websocket.client and hasattr(websocket.client, 'host'):
        client_ip = websocket.client.host
    elif hasattr(websocket, 'headers'):
        # Try to get real IP from headers (for proxy setups)
        client_ip = websocket.headers.get('x-forwarded-for', 
                     websocket.headers.get('x-real-ip', 
                     websocket.headers.get('cf-connecting-ip', 'unknown')))
        if ',' in client_ip:
            client_ip = client_ip.split(',')[0].strip()
    
    # Create session with connection counter like in main.py
    connection_counter += 1
    session_id = str(uuid.uuid4())
    client_id = f"{int(time.time())}_{connection_counter}"
    
    session = UserSession(
        session_id=session_id,
        client_id=client_id,
        websocket=websocket,
        created_at=time.time(),
        status=SessionStatus.QUEUED,
        ip_address=client_ip
    )
    
    logger.info(f"New WebSocket connection: {client_id} from {client_ip}")
    
    # Log new connection analytics
    analytics.log_new_connection(client_id, client_ip)
    
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
                
                # Update activity only for real user inputs, not auto inputs
                if not data.get("is_auto_input", False):
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
            await asyncio.sleep(2)  # Update every 2 seconds for responsive experience
        except Exception as e:
            logger.error(f"Error in periodic queue update: {e}")

# Background task to periodically log analytics summary
async def periodic_analytics_summary():
    while True:
        try:
            await asyncio.sleep(300)  # Log summary every 5 minutes
            analytics.log_periodic_summary()
        except Exception as e:
            logger.error(f"Error in periodic analytics summary: {e}")

# Background task to check worker health
async def periodic_worker_health_check():
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            current_time = time.time()
            disconnected_workers = []
            
            for worker_id, worker in list(session_manager.workers.items()):
                if current_time - worker.last_ping > 30:  # 30 second timeout
                    disconnected_workers.append((worker_id, worker.gpu_id))
            
            for worker_id, gpu_id in disconnected_workers:
                analytics.log_worker_disconnected(worker_id, gpu_id)
                del session_manager.workers[worker_id]
                logger.warning(f"Removed disconnected worker {worker_id} (GPU {gpu_id})")
                
            if disconnected_workers:
                # Log updated GPU status
                total_gpus = len(session_manager.workers)
                active_gpus = len([w for w in session_manager.workers.values() if not w.is_available])
                available_gpus = total_gpus - active_gpus
                analytics.log_gpu_status(total_gpus, active_gpus, available_gpus)
                
        except Exception as e:
            logger.error(f"Error in periodic worker health check: {e}")

@app.on_event("startup")
async def startup_event():
    # Start background tasks
    asyncio.create_task(periodic_queue_update())
    asyncio.create_task(periodic_analytics_summary())
    asyncio.create_task(periodic_worker_health_check())
    
    # Log initial system status
    analytics._write_log("🚀 System initialized and ready to accept connections")
    analytics._write_log("   Waiting for GPU workers to register...")

@app.on_event("shutdown")
async def shutdown_event():
    # Log final system summary
    analytics._write_log("")
    analytics._write_log("🛑 System shutting down...")
    analytics.log_periodic_summary()
    analytics._write_log("System shutdown complete.")

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Dispatcher for Neural OS")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the dispatcher on")
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port) 