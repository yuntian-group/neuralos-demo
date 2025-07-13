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
        
        # File handles for different analytics
        self.log_file = None
        self.gpu_metrics_file = None
        self.connection_events_file = None
        self.queue_metrics_file = None
        self.ip_stats_file = None
        self._init_log_files()
    
    def _init_log_files(self):
        """Initialize all analytics log files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main human-readable log
        self.log_file = f"system_analytics_{timestamp}.log"
        self._write_log("="*80)
        self._write_log("NEURAL OS MULTI-GPU SYSTEM ANALYTICS")
        self._write_log("="*80)
        self._write_log(f"System started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_log("")
        
        # Structured data files for analysis
        self.gpu_metrics_file = f"gpu_metrics_{timestamp}.jsonl"
        self.connection_events_file = f"connection_events_{timestamp}.jsonl"
        self.queue_metrics_file = f"queue_metrics_{timestamp}.jsonl"
        self.ip_stats_file = f"ip_stats_{timestamp}.jsonl"
        
        # Initialize with headers/metadata
        self._write_json_log(self.gpu_metrics_file, {
            "type": "metadata",
            "timestamp": time.time(),
            "description": "GPU utilization metrics",
            "fields": ["timestamp", "total_gpus", "active_gpus", "available_gpus", "utilization_percent"]
        })
        
        self._write_json_log(self.connection_events_file, {
            "type": "metadata", 
            "timestamp": time.time(),
            "description": "Connection lifecycle events",
            "fields": ["timestamp", "event_type", "client_id", "ip_address", "duration", "interactions", "reason"]
        })
        
        self._write_json_log(self.queue_metrics_file, {
            "type": "metadata",
            "timestamp": time.time(), 
            "description": "Queue performance metrics",
            "fields": ["timestamp", "queue_size", "estimated_wait", "bypass_rate", "avg_wait_time"]
        })
        
        self._write_json_log(self.ip_stats_file, {
            "type": "metadata",
            "timestamp": time.time(),
            "description": "IP address usage statistics", 
            "fields": ["timestamp", "ip_address", "connection_count", "total_unique_ips"]
        })
    
    def _write_log(self, message):
        """Write message to main log file and console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")
    
    def _write_json_log(self, filename, data):
        """Write structured data to JSON lines file"""
        with open(filename, "a") as f:
            f.write(json.dumps(data) + "\n")
    
    def log_new_connection(self, client_id: str, ip: str):
        """Log new connection"""
        self.total_connections += 1
        self.active_connections += 1
        self.ip_addresses[ip] += 1
        
        unique_ips = len(self.ip_addresses)
        timestamp = time.time()
        
        # Human-readable log
        self._write_log(f"🔗 NEW CONNECTION: {client_id} from {ip}")
        self._write_log(f"   📊 Total connections: {self.total_connections} | Active: {self.active_connections} | Unique IPs: {unique_ips}")
        
        # Structured data logs
        self._write_json_log(self.connection_events_file, {
            "type": "connection_open",
            "timestamp": timestamp,
            "client_id": client_id,
            "ip_address": ip,
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "unique_ips": unique_ips
        })
        
        self._write_json_log(self.ip_stats_file, {
            "type": "ip_update",
            "timestamp": timestamp,
            "ip_address": ip,
            "connection_count": self.ip_addresses[ip],
            "total_unique_ips": unique_ips
        })
    
    def log_connection_closed(self, client_id: str, duration: float, interactions: int, reason: str = "normal"):
        """Log connection closed"""
        self.active_connections -= 1
        self.total_interactions += interactions
        self.session_durations.append(duration)
        
        avg_duration = sum(self.session_durations) / len(self.session_durations) if self.session_durations else 0
        timestamp = time.time()
        
        # Human-readable log
        self._write_log(f"🚪 CONNECTION CLOSED: {client_id}")
        self._write_log(f"   ⏱️  Duration: {duration:.1f}s | Interactions: {interactions} | Reason: {reason}")
        self._write_log(f"   📊 Active connections: {self.active_connections} | Avg session duration: {avg_duration:.1f}s")
        
        # Structured data log
        self._write_json_log(self.connection_events_file, {
            "type": "connection_close",
            "timestamp": timestamp,
            "client_id": client_id,
            "duration": duration,
            "interactions": interactions,
            "reason": reason,
            "active_connections": self.active_connections,
            "avg_session_duration": avg_duration
        })
    
    def log_queue_bypass(self, client_id: str):
        """Log when user bypasses queue (gets GPU immediately)"""
        self.users_bypassed_queue += 1
        bypass_rate = (self.users_bypassed_queue / self.total_connections) * 100 if self.total_connections > 0 else 0
        timestamp = time.time()
        
        # Human-readable log
        self._write_log(f"⚡ QUEUE BYPASS: {client_id} got GPU immediately")
        self._write_log(f"   📊 Bypass rate: {bypass_rate:.1f}% ({self.users_bypassed_queue}/{self.total_connections})")
        
        # Structured data log
        self._write_json_log(self.queue_metrics_file, {
            "type": "queue_bypass",
            "timestamp": timestamp,
            "client_id": client_id,
            "bypass_rate": bypass_rate,
            "users_bypassed": self.users_bypassed_queue,
            "total_connections": self.total_connections
        })
    
    def log_queue_wait(self, client_id: str, wait_time: float, queue_position: int):
        """Log when user had to wait in queue"""
        self.users_waited_in_queue += 1
        self.waiting_times.append(wait_time)
        
        avg_wait = sum(self.waiting_times) / len(self.waiting_times) if self.waiting_times else 0
        wait_rate = (self.users_waited_in_queue / self.total_connections) * 100 if self.total_connections > 0 else 0
        timestamp = time.time()
        
        # Human-readable log
        self._write_log(f"⏳ QUEUE WAIT: {client_id} waited {wait_time:.1f}s (was #{queue_position})")
        self._write_log(f"   📊 Wait rate: {wait_rate:.1f}% | Avg wait time: {avg_wait:.1f}s")
        
        # Structured data log
        self._write_json_log(self.queue_metrics_file, {
            "type": "queue_wait",
            "timestamp": timestamp,
            "client_id": client_id,
            "wait_time": wait_time,
            "queue_position": queue_position,
            "wait_rate": wait_rate,
            "avg_wait_time": avg_wait,
            "users_waited": self.users_waited_in_queue
        })
    
    def log_gpu_status(self, total_gpus: int, active_gpus: int, available_gpus: int):
        """Log GPU utilization"""
        utilization = (active_gpus / total_gpus) * 100 if total_gpus > 0 else 0
        self.gpu_utilization_samples.append(utilization)
        
        avg_utilization = sum(self.gpu_utilization_samples) / len(self.gpu_utilization_samples) if self.gpu_utilization_samples else 0
        timestamp = time.time()
        
        # Human-readable log
        self._write_log(f"🖥️  GPU STATUS: {active_gpus}/{total_gpus} in use ({utilization:.1f}% utilization)")
        self._write_log(f"   📊 Available: {available_gpus} | Avg utilization: {avg_utilization:.1f}%")
        
        # Structured data log
        self._write_json_log(self.gpu_metrics_file, {
            "type": "gpu_status",
            "timestamp": timestamp,
            "total_gpus": total_gpus,
            "active_gpus": active_gpus,
            "available_gpus": available_gpus,
            "utilization_percent": utilization,
            "avg_utilization_percent": avg_utilization
        })
    
    def log_worker_registered(self, worker_id: str, worker_address: str, endpoint: str):
        """Log when a worker registers"""
        self._write_log(f"⚙️  WORKER REGISTERED: {worker_id} ({worker_address}) at {endpoint}")
    
    def log_worker_disconnected(self, worker_id: str, worker_address: str):
        """Log when a worker disconnects"""
        self._write_log(f"⚙️  WORKER DISCONNECTED: {worker_id} ({worker_address})")
    
    def log_no_workers_available(self, queue_size: int):
        """Log critical situation when no workers are available"""
        self._write_log(f"⚠️  CRITICAL: No GPU workers available! {queue_size} users waiting")
        self._write_log("   Please check worker processes and GPU availability")
    
    def log_queue_limits_applied(self, affected_sessions: int, queue_size: int):
        """Log when time limits are applied to existing sessions due to queue formation"""
        timestamp = time.time()
        
        # Human-readable log
        self._write_log(f"🕐 QUEUE LIMITS APPLIED: {affected_sessions} existing sessions now have 60s limits")
        self._write_log(f"   📊 Reason: Queue formed with {queue_size} waiting users")
        
        # Structured data log
        self._write_json_log(self.queue_metrics_file, {
            "type": "queue_limits_applied",
            "timestamp": timestamp,
            "affected_sessions": affected_sessions,
            "queue_size": queue_size,
            "time_limit_applied": 60.0
        })
    
    def log_queue_status(self, queue_size: int, maximum_wait: float):
        """Log queue status"""
        self.queue_size_samples.append(queue_size)
        
        avg_queue_size = sum(self.queue_size_samples) / len(self.queue_size_samples) if self.queue_size_samples else 0
        timestamp = time.time()
        
        # Always log to structured data for analysis
        self._write_json_log(self.queue_metrics_file, {
            "type": "queue_status",
            "timestamp": timestamp,
            "queue_size": queue_size,
            "maximum_wait": maximum_wait,
            "avg_queue_size": avg_queue_size
        })
        
        # Only log to human-readable if there's a queue
        if queue_size > 0:
            self._write_log(f"📝 QUEUE STATUS: {queue_size} users waiting | Max wait: {maximum_wait:.1f}s")
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
    session_limit_start_time: Optional[float] = None  # When the time limit was first applied
    user_has_interacted: bool = False
    ip_address: Optional[str] = None
    interaction_count: int = 0
    queue_start_time: Optional[float] = None
    idle_warning_sent: bool = False
    session_warning_sent: bool = False

@dataclass
class WorkerInfo:
    worker_id: str
    worker_address: str  # e.g., "localhost:8001", "192.168.1.100:8002"
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
        self._queue_lock = asyncio.Lock()  # Prevent race conditions in queue processing
        
        # Configuration
        self.IDLE_TIMEOUT = 20.0  # When no queue
        self.QUEUE_WARNING_TIME = 10.0
        self.MAX_SESSION_TIME_WITH_QUEUE = 60.0  # When there's a queue
        self.QUEUE_SESSION_WARNING_TIME = 45.0  # 15 seconds before timeout
        self.GRACE_PERIOD = 10.0

    async def register_worker(self, worker_id: str, worker_address: str, endpoint: str):
        """Register a new worker"""
        # Check for duplicate registrations
        if worker_id in self.workers:
            logger.warning(f"Worker {worker_id} already registered! Overwriting previous registration.")
            logger.warning(f"Previous: {self.workers[worker_id].worker_address}, {self.workers[worker_id].endpoint}")
            logger.warning(f"New: {worker_address}, {endpoint}")
        
        self.workers[worker_id] = WorkerInfo(
            worker_id=worker_id,
            worker_address=worker_address,
            endpoint=endpoint,
            is_available=True,
            last_ping=time.time()
        )
        logger.info(f"Registered worker {worker_id} ({worker_address}) at {endpoint}")
        logger.info(f"Total workers now: {len(self.workers)} - {[w.worker_id for w in self.workers.values()]}")
        
        # Log worker registration
        analytics.log_worker_registered(worker_id, worker_address, endpoint)
        
        # Log GPU status
        total_gpus = len(self.workers)
        active_gpus = len([w for w in self.workers.values() if not w.is_available])
        available_gpus = total_gpus - active_gpus
        analytics.log_gpu_status(total_gpus, active_gpus, available_gpus)

    async def get_available_worker(self) -> Optional[WorkerInfo]:
        """Get an available worker"""
        for worker in self.workers.values():
            if worker.is_available and time.time() - worker.last_ping < 20:  # Worker ping timeout
                return worker
        return None

    async def add_session_to_queue(self, session: UserSession):
        """Add a session to the queue"""        
        self.sessions[session.session_id] = session
        self.session_queue.append(session.session_id)
        session.status = SessionStatus.QUEUED
        session.queue_start_time = time.time()
        logger.info(f"Added session {session.session_id} to queue. Queue size: {len(self.session_queue)}")
        
        # Don't apply time limits here - wait until after processing queue
        # to see if users actually have to wait

    async def apply_queue_limits_to_existing_sessions(self):
        """Apply 60-second time limits to existing unlimited sessions when queue forms"""
        current_time = time.time()
        affected_sessions = 0
        
        for session_id in list(self.active_sessions.keys()):
            session = self.sessions.get(session_id)
            if session and session.max_session_time is None:  # Currently unlimited
                # Give them 60 seconds from now
                session.max_session_time = 60.0
                session.session_limit_start_time = current_time  # Track when limit started
                # Don't reset last_activity - that would break idle timeout tracking
                session.session_warning_sent = False  # Reset warning flag
                affected_sessions += 1
                
                # Notify the user about the new time limit
                try:
                    queue_size = len(self.session_queue)
                    
                    await session.websocket.send_json({
                        "type": "queue_limit_applied",
                        "time_remaining": 60.0,
                        "queue_size": queue_size
                    })
                    logger.info(f"Applied 60s time limit to existing session {session_id} due to queue formation ({queue_size} users waiting)")
                except Exception as e:
                    logger.error(f"Failed to notify session {session_id} about queue limit: {e}")
        
        if affected_sessions > 0:
            analytics.log_queue_limits_applied(affected_sessions, len(self.session_queue))

    async def remove_time_limits_if_queue_empty(self):
        """Remove time limits from active sessions when queue becomes empty"""
        if len(self.session_queue) > 0:
            return  # Queue not empty, don't remove limits
            
        removed_limits = 0
        for session_id in list(self.active_sessions.keys()):
            session = self.sessions.get(session_id)
            if session and session.max_session_time is not None:
                # Remove time limit
                session.max_session_time = None
                session.session_limit_start_time = None
                session.session_warning_sent = False
                removed_limits += 1
                
                # Notify user that time limit was removed
                try:
                    await session.websocket.send_json({
                        "type": "time_limit_removed",
                        "reason": "queue_empty"
                    })
                    logger.info(f"Time limit removed from active session {session_id} (queue empty)")
                except Exception as e:
                    logger.error(f"Failed to notify session {session_id} about time limit removal: {e}")
        
        if removed_limits > 0:
            logger.info(f"Removed time limits from {removed_limits} active sessions (queue became empty)")

    async def process_queue(self):
        """Process the session queue"""
        async with self._queue_lock:  # Prevent race conditions
            # Track if we had any existing active sessions before processing
            had_active_sessions = len(self.active_sessions) > 0
            
            # Add detailed logging for debugging
            logger.info(f"Processing queue: {len(self.session_queue)} waiting, {len(self.active_sessions)} active")
            logger.info(f"Available workers: {[f'{w.worker_id}({w.worker_address})' for w in self.workers.values() if w.is_available]}")
            logger.info(f"Busy workers: {[f'{w.worker_id}({w.worker_address})' for w in self.workers.values() if not w.is_available]}")
            
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
                    logger.info(f"No available workers for session {session_id}. Queue processing stopped.")
                    break  # No available workers
                    
                # Calculate wait time
                wait_time = time.time() - session.queue_start_time if session.queue_start_time else 0
                queue_position = self.session_queue.index(session_id) + 1
                
                # Assign session to worker
                self.session_queue.pop(0)
                session.status = SessionStatus.ACTIVE
                session.worker_id = worker.worker_id
                session.last_activity = time.time()
                
                # Set session time limit based on queue status AFTER processing
                if len(self.session_queue) > 0:
                    session.max_session_time = self.MAX_SESSION_TIME_WITH_QUEUE
                    session.session_limit_start_time = time.time()  # Track when limit started
                
                worker.is_available = False
                worker.current_session = session_id
                self.active_sessions[session_id] = worker.worker_id
                
                logger.info(f"Assigned session {session_id} to worker {worker.worker_id}")
                logger.info(f"Active sessions now: {len(self.active_sessions)}, Available workers: {len([w for w in self.workers.values() if w.is_available])}")
                
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
            
            # After processing queue, if there are still users waiting AND we had existing active sessions,
            # apply time limits to those existing sessions
            if len(self.session_queue) > 0 and had_active_sessions:
                await self.apply_queue_limits_to_existing_sessions()
            
            # If queue became empty and there are active sessions with time limits, remove them
            elif len(self.session_queue) == 0:
                await self.remove_time_limits_if_queue_empty()

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
                
                # Check timeouts - both session limit AND idle timeout can apply
                session_timeout = False
                idle_timeout = False
                
                # Check session time limit (when queue exists)
                if session.max_session_time:
                    # Use session_limit_start_time for absolute timeout, fall back to last_activity if not set
                    start_time = session.session_limit_start_time if session.session_limit_start_time else session.last_activity
                    elapsed = current_time - start_time if start_time else 0
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
                            session.session_limit_start_time = None  # Clear time limit start time
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
                    
                    # Session timeout
                    elif remaining <= 0:
                        session_timeout = True
                
                # Check idle timeout (always check when user has interacted)
                if session.last_activity:
                    idle_time = current_time - session.last_activity
                    if idle_time >= self.IDLE_TIMEOUT:
                        idle_timeout = True
                    elif idle_time >= self.QUEUE_WARNING_TIME and not session.idle_warning_sent:
                        # Send idle warning (even when session limit exists)
                        await session.websocket.send_json({
                            "type": "idle_warning",
                            "time_remaining": self.IDLE_TIMEOUT - idle_time
                        })
                        session.idle_warning_sent = True
                        reason = "with session limit" if session.max_session_time else "no session limit"
                        logger.info(f"Idle warning sent to {session_id}, time remaining: {self.IDLE_TIMEOUT - idle_time:.1f}s ({reason})")
                
                # End session if either timeout triggered
                if session_timeout or idle_timeout:
                    timeout_reason = "session_limit" if session_timeout else "idle"
                    logger.info(f"Ending session {session_id} due to {timeout_reason} timeout")
                    await self.end_session(session_id, SessionStatus.TIMEOUT)
                    return
                
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
            if not worker.is_available:  # Only log if worker was actually busy
                logger.info(f"Freeing worker {worker.worker_id} from session {session_id}")
            else:
                logger.warning(f"Worker {worker.worker_id} was already available when freeing from session {session_id}")
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
        
        # Validate system state consistency
        await self._validate_system_state()
        
        # Process next in queue
        asyncio.create_task(self.process_queue())

    async def update_queue_info(self):
        """Send queue information to waiting users"""
        for i, session_id in enumerate(self.session_queue):
            session = self.sessions.get(session_id)
            if session and session.status == SessionStatus.QUEUED:
                try:
                    # Calculate maximum possible wait time
                    maximum_wait = self._calculate_maximum_wait_time(i + 1)
                    
                    await session.websocket.send_json({
                        "type": "queue_update",
                        "position": i + 1,
                        "total_waiting": len(self.session_queue),
                        "maximum_wait_seconds": maximum_wait,
                        "active_sessions": len(self.active_sessions),
                        "available_workers": len([w for w in self.workers.values() if w.is_available])
                    })
                except Exception as e:
                    logger.error(f"Failed to send queue update to session {session_id}: {e}")
        
        # Log queue status if there's a queue
        if self.session_queue:
            maximum_wait = self._calculate_maximum_wait_time(1)
            analytics.log_queue_status(len(self.session_queue), maximum_wait)

    def _calculate_maximum_wait_time(self, position_in_queue: int) -> float:
        """Calculate realistic wait time based on actual session remaining times"""
        available_workers = len([w for w in self.workers.values() if w.is_available])
        
        # If there are available workers, no wait time
        if available_workers > 0:
            return 0
        
        # Calculate wait time based on position and worker count
        num_workers = len(self.workers)
        if num_workers == 0:
            return 999  # No workers available
        
        # Get actual remaining times for active sessions
        current_time = time.time()
        active_session_remaining_times = []
        
        for session_id in self.active_sessions:
            session = self.sessions.get(session_id)
            if session:
                # Only consider session time limit (hard limit), not idle timeout
                # Users can reset idle timeout by moving mouse, but session limit is fixed
                if session.max_session_time and session.session_limit_start_time:
                    elapsed = current_time - session.session_limit_start_time
                    remaining_time = max(0, session.max_session_time - elapsed)
                else:
                    # No session limit set (shouldn't happen when queue exists, but fallback)
                    remaining_time = self.MAX_SESSION_TIME_WITH_QUEUE
                
                active_session_remaining_times.append(remaining_time)
        
        # Sort remaining times (shortest first)
        active_session_remaining_times.sort()
        
        # Calculate when this user will get a worker
        if position_in_queue <= len(active_session_remaining_times):
            # User will get a worker when the Nth session ends
            estimated_wait = active_session_remaining_times[position_in_queue - 1]
            logger.info(f"Queue position {position_in_queue}: Will get worker when session #{position_in_queue} ends in {estimated_wait:.1f}s")
            logger.info(f"Active session remaining times: {[f'{t:.1f}s' for t in active_session_remaining_times]}")
        else:
            # More people in queue than active sessions
            # Need to wait for multiple "waves" to complete
            full_waves = (position_in_queue - 1) // num_workers
            position_in_final_wave = (position_in_queue - 1) % num_workers
            
            # Wait for current sessions to end, then full waves, then partial wave
            if active_session_remaining_times:
                first_wave_time = max(active_session_remaining_times)
            else:
                first_wave_time = self.MAX_SESSION_TIME_WITH_QUEUE
            
            additional_wave_time = full_waves * self.MAX_SESSION_TIME_WITH_QUEUE
            estimated_wait = first_wave_time + additional_wave_time
            logger.info(f"Queue position {position_in_queue}: Need {full_waves} full waves + current sessions. Wait: {estimated_wait:.1f}s")
        
        return estimated_wait

    async def handle_user_activity(self, session_id: str):
        """Update user activity timestamp and reset warning flags"""
        session = self.sessions.get(session_id)
        if session:
            old_time = session.last_activity
            session.interaction_count += 1
            
            # Always update last_activity for idle detection
            # Session time limits are now tracked separately via session_limit_start_time
            session.last_activity = time.time()
            
            if session.max_session_time is not None:
                logger.info(f"Activity detected for session {session_id} - idle timer reset (session limit still active)")
            else:
                logger.info(f"Activity detected for session {session_id} - idle timer reset (no session limit)")
            
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
                    # Include current session time remaining if session limit is active
                    message = {"type": "activity_reset"}
                    if session.max_session_time and session.session_limit_start_time:
                        elapsed = time.time() - session.session_limit_start_time
                        remaining = max(0, session.max_session_time - elapsed)
                        if remaining > 0:
                            message["session_time_remaining"] = remaining
                            message["queue_size"] = len(self.session_queue)
                    
                    await session.websocket.send_json(message)
                    logger.info(f"Activity reset message sent to session {session_id}")
                except Exception as e:
                    logger.error(f"Failed to send activity reset to session {session_id}: {e}")
            
            if not session.user_has_interacted:
                session.user_has_interacted = True
                logger.info(f"User started interacting in session {session_id}")

    async def _validate_system_state(self):
        """Validate system state consistency for debugging"""
        try:
            # Count active sessions
            active_sessions_count = len(self.active_sessions)
            busy_workers_count = len([w for w in self.workers.values() if not w.is_available])
            
            # Check for inconsistencies
            if active_sessions_count != busy_workers_count:
                logger.error(f"INCONSISTENCY: Active sessions ({active_sessions_count}) != Busy workers ({busy_workers_count})")
                logger.error(f"Active sessions: {list(self.active_sessions.keys())}")
                logger.error(f"Busy workers: {[w.worker_id for w in self.workers.values() if not w.is_available]}")
                
                # Log detailed state
                for session_id, worker_id in self.active_sessions.items():
                    session = self.sessions.get(session_id)
                    worker = self.workers.get(worker_id)
                    logger.error(f"Session {session_id}: status={session.status if session else 'MISSING'}, worker={worker_id}")
                    if worker:
                        logger.error(f"Worker {worker_id}: available={worker.is_available}, current_session={worker.current_session}")
                
            # Check for orphaned workers
            for worker in self.workers.values():
                if not worker.is_available and worker.current_session not in self.active_sessions:
                    logger.error(f"ORPHANED WORKER: {worker.worker_id} is busy but session {worker.current_session} not in active_sessions")
                    
            # Check for sessions without workers
            for session_id in self.active_sessions:
                session = self.sessions.get(session_id)
                if session and session.status == SessionStatus.ACTIVE:
                    worker = self.workers.get(session.worker_id)
                    if not worker or worker.is_available:
                        logger.error(f"ACTIVE SESSION WITHOUT WORKER: {session_id} has no busy worker assigned")
                        
        except Exception as e:
            logger.error(f"Error in system state validation: {e}")

    async def _handle_worker_failure(self, failed_worker_id: str):
        """Handle sessions when a worker fails - end sessions and put users back in queue"""
        logger.warning(f"Handling failure of worker {failed_worker_id}")
        
        # Find all sessions assigned to this worker
        failed_sessions = []
        for session_id, worker_id in list(self.active_sessions.items()):
            if worker_id == failed_worker_id:
                failed_sessions.append(session_id)
        
        logger.warning(f"Found {len(failed_sessions)} sessions on failed worker {failed_worker_id}")
        
        for session_id in failed_sessions:
            session = self.sessions.get(session_id)
            if session:
                logger.info(f"Recovering session {session_id} from failed worker")
                
                # Notify user about the worker failure
                try:
                    await session.websocket.send_json({
                        "type": "worker_failure",
                        "message": "GPU worker failed. Reconnecting you to a healthy worker..."
                    })
                except Exception as e:
                    logger.error(f"Failed to notify session {session_id} about worker failure: {e}")
                
                # Remove from active sessions
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
                
                # Reset session state and put back in queue
                session.status = SessionStatus.QUEUED
                session.worker_id = None
                session.queue_start_time = time.time()
                session.max_session_time = None  # Reset time limits
                session.session_limit_start_time = None
                session.session_warning_sent = False
                session.idle_warning_sent = False
                
                # Add back to front of queue (they were already active)
                if session_id not in self.session_queue:
                    self.session_queue.insert(0, session_id)
                    logger.info(f"Added session {session_id} to front of queue for recovery")
        
        # Process queue to reassign recovered sessions to healthy workers
        if failed_sessions:
            logger.info(f"Processing queue to reassign {len(failed_sessions)} recovered sessions")
            await self.process_queue()

    async def _forward_to_worker(self, worker: WorkerInfo, session_id: str, data: dict):
        """Forward input to worker asynchronously"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as client_session:
                async with client_session.post(
                    f"{worker.endpoint}/process_input",
                    json={
                        "session_id": session_id,
                        "data": data
                    }
                ) as response:
                    if response.status != 200:
                        logger.error(f"Worker {worker.worker_id} returned status {response.status}")
        except asyncio.TimeoutError:
            logger.error(f"Worker {worker.worker_id} timeout - may be unresponsive")
            # Mark worker as potentially dead for faster detection
            worker.last_ping = 0  # This will cause it to be removed on next health check
        except Exception as e:
            logger.error(f"Error forwarding to worker {worker.worker_id}: {e}")
            # Mark worker as potentially dead for faster detection  
            worker.last_ping = 0

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
    logger.info(f"📥 Received worker registration request")
    logger.info(f"📊 Worker info: {worker_info}")
    
    try:
        await session_manager.register_worker(
            worker_info["worker_id"],
            worker_info["worker_address"], 
            worker_info["endpoint"]
        )
        logger.info(f"✅ Successfully processed worker registration")
        return {"status": "registered"}
    except Exception as e:
        logger.error(f"❌ Failed to register worker: {e}")
        import traceback
        logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
        raise

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
                    # Log stay-connected attempts for monitoring
                    if data.get("is_stay_connected", False):
                        logger.info(f"Stay-connected ping from session {session_id} - idle timer will be reset")
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

# Background task to periodically validate system state
async def periodic_system_validation():
    while True:
        try:
            await asyncio.sleep(10)  # Validate every 10 seconds
            await session_manager._validate_system_state()
        except Exception as e:
            logger.error(f"Error in periodic system validation: {e}")

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
            await asyncio.sleep(15)  # Check every 15 seconds
            current_time = time.time()
            disconnected_workers = []
            
            for worker_id, worker in list(session_manager.workers.items()):
                if current_time - worker.last_ping > 20:  # 20 second timeout
                    disconnected_workers.append((worker_id, worker.worker_address))
            
            for worker_id, worker_address in disconnected_workers:
                analytics.log_worker_disconnected(worker_id, worker_address)
                
                # Handle any active sessions on this dead worker
                await session_manager._handle_worker_failure(worker_id)
                
                del session_manager.workers[worker_id]
                logger.warning(f"Removed disconnected worker {worker_id} ({worker_address})")
                
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
    logger.info("🚀 Dispatcher startup event triggered")
    
    # Start background tasks
    asyncio.create_task(periodic_queue_update())
    asyncio.create_task(periodic_system_validation())
    asyncio.create_task(periodic_analytics_summary())
    asyncio.create_task(periodic_worker_health_check())
    
    # Log initial system status
    analytics._write_log("🚀 System initialized and ready to accept connections")
    analytics._write_log("   Waiting for GPU workers to register...")
    
    logger.info("✅ Dispatcher startup complete - ready to accept worker registrations")

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
    parser.add_argument("--port", type=int, default=7860, help="Port to run the dispatcher on")
    args = parser.parse_args()
    
    logger.info(f"🌐 Starting dispatcher on 0.0.0.0:{args.port}")
    logger.info(f"🔗 Dispatcher will be available at http://localhost:{args.port}")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    except Exception as e:
        logger.error(f"❌ Failed to start dispatcher: {e}")
        import traceback
        logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
        raise 