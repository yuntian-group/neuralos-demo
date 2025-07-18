from fastapi import FastAPI, HTTPException
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageDraw
import base64
import io
import json
import asyncio
import time
import torch
import os
import logging
from utils import initialize_model, sample_frame
from ldm.models.diffusion.ddpm import LatentDiffusion, DDIMSampler
import concurrent.futures
import aiohttp
import argparse
import uuid
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class GPUWorker:
    def __init__(self, worker_address: str, dispatcher_url: str = "http://localhost:7860"):
        self.worker_address = worker_address  # e.g., "localhost:8001", "192.168.1.100:8002"
        # Parse port from worker address
        if ':' in worker_address:
            self.host, port_str = worker_address.split(':')
            self.port = int(port_str)
        else:
            raise ValueError(f"Invalid worker address format: {worker_address}. Expected format: 'host:port'")
        
        self.dispatcher_url = dispatcher_url
        self.worker_id = f"worker_{worker_address.replace(':', '_')}_{uuid.uuid4().hex[:8]}"
        # Always use GPU 0 since CUDA_VISIBLE_DEVICES limits visibility to one GPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.current_session: Optional[str] = None
        self.session_data: Dict[str, Any] = {}
        
        # Model configuration from main.py
        self.DEBUG_MODE = False
        self.DEBUG_MODE_2 = False
        self.NUM_MAX_FRAMES = 1
        self.TIMESTEPS = 1000
        self.SCREEN_WIDTH = 512
        self.SCREEN_HEIGHT = 384
        self.NUM_SAMPLING_STEPS = 32
        self.USE_RNN = False
        
        self.MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-222222k72-108k"
        self.MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-222222k722-130k"
        
        # Initialize model
        self._initialize_model()
        
        # Thread executor for heavy computation
        self.thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        # Load keyboard mappings
        self._load_keyboard_mappings()
        
        logger.info(f"GPU Worker {self.worker_id} initialized for {self.worker_address} on port {self.port}")

    def _initialize_model(self):
        """Initialize the model on the GPU"""
        logger.info(f"Initializing model for worker {self.worker_address}")
        
        # Log CUDA environment info
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        logger.info(f"Available CUDA devices: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            logger.info(f"Device name: {torch.cuda.get_device_name(0)}")  # Always GPU 0
        
        # Load latent stats
        with open('latent_stats.json', 'r') as f:
            latent_stats = json.load(f)
        self.DATA_NORMALIZATION = {
            'mean': torch.tensor(latent_stats['mean']).to(self.device), 
            'std': torch.tensor(latent_stats['std']).to(self.device)
        }
        self.LATENT_DIMS = (16, self.SCREEN_HEIGHT // 8, self.SCREEN_WIDTH // 8)
        
        # Initialize model based on model name
        if 'origunet' in self.MODEL_NAME:
            if 'x0' in self.MODEL_NAME:
                if 'ddpm32' in self.MODEL_NAME:
                    self.TIMESTEPS = 32
                    self.model = initialize_model("config_final_model_origunet_nospatial_x0_ddpm32.yaml", self.MODEL_NAME)
                else:
                    self.model = initialize_model("config_final_model_origunet_nospatial_x0.yaml", self.MODEL_NAME)
            else:
                if 'ddpm32' in self.MODEL_NAME:
                    self.TIMESTEPS = 32
                    self.model = initialize_model("config_final_model_origunet_nospatial_ddpm32.yaml", self.MODEL_NAME)
                else:
                    self.model = initialize_model("config_final_model_origunet_nospatial.yaml", self.MODEL_NAME)
        else:
            self.model = initialize_model("config_final_model.yaml", self.MODEL_NAME)
        
        self.model = self.model.to(self.device)
        
        # Create padding image
        self.padding_image = torch.zeros(*self.LATENT_DIMS).unsqueeze(0).to(self.device)
        self.padding_image = (self.padding_image - self.DATA_NORMALIZATION['mean'].view(1, -1, 1, 1)) / self.DATA_NORMALIZATION['std'].view(1, -1, 1, 1)
        
        logger.info(f"Model initialized successfully for worker {self.worker_address}")

    def _load_keyboard_mappings(self):
        """Load keyboard mappings from main.py"""
        self.KEYS = ['\t', '\n', '\r', ' ', '!', '"', '#', '$', '%', '&', "'", '(',
                    ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
                    '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`',
                    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~',
                    'accept', 'add', 'alt', 'altleft', 'altright', 'apps', 'backspace',
                    'browserback', 'browserfavorites', 'browserforward', 'browserhome',
                    'browserrefresh', 'browsersearch', 'browserstop', 'capslock', 'clear',
                    'convert', 'ctrl', 'ctrlleft', 'ctrlright', 'decimal', 'del', 'delete',
                    'divide', 'down', 'end', 'enter', 'esc', 'escape', 'execute', 'f1', 'f10',
                    'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f2', 'f20',
                    'f21', 'f22', 'f23', 'f24', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
                    'final', 'fn', 'hanguel', 'hangul', 'hanja', 'help', 'home', 'insert', 'junja',
                    'kana', 'kanji', 'launchapp1', 'launchapp2', 'launchmail',
                    'launchmediaselect', 'left', 'modechange', 'multiply', 'nexttrack',
                    'nonconvert', 'num0', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6',
                    'num7', 'num8', 'num9', 'numlock', 'pagedown', 'pageup', 'pause', 'pgdn',
                    'pgup', 'playpause', 'prevtrack', 'print', 'printscreen', 'prntscrn',
                    'prtsc', 'prtscr', 'return', 'right', 'scrolllock', 'select', 'separator',
                    'shift', 'shiftleft', 'shiftright', 'sleep', 'space', 'stop', 'subtract', 'tab',
                    'up', 'volumedown', 'volumemute', 'volumeup', 'win', 'winleft', 'winright', 'yen',
                    'command', 'option', 'optionleft', 'optionright']

        self.KEYMAPPING = {
            'arrowup': 'up',
            'arrowdown': 'down',
            'arrowleft': 'left',
            'arrowright': 'right',
            'meta': 'command',
            'contextmenu': 'apps',
            'control': 'ctrl',
        }

        self.INVALID_KEYS = ['f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20',
                            'f21', 'f22', 'f23', 'f24', 'select', 'separator', 'execute']
        self.VALID_KEYS = [key for key in self.KEYS if key not in self.INVALID_KEYS]
        self.itos = self.VALID_KEYS
        self.stoi = {key: i for i, key in enumerate(self.itos)}

    async def register_with_dispatcher(self):
        """Register this worker with the dispatcher"""
        logger.info(f"🔗 Attempting to register with dispatcher at {self.dispatcher_url}")
        logger.info(f"📊 Worker details: ID={self.worker_id}, Address={self.worker_address}")
        
        # Test basic connectivity first
        logger.info(f"🧪 Testing basic connectivity to dispatcher...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.dispatcher_url}/") as response:
                    logger.info(f"🌐 Connectivity test successful - dispatcher responded with status {response.status}")
        except Exception as e:
            logger.error(f"❌ Connectivity test FAILED: {e}")
            logger.error(f"🔍 This means the dispatcher is not reachable at {self.dispatcher_url}")
            raise
        
        try:
            registration_data = {
                "worker_id": self.worker_id,
                "worker_address": self.worker_address,
                "endpoint": f"http://{self.worker_address}"
            }
            logger.info(f"📤 Sending registration data: {registration_data}")
            
            async with aiohttp.ClientSession() as session:
                logger.info(f"🌐 Making POST request to {self.dispatcher_url}/register_worker")
                
                async with session.post(f"{self.dispatcher_url}/register_worker", json=registration_data) as response:
                    logger.info(f"📥 Dispatcher response status: {response.status}")
                    response_text = await response.text()
                    logger.info(f"📥 Dispatcher response body: {response_text}")
                    
                    if response.status == 200:
                        logger.info(f"✅ Successfully registered worker {self.worker_id} ({self.worker_address}) with dispatcher")
                    else:
                        logger.error(f"❌ Dispatcher returned error status {response.status}: {response_text}")
                        
        except Exception as e:
            logger.error(f"❌ Failed to register with dispatcher: {e}")
            logger.error(f"🔍 Exception type: {type(e)}")
            logger.error(f"🔍 Dispatcher URL: {self.dispatcher_url}")
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")

    async def ping_dispatcher(self):
        """Periodically ping the dispatcher to maintain connection"""
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(f"{self.dispatcher_url}/worker_ping", json={
                        "worker_id": self.worker_id,
                        "is_available": self.current_session is None
                    })
                await asyncio.sleep(10)  # Ping every 10 seconds
            except Exception as e:
                logger.error(f"Failed to ping dispatcher: {e}")
                await asyncio.sleep(5)  # Retry after 5 seconds on error

    def prepare_model_inputs(
        self, 
        previous_frame: torch.Tensor,
        hidden_states: Any,
        x: int,
        y: int,
        right_click: bool,
        left_click: bool,
        keys_down: List[str],
        time_step: int
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the model (from main.py)"""
        # Clamp coordinates to valid ranges
        x = min(max(0, x), self.SCREEN_WIDTH - 1) if x is not None else 0
        y = min(max(0, y), self.SCREEN_HEIGHT - 1) if y is not None else 0
        
        if self.DEBUG_MODE:
            logger.info('DEBUG MODE, SETTING TIME STEP TO 0')
            time_step = 0
        if self.DEBUG_MODE_2:
            if time_step > self.NUM_MAX_FRAMES-1:
                logger.info('DEBUG MODE_2, SETTING TIME STEP TO 0')
                time_step = 0
        
        inputs = {
            'image_features': previous_frame.to(self.device),
            'is_padding': torch.BoolTensor([time_step == 0]).to(self.device),
            'x': torch.LongTensor([x]).unsqueeze(0).to(self.device),
            'y': torch.LongTensor([y]).unsqueeze(0).to(self.device),
            'is_leftclick': torch.BoolTensor([left_click]).unsqueeze(0).to(self.device),
            'is_rightclick': torch.BoolTensor([right_click]).unsqueeze(0).to(self.device),
            'key_events': torch.zeros(len(self.itos), dtype=torch.long).to(self.device)
        }
        
        for key in keys_down:
            key = key.lower()
            if key in self.KEYMAPPING:
                key = self.KEYMAPPING[key]
            if key in self.stoi:
                inputs['key_events'][self.stoi[key]] = 1
            else:
                logger.warning(f'Key {key} not found in stoi')
        
        if hidden_states is not None:
            inputs['hidden_states'] = hidden_states
            
        if self.DEBUG_MODE:
            logger.info('DEBUG MODE, REMOVING INPUTS')
            if 'hidden_states' in inputs:
                del inputs['hidden_states']
                
        if self.DEBUG_MODE_2:
            if time_step > self.NUM_MAX_FRAMES-1:
                logger.info('DEBUG MODE_2, REMOVING HIDDEN STATES')
                if 'hidden_states' in inputs:
                    del inputs['hidden_states']
                    
        logger.info(f'Time step: {time_step}')
        return inputs

    @torch.no_grad()
    async def process_frame(
        self,
        inputs: Dict[str, torch.Tensor],
        use_rnn: bool = False,
        num_sampling_steps: int = 32
    ) -> Tuple[torch.Tensor, np.ndarray, Any, Dict[str, float]]:
        """Process a single frame through the model"""
        # Run the heavy computation in a separate thread
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.thread_executor,
            lambda: self._process_frame_sync(inputs, use_rnn, num_sampling_steps)
        )

    def _process_frame_sync(self, inputs, use_rnn, num_sampling_steps):
        """Synchronous version of process_frame that runs in a thread"""
        timing = {}
        
        # Temporal encoding
        start = time.perf_counter()
        output_from_rnn, hidden_states = self.model.temporal_encoder.forward_step(inputs)
        timing['temporal_encoder'] = time.perf_counter() - start
        
        # UNet sampling
        start = time.perf_counter()
        logger.info(f"model.clip_denoised: {self.model.clip_denoised}")
        self.model.clip_denoised = False
        logger.info(f"USE_RNN: {use_rnn}, NUM_SAMPLING_STEPS: {num_sampling_steps}")
        
        if use_rnn:
            sample_latent = output_from_rnn[:, :16]
        else:
            if num_sampling_steps >= self.TIMESTEPS:
                sample_latent = self.model.p_sample_loop(
                    cond={'c_concat': output_from_rnn}, 
                    shape=[1, *self.LATENT_DIMS], 
                    return_intermediates=False, 
                    verbose=True
                )
            else:
                if num_sampling_steps == 1:
                    x = torch.randn([1, *self.LATENT_DIMS], device=self.device)
                    t = torch.full((1,), self.TIMESTEPS-1, device=self.device, dtype=torch.long)
                    sample_latent = self.model.apply_model(x, t, {'c_concat': output_from_rnn})
                else:
                    sampler = DDIMSampler(self.model)
                    sample_latent, _ = sampler.sample(
                        S=num_sampling_steps,
                        conditioning={'c_concat': output_from_rnn},
                        batch_size=1,
                        shape=self.LATENT_DIMS,
                        verbose=False
                    )
        timing['unet'] = time.perf_counter() - start
        
        # Decoding
        start = time.perf_counter()
        sample = sample_latent * self.DATA_NORMALIZATION['std'].view(1, -1, 1, 1) + self.DATA_NORMALIZATION['mean'].view(1, -1, 1, 1)
        sample = self.model.decode_first_stage(sample)
        sample = sample.squeeze(0).clamp(-1, 1)
        timing['decode'] = time.perf_counter() - start
        
        # Convert to image
        sample_img = ((sample[:3].transpose(0,1).transpose(1,2).cpu().float().numpy() + 1) * 127.5).astype(np.uint8)
        
        timing['total'] = sum(timing.values())
        
        return sample_latent, sample_img, hidden_states, timing

    def initialize_session(self, session_id: str, client_id: str = None):
        """Initialize a new session"""
        self.current_session = session_id
        # Use client_id from dispatcher if provided, otherwise create one
        if client_id:
            log_session_id = client_id
        else:
            # Fallback: create a time-prefixed session identifier for logging
            session_start_time = int(time.time())
            log_session_id = f"{session_start_time}_{session_id}"
        
        self.session_data[session_id] = {
            'previous_frame': self.padding_image,
            'hidden_states': None,
            'keys_down': set(),
            'frame_num': -1,
            'client_settings': {
                'use_rnn': self.USE_RNN,
                'sampling_steps': self.NUM_SAMPLING_STEPS
            },
            'input_queue': asyncio.Queue(),
            'is_processing': False,
            'log_session_id': log_session_id  # Store the time-prefixed ID for logging
        }
        logger.info(f"Initialized session {session_id} with log ID {log_session_id}")
        
        # Start processing task for this session
        asyncio.create_task(self._process_session_queue(session_id))

    def end_session(self, session_id: str):
        """End a session and clean up"""
        if session_id in self.session_data:
            # Log session end using the stored log_session_id
            session = self.session_data[session_id]
            log_session_id = session.get('log_session_id', session_id)  # Fallback to session_id if not found
            log_interaction(log_session_id, {}, is_end_of_session=True)
            
            # Clear any remaining items in the queue
            while not session['input_queue'].empty():
                try:
                    session['input_queue'].get_nowait()
                    session['input_queue'].task_done()
                except asyncio.QueueEmpty:
                    break
            del self.session_data[session_id]
        if self.current_session == session_id:
            self.current_session = None
        logger.info(f"Ended session {session_id}")

    async def _process_session_queue(self, session_id: str):
        """Process the input queue for a specific session with interesting input filtering"""
        while session_id in self.session_data:
            try:
                session = self.session_data[session_id]
                input_queue = session['input_queue']
                
                # Wait for input to be available
                if input_queue.empty():
                    await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                    continue
                
                # If already processing, skip
                if session['is_processing']:
                    await asyncio.sleep(0.01)
                    continue
                
                # Set processing flag
                session['is_processing'] = True
                
                try:
                    # Process queue with interesting input filtering
                    await self._process_next_input(session_id)
                finally:
                    session['is_processing'] = False
                    
            except Exception as e:
                logger.error(f"Error in session queue processing for {session_id}: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)  # Prevent tight error loop
        
        logger.info(f"Session queue processor ended for {session_id}")

    async def _process_next_input(self, session_id: str):
        """Process next input with interesting input filtering (from main.py logic)"""
        session = self.session_data[session_id]
        input_queue = session['input_queue']
        
        if input_queue.empty():
            return
        
        queue_size = input_queue.qsize()
        logger.info(f"Processing next input for session {session_id}. Queue size: {queue_size}")
        
        try:
            # Initialize variables to track progress
            skipped = 0
            latest_input = None
            
            # Process the queue one item at a time
            while not input_queue.empty():
                current_input = await input_queue.get()
                input_queue.task_done()
                
                # Always update the latest input
                latest_input = current_input
                
                # Check if this is an interesting event
                CONSIDER_SCROLL = False # TODO: consider scroll in future versions
                if CONSIDER_SCROLL:
                    is_interesting = (current_input.get("is_left_click") or
                                  current_input.get("is_right_click") or
                                  (current_input.get("keys_down") and len(current_input.get("keys_down")) > 0) or
                                  (current_input.get("keys_up") and len(current_input.get("keys_up")) > 0) or
                                  current_input.get("wheel_delta_x", 0) != 0 or
                                  current_input.get("wheel_delta_y", 0) != 0)
                else:
                    is_interesting = (current_input.get("is_left_click") or
                                    current_input.get("is_right_click") or
                                    (current_input.get("keys_down") and len(current_input.get("keys_down")) > 0) or
                                    (current_input.get("keys_up") and len(current_input.get("keys_up")) > 0))
                
                # Process immediately if interesting
                if is_interesting:
                    logger.info(f"Found interesting input for session {session_id} (skipped {skipped} events)")
                    await self._process_single_input(session_id, current_input)
                    return
                
                # Otherwise, continue to the next item
                skipped += 1
                
                # If this is the last item and no interesting inputs were found
                if input_queue.empty():
                    logger.info(f"No interesting inputs for session {session_id}, processing latest movement (skipped {skipped-1} events)")
                    await self._process_single_input(session_id, latest_input)
                    return
                    
        except Exception as e:
            logger.error(f"Error in _process_next_input for session {session_id}: {e}")
            import traceback
            traceback.print_exc()

    async def process_input(self, session_id: str, data: dict) -> dict:
        """Process input for a session - adds to queue or handles control messages"""
        if session_id not in self.session_data:
            self.initialize_session(session_id)  # Fallback initialization without client_id
        
        session = self.session_data[session_id]
        
        # Handle control messages immediately (don't queue these)
        if data.get("type") == "reset":
            logger.info(f"Received reset command for session {session_id}")
            # Log the reset action using the stored log_session_id
            log_session_id = session.get('log_session_id', session_id)  # Fallback to session_id if not found
            log_interaction(log_session_id, data, is_reset=True)
            
            # Clear the queue
            while not session['input_queue'].empty():
                try:
                    session['input_queue'].get_nowait()
                    session['input_queue'].task_done()
                except asyncio.QueueEmpty:
                    break
            session['previous_frame'] = self.padding_image
            session['hidden_states'] = None
            session['keys_down'] = set()
            session['frame_num'] = -1
            return {"type": "reset_confirmed"}
        
        elif data.get("type") == "update_sampling_steps":
            steps = data.get("steps", 32)
            if steps < 1:
                return {"type": "error", "message": "Invalid sampling steps value"}
            session['client_settings']['sampling_steps'] = steps
            logger.info(f"Updated sampling steps to {steps} for session {session_id}")
            return {"type": "steps_updated", "steps": steps}
        
        elif data.get("type") == "update_use_rnn":
            use_rnn = data.get("use_rnn", False)
            session['client_settings']['use_rnn'] = use_rnn
            logger.info(f"Updated USE_RNN to {use_rnn} for session {session_id}")
            return {"type": "rnn_updated", "use_rnn": use_rnn}
        
        elif data.get("type") == "get_settings":
            return {
                "type": "settings",
                "sampling_steps": session['client_settings']['sampling_steps'],
                "use_rnn": session['client_settings']['use_rnn']
            }
        
        elif data.get("type") == "heartbeat":
            return {"type": "heartbeat_response"}
        
        # For regular input data, add to queue and return immediately
        # The actual processing will happen asynchronously in the queue processor
        await session['input_queue'].put(data)
        queue_size = session['input_queue'].qsize()
        logger.info(f"Added input to queue for session {session_id}. Queue size: {queue_size}")
        
        # Return a placeholder response - the real response will be sent via WebSocket
        return {"type": "queued", "queue_size": queue_size}

    async def _process_single_input(self, session_id: str, data: dict):
        """Process a single input for a session (the actual processing logic)"""
        session = self.session_data[session_id]
        
        # Process regular input
        try:
            session['frame_num'] += 1
            
            # Extract input data
            x = max(0, min(data.get("x", 0), self.SCREEN_WIDTH - 1))
            y = max(0, min(data.get("y", 0), self.SCREEN_HEIGHT - 1))
            is_left_click = data.get("is_left_click", False)
            is_right_click = data.get("is_right_click", False)
            keys_down_list = data.get("keys_down", [])
            keys_up_list = data.get("keys_up", [])
            wheel_delta_x = data.get("wheel_delta_x", 0)
            wheel_delta_y = data.get("wheel_delta_y", 0)
            
            # Update keys_down set
            for key in keys_down_list:
                key = key.lower()
                if key in self.KEYMAPPING:
                    key = self.KEYMAPPING[key]
                session['keys_down'].add(key)
            
            for key in keys_up_list:
                key = key.lower()
                if key in self.KEYMAPPING:
                    key = self.KEYMAPPING[key]
                session['keys_down'].discard(key)
            
            # Handle debug modes
            if self.DEBUG_MODE:
                logger.info("DEBUG MODE, REMOVING HIDDEN STATES")
                session['previous_frame'] = self.padding_image
            
            if self.DEBUG_MODE_2:
                if session['frame_num'] > self.NUM_MAX_FRAMES-1:
                    logger.info("DEBUG MODE_2, REMOVING HIDDEN STATES")
                    session['previous_frame'] = self.padding_image
                    session['frame_num'] = 0
            
            # Prepare model inputs
            inputs = self.prepare_model_inputs(
                session['previous_frame'],
                session['hidden_states'],
                x, y, is_right_click, is_left_click,
                list(session['keys_down']),
                session['frame_num']
            )
            
            # Log the input data being processed
            logger.info(f"Processing frame {session['frame_num']} for session {session_id}: "
                       f"pos=({x},{y}), clicks=(L:{is_left_click},R:{is_right_click}), "
                       f"keys_down={keys_down_list}, keys_up={keys_up_list}, "
                       f"wheel=({wheel_delta_x},{wheel_delta_y})")
            
            # Process frame
            sample_latent, sample_img, hidden_states, timing_info = await self.process_frame(
                inputs,
                use_rnn=session['client_settings']['use_rnn'],
                num_sampling_steps=session['client_settings']['sampling_steps']
            )
            
            # Update session state
            session['previous_frame'] = sample_latent
            session['hidden_states'] = hidden_states
            
            # Convert image to base64
            img = Image.fromarray(sample_img)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Log timing
            logger.info(f"Frame {session['frame_num']} processed in {timing_info['total']:.4f}s (FPS: {1.0/timing_info['total']:.2f})")
            
            # Log the interaction using the stored log_session_id
            log_session_id = session.get('log_session_id', session_id)  # Fallback to session_id if not found
            log_interaction(log_session_id, data, generated_frame=sample_img)
            
            # Send result back to dispatcher
            await self._send_result_to_dispatcher(session_id, {"image": img_str})
            
        except Exception as e:
            logger.error(f"Error processing input for session {session_id}: {e}")
            import traceback
            traceback.print_exc()
            await self._send_result_to_dispatcher(session_id, {"type": "error", "message": str(e)})

    async def _send_result_to_dispatcher(self, session_id: str, result: dict):
        """Send processing result back to dispatcher"""
        try:
            async with aiohttp.ClientSession() as client_session:
                await client_session.post(f"{self.dispatcher_url}/worker_result", json={
                    "session_id": session_id,
                    "worker_id": self.worker_id,
                    "result": result
                })
        except Exception as e:
            logger.error(f"Failed to send result to dispatcher: {e}")

# FastAPI app for the worker
app = FastAPI()

# Global worker instance
worker: Optional[GPUWorker] = None

def log_interaction(log_session_id, data, generated_frame=None, is_end_of_session=False, is_reset=False):
    """Log user interaction and optionally the generated frame."""
    timestamp = time.time()
    
    # Create directory structure if it doesn't exist
    os.makedirs("interaction_logs", exist_ok=True)
    
    # Structure the log entry
    log_entry = {
        "timestamp": timestamp,
        "session_id": log_session_id,  # Use the time-prefixed session ID
        "is_eos": is_end_of_session,
        "is_reset": is_reset
    }
    
    # Include type if present (for reset, etc.)
    if data.get("type"):
        log_entry["type"] = data.get("type")
    
    # Only include input data if this isn't just a control message
    if not is_end_of_session and not is_reset:
        log_entry["inputs"] = {
            "x": data.get("x"),
            "y": data.get("y"),
            "is_left_click": data.get("is_left_click"),
            "is_right_click": data.get("is_right_click"),
            "keys_down": data.get("keys_down", []),
            "keys_up": data.get("keys_up", []),
            "wheel_delta_x": data.get("wheel_delta_x", 0),
            "wheel_delta_y": data.get("wheel_delta_y", 0),
            "is_auto_input": data.get("is_auto_input", False)
        }
    else:
        # For EOS/reset records, just include minimal info
        log_entry["inputs"] = None
    
    # Use the time-prefixed session ID for the filename (already includes timestamp)
    session_file = f"interaction_logs/session_{log_session_id}.jsonl"
    with open(session_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    # Optionally save the frame if provided
    if generated_frame is not None and not is_end_of_session and not is_reset:
        frame_dir = f"interaction_logs/frames_{log_session_id}"
        os.makedirs(frame_dir, exist_ok=True)
        frame_file = f"{frame_dir}/{timestamp:.6f}.png"
        # Save the frame as PNG
        Image.fromarray(generated_frame).save(frame_file)

@app.post("/process_input")
async def process_input_endpoint(request: dict):
    """Process input from dispatcher"""
    if not worker:
        raise HTTPException(status_code=500, detail="Worker not initialized")
    
    session_id = request.get("session_id")
    data = request.get("data")
    
    if not session_id or not data:
        raise HTTPException(status_code=400, detail="Missing session_id or data")
    
    result = await worker.process_input(session_id, data)
    return result

@app.post("/init_session")
async def init_session_endpoint(request: dict):
    """Initialize session from dispatcher with client_id"""
    if not worker:
        raise HTTPException(status_code=500, detail="Worker not initialized")
    
    session_id = request.get("session_id")
    client_id = request.get("client_id")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")
    
    worker.initialize_session(session_id, client_id)
    return {"status": "session_initialized"}

@app.post("/end_session")
async def end_session_endpoint(request: dict):
    """End session from dispatcher"""
    if not worker:
        raise HTTPException(status_code=500, detail="Worker not initialized")
    
    session_id = request.get("session_id")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")
    
    worker.end_session(session_id)
    return {"status": "session_ended"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "worker_id": worker.worker_id if worker else None,
        "worker_address": worker.worker_address if worker else None,
        "port": worker.port if worker else None,
        "current_session": worker.current_session if worker else None
    }

async def startup_worker(worker_address: str, dispatcher_url: str):
    """Initialize the worker"""
    logger.info(f"🔧 Initializing worker with address {worker_address}")
    
    global worker
    worker = GPUWorker(worker_address, dispatcher_url)
    logger.info(f"🏗️ Worker object created: {worker.worker_id}")
    
    # Register with dispatcher
    logger.info(f"📞 About to register with dispatcher")
    await worker.register_with_dispatcher()
    logger.info(f"📝 Registration attempt completed")
    
    # Start ping task
    logger.info(f"💓 Starting ping task")
    asyncio.create_task(worker.ping_dispatcher())
    logger.info(f"✅ Worker initialization completed")

if __name__ == "__main__":
    import uvicorn
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="GPU Worker for Neural OS")
    parser.add_argument("--worker-address", type=str, required=True, help="Worker address (e.g., 'localhost:8001', '192.168.1.100:8002')")
    parser.add_argument("--dispatcher-url", type=str, default="http://localhost:7860", help="Dispatcher URL")
    args = parser.parse_args()
    
    # Parse port from worker address for validation
    if ':' not in args.worker_address:
        print(f"Error: Invalid worker address format: {args.worker_address}")
        print("Expected format: 'host:port' (e.g., 'localhost:8001')")
        sys.exit(1)
    
    try:
        host, port_str = args.worker_address.split(':')
        port = int(port_str)
    except ValueError:
        print(f"Error: Invalid port in worker address: {args.worker_address}")
        sys.exit(1)
    
    @app.on_event("startup")
    async def startup_event():
        logger.info(f"🚀 Worker startup event triggered for {args.worker_address}")
        await startup_worker(args.worker_address, args.dispatcher_url)
        logger.info(f"✅ Worker startup complete for {args.worker_address}")
    
    logger.info(f"🌐 Starting worker {args.worker_address} on 0.0.0.0:{port}")
    logger.info(f"🔗 Worker will be available at http://{args.worker_address}")
    logger.info(f"📡 Will register with dispatcher at {args.dispatcher_url}")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        logger.error(f"❌ Failed to start worker: {e}")
        import traceback
        logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
        raise 
