from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw
import base64
import io
import json
import asyncio
from utils import initialize_model, sample_frame
import torch
import os
import time
from typing import Any, Dict
from ldm.models.diffusion.ddpm import LatentDiffusion, DDIMSampler
import concurrent.futures

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DEBUG_MODE = False
DEBUG_MODE_2 = False
NUM_MAX_FRAMES = 1
TIMESTEPS = 1000
SCREEN_WIDTH = 512
SCREEN_HEIGHT = 384
NUM_SAMPLING_STEPS = 32
USE_RNN = False

MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-384k"

MODEL_NAME = "yuntian-deng/computer-model-noss-forsure"

MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-2k"

MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-10k"

MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-54k"



MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-newnewd-unfreezernn-160k"
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-newnewd-freezernn-origunet-nospatial-368k"
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-newnewd-unfreezernn-198k"
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-newnewd-freezernn-origunet-nospatial-674k"
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-newnewd-freezernn-origunet-nospatial-online-74k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-online-70k"
MODEL_NAME = "yuntian-deng/computer-model-ss005-cont-lr2e5-computecanada-newnewd-freezernn-origunet-nospatial-online-x0-46k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-142k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-338k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-ddpm32-x0-140k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-ddpm32-eps-144k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-70k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-joint-onlineonly-eps22-40k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-22-38k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-222-42k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-2222-70k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-222222-48k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-222222k7-06k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-222222k7-114k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-222222k7-136k"
#MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-222222k7-184k"
#MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-222222k7-272k"
#MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-222222k7-272k"
#MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-oo-eps222222k72-270k"
MODEL_NAME = "yuntian-deng/computer-model-s-newnewd-freezernn-origunet-nospatial-online-x0-joint-onlineonly-222222k72-108k"


print (f'setting: DEBUG_MODE: {DEBUG_MODE}, DEBUG_MODE_2: {DEBUG_MODE_2}, NUM_MAX_FRAMES: {NUM_MAX_FRAMES}, NUM_SAMPLING_STEPS: {NUM_SAMPLING_STEPS}, MODEL_NAME: {MODEL_NAME}')

with open('latent_stats.json', 'r') as f:
    latent_stats = json.load(f)
DATA_NORMALIZATION = {'mean': torch.tensor(latent_stats['mean']).to(device), 'std': torch.tensor(latent_stats['std']).to(device)}
LATENT_DIMS = (16, SCREEN_HEIGHT // 8, SCREEN_WIDTH // 8)

# Initialize the model at the start of your application
#model = initialize_model("config_csllm.yaml", "yuntian-deng/computer-model")
#model = initialize_model("config_rnn.yaml", "yuntian-deng/computer-model")
#model = initialize_model("config_final_model.yaml", "yuntian-deng/computer-model-noss")
#model = initialize_model("config_final_model.yaml", "yuntian-deng/computer-model")

if 'origunet' in MODEL_NAME:
    if 'x0' in MODEL_NAME and 'eps' not in MODEL_NAME:
        if 'ddpm32' in MODEL_NAME:
            TIMESTEPS = 32
            model = initialize_model("config_final_model_origunet_nospatial_x0_ddpm32.yaml", MODEL_NAME)
        else:
            model = initialize_model("config_final_model_origunet_nospatial_x0.yaml", MODEL_NAME)
    else:
        if 'ddpm32' in MODEL_NAME:
            TIMESTEPS = 32
            model = initialize_model("config_final_model_origunet_nospatial_ddpm32.yaml", MODEL_NAME)
        else:
            model = initialize_model("config_final_model_origunet_nospatial.yaml", MODEL_NAME)
else:
    model = initialize_model("config_final_model.yaml", MODEL_NAME)


model = model.to(device)
#model = torch.compile(model)
padding_image = torch.zeros(*LATENT_DIMS).unsqueeze(0).to(device)
padding_image = (padding_image - DATA_NORMALIZATION['mean'].view(1, -1, 1, 1)) / DATA_NORMALIZATION['std'].view(1, -1, 1, 1)

# Valid keyboard inputs
KEYS = ['\t', '\n', '\r', ' ', '!', '"', '#', '$', '%', '&', "'", '(',
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

KEYMAPPING = {
    'arrowup': 'up',
    'arrowdown': 'down',
    'arrowleft': 'left',
    'arrowright': 'right',
    'meta': 'command',
    'contextmenu': 'apps',
    'control': 'ctrl',
}

INVALID_KEYS = ['f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20',
                'f21', 'f22', 'f23', 'f24', 'select', 'separator', 'execute']
VALID_KEYS = [key for key in KEYS if key not in INVALID_KEYS]
itos = VALID_KEYS
stoi = {key: i for i, key in enumerate(itos)}

app = FastAPI()

# Mount the static directory to serve HTML, JavaScript, and CSS files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add this at the top with other global variables
connection_counter = 0

# Connection timeout settings
CONNECTION_TIMEOUT = 20 + 1  # 20 seconds timeout plus 1 second grace period
WARNING_TIME = 10 + 1 # 10 seconds warning before timeout plus 1 second grace period

# Create a thread pool executor
thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

def prepare_model_inputs(
    previous_frame: torch.Tensor,
    hidden_states: Any,
    x: int,
    y: int,
    right_click: bool,
    left_click: bool,
    keys_down: List[str],
    stoi: Dict[str, int],
    itos: List[str],
    time_step: int
) -> Dict[str, torch.Tensor]:
    """Prepare inputs for the model."""
    # Clamp coordinates to valid ranges
    x = min(max(0, x), SCREEN_WIDTH - 1) if x is not None else 0
    y = min(max(0, y), SCREEN_HEIGHT - 1) if y is not None else 0
    if DEBUG_MODE:
        print ('DEBUG MODE, SETTING TIME STEP TO 0')
        time_step = 0
    if DEBUG_MODE_2:
        if time_step > NUM_MAX_FRAMES-1:
            print ('DEBUG MODE_2, SETTING TIME STEP TO 0')
            time_step = 0
    
    inputs = {
        'image_features': previous_frame.to(device),
        'is_padding': torch.BoolTensor([time_step == 0]).to(device),
        'x': torch.LongTensor([x]).unsqueeze(0).to(device),
        'y': torch.LongTensor([y]).unsqueeze(0).to(device),
        'is_leftclick': torch.BoolTensor([left_click]).unsqueeze(0).to(device),
        'is_rightclick': torch.BoolTensor([right_click]).unsqueeze(0).to(device),
        'key_events': torch.zeros(len(itos), dtype=torch.long).to(device)
    }
    for key in keys_down:
        key = key.lower()
        if key in KEYMAPPING:
            key = KEYMAPPING[key]
        if key in stoi:
            inputs['key_events'][stoi[key]] = 1
        else:
            print (f'Key {key} not found in stoi')
    
    if hidden_states is not None:
        inputs['hidden_states'] = hidden_states
    if DEBUG_MODE:
        print ('DEBUG MODE, REMOVING INPUTS')
        if 'hidden_states' in inputs:
            del inputs['hidden_states']
    if DEBUG_MODE_2:
        if time_step > NUM_MAX_FRAMES-1:
            print ('DEBUG MODE_2, REMOVING HIDDEN STATES')
            if 'hidden_states' in inputs:
                del inputs['hidden_states']
    print (f'Time step: {time_step}')
    return inputs

@torch.no_grad()
async def process_frame(
    model: LatentDiffusion,
    inputs: Dict[str, torch.Tensor],
    use_rnn: bool = False,
    num_sampling_steps: int = 32
) -> Tuple[torch.Tensor, np.ndarray, Any, Dict[str, float]]:
    """Process a single frame through the model."""
    # Run the heavy computation in a separate thread
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        thread_executor,
        lambda: _process_frame_sync(model, inputs, use_rnn, num_sampling_steps)
    )

def _process_frame_sync(model, inputs, use_rnn, num_sampling_steps):
    """Synchronous version of process_frame that runs in a thread"""
    timing = {}
    # Temporal encoding
    start = time.perf_counter()
    output_from_rnn, hidden_states = model.temporal_encoder.forward_step(inputs)
    timing['temporal_encoder'] = time.perf_counter() - start
    
    # UNet sampling
    start = time.perf_counter()
    print (f"model.clip_denoised: {model.clip_denoised}")
    model.clip_denoised = False
    print (f"USE_RNN: {use_rnn}, NUM_SAMPLING_STEPS: {num_sampling_steps}")
    if use_rnn:
        sample_latent = output_from_rnn[:, :16]
    else:
        #NUM_SAMPLING_STEPS = 8
        if num_sampling_steps >= TIMESTEPS:
            sample_latent = model.p_sample_loop(cond={'c_concat': output_from_rnn}, shape=[1, *LATENT_DIMS], return_intermediates=False, verbose=True)
        else:
            if num_sampling_steps == 1:
                x = torch.randn([1, *LATENT_DIMS], device=device)
                t = torch.full((1,), TIMESTEPS-1, device=device, dtype=torch.long)
                sample_latent = model.apply_model(x, t, {'c_concat': output_from_rnn})
            else:
                sampler = DDIMSampler(model)
                sample_latent, _ = sampler.sample(
                    S=num_sampling_steps,
                    conditioning={'c_concat': output_from_rnn},
                    batch_size=1,
                    shape=LATENT_DIMS,
                    verbose=False
                )
    timing['unet'] = time.perf_counter() - start
    
    # Decoding
    start = time.perf_counter()
    sample = sample_latent * DATA_NORMALIZATION['std'].view(1, -1, 1, 1) + DATA_NORMALIZATION['mean'].view(1, -1, 1, 1)
    
    # Use time.sleep(10) here since it's in a separate thread
    #time.sleep(10)
    
    sample = model.decode_first_stage(sample)
    sample = sample.squeeze(0).clamp(-1, 1)
    timing['decode'] = time.perf_counter() - start
    
    # Convert to image
    sample_img = ((sample[:3].transpose(0,1).transpose(1,2).cpu().float().numpy() + 1) * 127.5).astype(np.uint8)
    
    timing['total'] = sum(timing.values())
    
    return sample_latent, sample_img, hidden_states, timing

def print_timing_stats(timing_info: Dict[str, float], frame_num: int):
    """Print timing statistics for a frame."""
    print(f"\nFrame {frame_num} timing (seconds):")
    for key, value in timing_info.items():
        print(f"  {key.title()}: {value:.4f}")
    print(f"  FPS: {1.0/timing_info['full_frame']:.2f}")

# Serve the index.html file at the root URL
@app.get("/")
async def get():
    return HTMLResponse(open("static/index.html").read())

# WebSocket endpoint for continuous user interaction
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):    
    global connection_counter
    connection_counter += 1
    client_id = f"{int(time.time())}_{connection_counter}"
    print(f"New WebSocket connection: {client_id}")
    await websocket.accept()

    try:
        previous_frame = padding_image
        hidden_states = None
        keys_down = set()  # Initialize as an empty set
        frame_num = -1
        
        # Client-specific settings
        client_settings = {
            "use_rnn": USE_RNN,  # Start with default global value
            "sampling_steps": NUM_SAMPLING_STEPS  # Start with default global value
        }
        
        # Connection timeout tracking
        last_user_activity_time = time.perf_counter()
        timeout_warning_sent = False
        timeout_task = None
        connection_active = True  # Flag to track if connection is still active
        user_has_interacted = False  # Flag to track if user has started interacting
        
        # Start timing for global FPS calculation
        connection_start_time = time.perf_counter()
        frame_count = 0
        
        # Input queue management - use asyncio.Queue instead of a list
        input_queue = asyncio.Queue()
        is_processing = False
        
        # Add a function to reset the simulation
        async def reset_simulation():
            nonlocal previous_frame, hidden_states, keys_down, frame_num, is_processing, input_queue, user_has_interacted
            # Keep the client settings during reset
            temp_client_settings = client_settings.copy()
            
            # Log the reset action
            log_interaction(
                client_id, 
                {"type": "reset"}, 
                is_end_of_session=False,
                is_reset=True  # Add this parameter to the log_interaction function
            )
            
            # Clear the input queue
            while not input_queue.empty():
                try:
                    input_queue.get_nowait()
                    input_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            
            # Reset all state variables
            previous_frame = padding_image
            hidden_states = None
            keys_down = set()
            frame_num = -1
            is_processing = False
            user_has_interacted = False  # Reset user interaction state
            
            # Restore client settings
            client_settings.update(temp_client_settings)
            
            print(f"[{time.perf_counter():.3f}] Simulation reset to initial state (preserved settings: USE_RNN={client_settings['use_rnn']}, SAMPLING_STEPS={client_settings['sampling_steps']})")
            print(f"[{time.perf_counter():.3f}] User interaction state reset - waiting for user to interact again")
            
            # Send confirmation to client
            await websocket.send_json({"type": "reset_confirmed"})
            
            # Also send the current settings to update the UI
            await websocket.send_json({
                "type": "settings",
                "sampling_steps": client_settings["sampling_steps"],
                "use_rnn": client_settings["use_rnn"]
            })
        
        # Add a function to update sampling steps
        async def update_sampling_steps(steps):
            nonlocal client_settings
            
            # Validate the input
            if steps < 1:
                print(f"[{time.perf_counter():.3f}] Invalid sampling steps value: {steps}")
                await websocket.send_json({"type": "error", "message": "Invalid sampling steps value"})
                return
                
            # Update the client-specific setting
            old_steps = client_settings["sampling_steps"]
            client_settings["sampling_steps"] = steps
            
            print(f"[{time.perf_counter():.3f}] Updated sampling steps for client {client_id} from {old_steps} to {steps}")
            
            # Send confirmation to client
            await websocket.send_json({"type": "steps_updated", "steps": steps})
        
        # Add a function to update USE_RNN setting
        async def update_use_rnn(use_rnn):
            nonlocal client_settings
            
            # Update the client-specific setting
            old_setting = client_settings["use_rnn"]
            client_settings["use_rnn"] = use_rnn
            
            print(f"[{time.perf_counter():.3f}] Updated USE_RNN for client {client_id} from {old_setting} to {use_rnn}")
            
            # Send confirmation to client
            await websocket.send_json({"type": "rnn_updated", "use_rnn": use_rnn})
        
        # Add timeout checking function
        async def check_timeout():
            nonlocal timeout_warning_sent, timeout_task, connection_active, user_has_interacted
            
            while True:
                try:
                    # Check if WebSocket is still connected and connection is still active
                    if not connection_active or websocket.client_state.value >= 2:  # CLOSING or CLOSED
                        print(f"[{time.perf_counter():.3f}] Connection inactive or WebSocket closed, stopping timeout check for client {client_id}")
                        return
                    
                    # Don't start timeout tracking until user has actually interacted
                    if not user_has_interacted:
                        print(f"[{time.perf_counter():.3f}] User hasn't interacted yet, skipping timeout check for client {client_id}")
                        await asyncio.sleep(1)  # Check every second
                        continue
                    
                    current_time = time.perf_counter()
                    time_since_activity = current_time - last_user_activity_time
                    
                    print(f"[{current_time:.3f}] Timeout check - time_since_activity: {time_since_activity:.1f}s, WARNING_TIME: {WARNING_TIME}s, CONNECTION_TIMEOUT: {CONNECTION_TIMEOUT}s")
                    
                    # Send warning at 10 seconds
                    if time_since_activity >= WARNING_TIME and not timeout_warning_sent:
                        print(f"[{current_time:.3f}] Sending timeout warning to client {client_id}")
                        await websocket.send_json({
                            "type": "timeout_warning",
                            "timeout_in": CONNECTION_TIMEOUT - WARNING_TIME
                        })
                        timeout_warning_sent = True
                        print(f"[{current_time:.3f}] Timeout warning sent, timeout_warning_sent: {timeout_warning_sent}")
                    
                    # Close connection at 20 seconds
                    if time_since_activity >= CONNECTION_TIMEOUT:
                        print(f"[{current_time:.3f}] TIMEOUT REACHED! Closing connection {client_id} due to timeout")
                        print(f"[{current_time:.3f}] time_since_activity: {time_since_activity:.1f}s >= CONNECTION_TIMEOUT: {CONNECTION_TIMEOUT}s")
                        
                        # Clear the input queue before closing
                        queue_size_before = input_queue.qsize()
                        print(f"[{current_time:.3f}] Clearing input queue, size before: {queue_size_before}")
                        while not input_queue.empty():
                            try:
                                input_queue.get_nowait()
                                input_queue.task_done()
                            except asyncio.QueueEmpty:
                                break
                        print(f"[{current_time:.3f}] Input queue cleared, size after: {input_queue.qsize()}")
                        
                        print(f"[{current_time:.3f}] About to close WebSocket connection...")
                        await websocket.close(code=1000, reason="User inactivity timeout")
                        print(f"[{current_time:.3f}] WebSocket.close() called, returning from check_timeout")
                        return
                    
                    await asyncio.sleep(1)  # Check every second
                    
                except Exception as e:
                    print(f"[{time.perf_counter():.3f}] Error in timeout check for client {client_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
        
        # Function to update user activity
        def update_user_activity():
            nonlocal last_user_activity_time, timeout_warning_sent
            old_time = last_user_activity_time
            last_user_activity_time = time.perf_counter()
            print(f"[{time.perf_counter():.3f}] User activity detected for client {client_id}")
            print(f"[{time.perf_counter():.3f}] last_user_activity_time updated: {old_time:.3f} -> {last_user_activity_time:.3f}")
            
            if timeout_warning_sent:
                print(f"[{time.perf_counter():.3f}] User activity detected, resetting timeout warning for client {client_id}")
                timeout_warning_sent = False
                print(f"[{time.perf_counter():.3f}] timeout_warning_sent reset to: {timeout_warning_sent}")
                # Send activity reset notification to client
                asyncio.create_task(websocket.send_json({"type": "activity_reset"}))
                print(f"[{time.perf_counter():.3f}] Activity reset message sent to client")
        
        # Start timeout checking
        timeout_task = asyncio.create_task(check_timeout())
        print(f"[{time.perf_counter():.3f}] Timeout task started for client {client_id} (waiting for user interaction)")
        
        async def process_input(data):
            nonlocal previous_frame, hidden_states, keys_down, frame_num, frame_count, is_processing, user_has_interacted
            
            try:
                process_start_time = time.perf_counter()
                queue_size = input_queue.qsize()
                print(f"[{process_start_time:.3f}] Starting to process input. Queue size before: {queue_size}")
                frame_num += 1
                frame_count += 1  # Increment total frame counter
                
                # Calculate global FPS
                total_elapsed = process_start_time - connection_start_time
                global_fps = frame_count / total_elapsed if total_elapsed > 0 else 0
                
                # change x and y to be between 0 and width/height-1 in data
                data['x'] = max(0, min(data['x'], SCREEN_WIDTH - 1))
                data['y'] = max(0, min(data['y'], SCREEN_HEIGHT - 1))
                x = data.get("x")
                y = data.get("y")
                assert 0 <= x < SCREEN_WIDTH, f"x: {x} is out of range"
                assert 0 <= y < SCREEN_HEIGHT, f"y: {y} is out of range"
                is_left_click = data.get("is_left_click")
                is_right_click = data.get("is_right_click")
                keys_down_list = data.get("keys_down", [])  # Get as list
                keys_up_list = data.get("keys_up", [])
                is_auto_input = data.get("is_auto_input", False)
                if is_auto_input:
                    print (f'[{time.perf_counter():.3f}] Auto-input detected')
                else:
                    # Update user activity for non-auto inputs
                    update_user_activity()
                    # Mark that user has started interacting
                    if not user_has_interacted:
                        user_has_interacted = True
                        print(f"[{time.perf_counter():.3f}] User has started interacting with canvas for client {client_id}")
                print(f'[{time.perf_counter():.3f}] Processing: x: {x}, y: {y}, is_left_click: {is_left_click}, is_right_click: {is_right_click}, keys_down_list: {keys_down_list}, keys_up_list: {keys_up_list}, time_since_activity: {time.perf_counter() - last_user_activity_time:.3f}')
                
                # Update the set based on the received data
                for key in keys_down_list:
                    key = key.lower()
                    if key in KEYMAPPING:
                        key = KEYMAPPING[key]
                    keys_down.add(key)
                for key in keys_up_list:
                    key = key.lower()
                    if key in KEYMAPPING:
                        key = KEYMAPPING[key]
                    if key in keys_down:  # Check if key exists to avoid KeyError
                        keys_down.remove(key)
                if DEBUG_MODE:
                    print (f"DEBUG MODE, REMOVING HIDDEN STATES")
                    previous_frame = padding_image

                if DEBUG_MODE_2:
                    print (f'dsfdasdf frame_num: {frame_num}')
                    if frame_num > NUM_MAX_FRAMES-1:
                        print (f"DEBUG MODE_2, REMOVING HIDDEN STATES")
                        previous_frame = padding_image
                        frame_num = 0
                inputs = prepare_model_inputs(previous_frame, hidden_states, x, y, is_right_click, is_left_click, list(keys_down), stoi, itos, frame_num)
                
                # Use client-specific settings
                client_use_rnn = client_settings["use_rnn"]
                client_sampling_steps = client_settings["sampling_steps"]
                
                print(f"[{time.perf_counter():.3f}] Starting model inference with client settings - USE_RNN: {client_use_rnn}, SAMPLING_STEPS: {client_sampling_steps}...")
                
                # Pass client-specific settings to process_frame
                previous_frame, sample_img, hidden_states, timing_info = await process_frame(
                    model, 
                    inputs, 
                    use_rnn=client_use_rnn, 
                    num_sampling_steps=client_sampling_steps
                )
                
                print (f'Client {client_id} settings: USE_RNN: {client_use_rnn}, SAMPLING_STEPS: {client_sampling_steps}')

                
                timing_info['full_frame'] = time.perf_counter() - process_start_time
                
                print(f"[{time.perf_counter():.3f}] Model inference complete. Queue size now: {input_queue.qsize()}")
                # Use the provided function to print timing statistics
                print_timing_stats(timing_info, frame_num)
                
                # Print global FPS measurement
                print(f"  Global FPS: {global_fps:.2f} (total: {frame_count} frames in {total_elapsed:.2f}s)")
                
                
                img = Image.fromarray(sample_img)
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Send the generated frame back to the client
                print(f"[{time.perf_counter():.3f}] Sending image to client...")
                try:
                    await websocket.send_json({"image": img_str})
                    print(f"[{time.perf_counter():.3f}] Image sent. Queue size before next_input: {input_queue.qsize()}")
                except RuntimeError as e:
                    if "Cannot call 'send' once a close message has been sent" in str(e):
                        print(f"[{time.perf_counter():.3f}] WebSocket closed, skipping image send")
                    else:
                        raise e
                except Exception as e:
                    print(f"[{time.perf_counter():.3f}] Error sending image: {e}")

                # Log the input
                log_interaction(client_id, data, generated_frame=sample_img)
            finally:
                is_processing = False
                print(f"[{time.perf_counter():.3f}] Processing complete. Queue size before checking next input: {input_queue.qsize()}")
                # Check if we have more inputs to process after this one
                if not input_queue.empty():
                    print(f"[{time.perf_counter():.3f}] Queue not empty, processing next input")
                    asyncio.create_task(process_next_input())
        
        async def process_next_input():
            nonlocal is_processing
            
            current_time = time.perf_counter()
            if input_queue.empty():
                print(f"[{current_time:.3f}] No inputs to process. Queue is empty.")
                is_processing = False
                return
            
            # Check if WebSocket is still open by checking if it's in a closed state
            if websocket.client_state.value >= 2:  # CLOSING or CLOSED
                print(f"[{current_time:.3f}] WebSocket in closed state ({websocket.client_state.value}), stopping processing")
                is_processing = False
                return
            
            #if is_processing:
            #    print(f"[{current_time:.3f}] Already processing an input. Will check again later.")
            #    return
            
            # Set is_processing to True before proceeding
            is_processing = True
            
            queue_size = input_queue.qsize()
            print(f"[{current_time:.3f}] Processing next input. Queue size: {queue_size}")
            
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
                    is_interesting = (current_input.get("is_left_click") or 
                                      current_input.get("is_right_click") or 
                                      (current_input.get("keys_down") and len(current_input.get("keys_down")) > 0) or 
                                      (current_input.get("keys_up") and len(current_input.get("keys_up")) > 0))
                    
                    # Process immediately if interesting
                    if is_interesting:
                        print(f"[{current_time:.3f}] Found interesting input (skipped {skipped} events)")
                        await process_input(current_input)  # AWAIT here instead of creating a task
                        is_processing = False
                        return
                    
                    # Otherwise, continue to the next item
                    skipped += 1
                    
                    # If this is the last item and no interesting inputs were found
                    if input_queue.empty():
                        print(f"[{current_time:.3f}] No interesting inputs, processing latest movement (skipped {skipped-1} events)")
                        await process_input(latest_input)  # AWAIT here instead of creating a task
                        is_processing = False
                        return
            except Exception as e:
                print(f"[{current_time:.3f}] Error in process_next_input: {e}")
                import traceback
                traceback.print_exc()
                is_processing = False  # Make sure to reset on error
        
        while True:
            try:
                # Receive user input
                print(f"[{time.perf_counter():.3f}] Waiting for input... Queue size: {input_queue.qsize()}, is_processing: {is_processing}")
                data = await websocket.receive_json()
                receive_time = time.perf_counter()
                
                if data.get("type") == "heartbeat":
                    await websocket.send_json({"type": "heartbeat_response"})
                    continue
                
                # Handle reset command
                if data.get("type") == "reset":
                    print(f"[{receive_time:.3f}] Received reset command")
                    update_user_activity()  # Reset activity timer
                    await reset_simulation()
                    continue
                
                # Handle sampling steps update
                if data.get("type") == "update_sampling_steps":
                    print(f"[{receive_time:.3f}] Received request to update sampling steps")
                    update_user_activity()  # Reset activity timer
                    await update_sampling_steps(data.get("steps", 32))
                    continue
                
                # Handle USE_RNN update
                if data.get("type") == "update_use_rnn":
                    print(f"[{receive_time:.3f}] Received request to update USE_RNN")
                    update_user_activity()  # Reset activity timer
                    await update_use_rnn(data.get("use_rnn", False))
                    continue
                
                # Handle settings request
                if data.get("type") == "get_settings":
                    print(f"[{receive_time:.3f}] Received request for current settings")
                    update_user_activity()  # Reset activity timer
                    await websocket.send_json({
                        "type": "settings",
                        "sampling_steps": client_settings["sampling_steps"],
                        "use_rnn": client_settings["use_rnn"]
                    })
                    continue
                
                # Add the input to our queue
                await input_queue.put(data)
                print(f"[{receive_time:.3f}] Received input. Queue size now: {input_queue.qsize()}")
                
                # Check if WebSocket is still open before processing
                if websocket.client_state.value >= 2:  # CLOSING or CLOSED
                    print(f"[{receive_time:.3f}] WebSocket closed, skipping processing")
                    continue
                
                # If we're not currently processing, start processing this input
                if not is_processing:
                    print(f"[{receive_time:.3f}] Not currently processing, will call process_next_input()")
                    is_processing = True
                    asyncio.create_task(process_next_input())  # Create task but don't await it
                else:
                    print(f"[{receive_time:.3f}] Currently processing, new input queued for later")
            
            except asyncio.TimeoutError:
                print("WebSocket connection timed out")
            
            except WebSocketDisconnect:
                # Log final EOS entry
                log_interaction(client_id, {}, is_end_of_session=True)
                print(f"[{time.perf_counter():.3f}] WebSocket disconnected: {client_id}")
                print(f"[{time.perf_counter():.3f}] WebSocketDisconnect exception caught")
                break

    except Exception as e:
        print(f"Error in WebSocket connection {client_id}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up timeout task
        print(f"[{time.perf_counter():.3f}] Cleaning up connection {client_id}")
        connection_active = False  # Signal that connection is being cleaned up
        if timeout_task and not timeout_task.done():
            print(f"[{time.perf_counter():.3f}] Cancelling timeout task for client {client_id}")
            timeout_task.cancel()
            try:
                await timeout_task
                print(f"[{time.perf_counter():.3f}] Timeout task cancelled successfully for client {client_id}")
            except asyncio.CancelledError:
                print(f"[{time.perf_counter():.3f}] Timeout task cancelled with CancelledError for client {client_id}")
                pass
        else:
            print(f"[{time.perf_counter():.3f}] Timeout task already done or doesn't exist for client {client_id}")
        
        # Print final FPS statistics when connection ends
        if frame_num >= 0:  # Only if we processed at least one frame
            total_time = time.perf_counter() - connection_start_time
            print(f"\nConnection {client_id} summary:")
            print(f"  Total frames processed: {frame_count}")
            print(f"  Total elapsed time: {total_time:.2f} seconds")
            print(f"  Average FPS: {frame_count/total_time:.2f}")
        
        print(f"WebSocket connection closed: {client_id}")

def log_interaction(client_id, data, generated_frame=None, is_end_of_session=False, is_reset=False):
    """Log user interaction and optionally the generated frame."""
    timestamp = time.time()
    
    # Create directory structure if it doesn't exist
    os.makedirs("interaction_logs", exist_ok=True)
    
    # Structure the log entry
    log_entry = {
        "timestamp": timestamp,
        "client_id": client_id,
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
            "is_auto_input": data.get("is_auto_input", False)
        }
    else:
        # For EOS/reset records, just include minimal info
        log_entry["inputs"] = None
    
    # Save to a file (one file per session)
    session_file = f"interaction_logs/session_{client_id}.jsonl"
    with open(session_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    # Optionally save the frame if provided
    if generated_frame is not None and not is_end_of_session and not is_reset:
        frame_dir = f"interaction_logs/frames_{client_id}"
        os.makedirs(frame_dir, exist_ok=True)
        frame_file = f"{frame_dir}/{timestamp:.6f}.png"
        # Save the frame as PNG
        Image.fromarray(generated_frame).save(frame_file)
