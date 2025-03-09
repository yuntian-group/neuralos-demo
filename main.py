from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw
import base64
import io
import asyncio
from utils import initialize_model, sample_frame
import torch
import os
import time
from typing import Any, Dict
from ldm.models.diffusion.ddpm import LatentDiffusion, DDIMSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
SCREEN_WIDTH = 512
SCREEN_HEIGHT = 384
NUM_SAMPLING_STEPS = 8
DATA_NORMALIZATION = {
    'mean': -0.54,
    'std': 6.78,
}
LATENT_DIMS = (4, SCREEN_HEIGHT // 8, SCREEN_WIDTH // 8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize the model at the start of your application
#model = initialize_model("config_csllm.yaml", "yuntian-deng/computer-model")
model = initialize_model("config_rnn.yaml", "yuntian-deng/computer-model")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
#model = torch.compile(model)

padding_image = torch.zeros(1, SCREEN_HEIGHT // 8, SCREEN_WIDTH // 8, 4)
padding_image = (padding_image - DATA_NORMALIZATION['mean']) / DATA_NORMALIZATION['std']
padding_image = padding_image.to(device)

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
INVALID_KEYS = ['f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20',
                'f21', 'f22', 'f23', 'f24', 'select', 'separator', 'execute']
VALID_KEYS = [key for key in KEYS if key not in INVALID_KEYS]
itos = VALID_KEYS
stoi = {key: i for i, key in enumerate(itos)}

app = FastAPI()

# Mount the static directory to serve HTML, JavaScript, and CSS files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add this at the top with other global variables

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
        inputs['key_events'][stoi[key]] = 1
    
    if hidden_states is not None:
        inputs['hidden_states'] = hidden_states
    
    return inputs

@torch.no_grad()
def process_frame(
    model: LatentDiffusion,
    inputs: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, np.ndarray, Any, Dict[str, float]]:
    """Process a single frame through the model."""
    timing = {}
    # Temporal encoding
    start = time.perf_counter()
    output_from_rnn, hidden_states = model.temporal_encoder.forward_step(inputs)
    timing['temporal_encoder'] = time.perf_counter() - start
    
    # UNet sampling
    start = time.perf_counter()
    sampler = DDIMSampler(model)
    sample_latent, _ = sampler.sample(
        S=NUM_SAMPLING_STEPS,
        conditioning={'c_concat': output_from_rnn},
        batch_size=1,
        shape=LATENT_DIMS,
        verbose=False
    )
    timing['unet'] = time.perf_counter() - start
    
    # Decoding
    start = time.perf_counter()
    sample = sample_latent * DATA_NORMALIZATION['std'] + DATA_NORMALIZATION['mean']
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
    client_id = id(websocket)  # Use a unique identifier for each connection
    print(f"New WebSocket connection: {client_id}")
    await websocket.accept()

    try:
        previous_frame = padding_image
        hidden_states = None
        keys_down = set()  # Initialize as an empty set
        frame_num = -1
        
        # Start timing for global FPS calculation
        connection_start_time = time.perf_counter()
        frame_count = 0
        
        # Input queue management - use asyncio.Queue instead of a list
        input_queue = asyncio.Queue()
        is_processing = False
        
        async def process_input(data):
            nonlocal previous_frame, hidden_states, keys_down, frame_num, frame_count, is_processing
            
            try:
                process_start_time = time.perf_counter()
                queue_size = input_queue.qsize()
                print(f"[{process_start_time:.3f}] Starting to process input. Queue size before: {queue_size}")
                frame_num += 1
                frame_count += 1  # Increment total frame counter
                
                # Calculate global FPS
                total_elapsed = process_start_time - connection_start_time
                global_fps = frame_count / total_elapsed if total_elapsed > 0 else 0
                
                x = data.get("x")
                y = data.get("y")
                is_left_click = data.get("is_left_click")
                is_right_click = data.get("is_right_click")
                keys_down_list = data.get("keys_down", [])  # Get as list
                keys_up_list = data.get("keys_up", [])
                print(f'[{time.perf_counter():.3f}] Processing: x: {x}, y: {y}, is_left_click: {is_left_click}, is_right_click: {is_right_click}, keys_down_list: {keys_down_list}, keys_up_list: {keys_up_list}')
                
                # Update the set based on the received data
                for key in keys_down_list:
                    keys_down.add(key)
                for key in keys_up_list:
                    if key in keys_down:  # Check if key exists to avoid KeyError
                        keys_down.remove(key)
                
                inputs = prepare_model_inputs(previous_frame, hidden_states, x, y, is_right_click, is_left_click, list(keys_down), stoi, itos, frame_num)
                print(f"[{time.perf_counter():.3f}] Starting model inference...")
                previous_frame, sample_img, hidden_states, timing_info = process_frame(model, inputs)
                timing_info['full_frame'] = time.perf_counter() - process_start_time
                
                print(f"[{time.perf_counter():.3f}] Model inference complete. Queue size now: {input_queue.qsize()}")
                # Use the provided function to print timing statistics
                print_timing_stats(timing_info, frame_num)
                
                # Print global FPS measurement
                print(f"  Global FPS: {global_fps:.2f} (total: {frame_count} frames in {total_elapsed:.2f}s)")
                time.sleep(1)
                
                img = Image.fromarray(sample_img)
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Send the generated frame back to the client
                print(f"[{time.perf_counter():.3f}] Sending image to client...")
                await websocket.send_json({"image": img_str})
                print(f"[{time.perf_counter():.3f}] Image sent. Queue size before next_input: {input_queue.qsize()}")
            finally:
                is_processing = False
                print(f"[{time.perf_counter():.3f}] Processing complete. Queue size before checking next input: {input_queue.qsize()}")
                # Check if we have more inputs to process after this one TODO
                #asyncio.create_task(process_next_input())
        
        async def process_next_input():
            nonlocal is_processing
            
            current_time = time.perf_counter()
            if input_queue.empty():
                print(f"[{current_time:.3f}] No inputs to process. Queue is empty.")
                is_processing = False
                return
            
            if is_processing:
                print(f"[{current_time:.3f}] Already processing an input. Will check again later.")
                return
            
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
                
                # Add the input to our queue
                await input_queue.put(data)
                print(f"[{receive_time:.3f}] Received input. Queue size now: {input_queue.qsize()}")
                
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
                print("WebSocket disconnected")
                break

    except Exception as e:
        print(f"Error in WebSocket connection {client_id}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Print final FPS statistics when connection ends
        if frame_num >= 0:  # Only if we processed at least one frame
            total_time = time.perf_counter() - connection_start_time
            print(f"\nConnection {client_id} summary:")
            print(f"  Total frames processed: {frame_count}")
            print(f"  Total elapsed time: {total_time:.2f} seconds")
            print(f"  Average FPS: {frame_count/total_time:.2f}")
        
        print(f"WebSocket connection closed: {client_id}")
