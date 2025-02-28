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


SCREEN_WIDTH = 512
SCREEN_HEIGHT = 384
NUM_SAMPLING_STEPS = 8
DATA_NORMALIZATION = {
    'mean': -0.54,
    'std': 6.78,
}
LATENT_DIMS = (1, SCREEN_HEIGHT // 8, SCREEN_WIDTH // 8, 4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize the model at the start of your application
#model = initialize_model("config_csllm.yaml", "yuntian-deng/computer-model")
model = initialize_model("standard_challenging_context32_nocond_all.yaml", "yuntian-deng/computer-model")

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
    inputs = {
        'image_features': previous_frame.to(device),
        'is_padding': torch.BoolTensor([time_step == 0]).to(device),
        'x': torch.LongTensor([x if x is not None else 0]).unsqueeze(0).to(device),
        'y': torch.LongTensor([y if y is not None else 0]).unsqueeze(0).to(device),
        'is_leftclick': torch.BoolTensor([left_click]).unsqueeze(0).to(device),
        'is_rightclick': torch.BoolTensor([right_click]).unsqueeze(0).to(device),
        'key_events': torch.zeros(len(itos), dtype=torch.long).to(device)
    }
    for key in keys_down:
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
        while True:
            try:
                # Receive user input with a timeout
                #data = await asyncio.wait_for(websocket.receive_json(), timeout=90000.0)
                data = await websocket.receive_json()
                if data.get("type") == "heartbeat":
                    await websocket.send_json({"type": "heartbeat_response"})
                    continue
                frame_num += 1
                start_frame = time.perf_counter()
                x = data.get("x")
                y = data.get("y")
                is_left_click = data.get("is_left_click")
                is_right_click = data.get("is_right_click")
                keys_down_list = data.get("keys_down", [])  # Get as list
                keys_up_list = data.get("keys_up", [])
                
                # Update the set based on the received data
                for key in keys_down_list:
                    keys_down.add(key)
                for key in keys_up_list:
                    if key in keys_down:  # Check if key exists to avoid KeyError
                        keys_down.remove(key)

                inputs = prepare_model_inputs(previous_frame, hidden_states, x, y, is_right_click, is_left_click, list(keys_down), stoi, itos, frame_num)

                previous_frame, sample_img, hidden_states, timing_info = process_frame(model, inputs)
                timing_info['full_frame'] = time.perf_counter() - start_frame
    
                img = Image.fromarray(sample_img)
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Send the generated frame back to the client
                await websocket.send_json({"image": img_str})
            
            except asyncio.TimeoutError:
                print("WebSocket connection timed out")
                #break  # Exit the loop on timeout
            
            except WebSocketDisconnect:
                print("WebSocket disconnected")
                #break  # Exit the loop on disconnect

    except Exception as e:
        print(f"Error in WebSocket connection {client_id}: {e}")
    
    finally:
        print(f"WebSocket connection closed: {client_id}")
        #await websocket.close()  # Ensure the WebSocket is closed
