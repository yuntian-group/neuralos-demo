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

DEBUG = True
app = FastAPI()

# Mount the static directory to serve HTML, JavaScript, and CSS files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the index.html file at the root URL
@app.get("/")
async def get():
    return HTMLResponse(open("static/index.html").read())

def generate_random_image(width: int, height: int) -> np.ndarray:
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

def draw_trace(image: np.ndarray, previous_actions: List[Tuple[str, List[int]]]) -> np.ndarray:
    pil_image = Image.fromarray(image)
    #pil_image = Image.open('image_3.png')    
    draw = ImageDraw.Draw(pil_image)
    flag = True
    prev_x, prev_y = None, None
    for i, (action_type, position) in enumerate(previous_actions):
        color = (255, 0, 0) if action_type == "move" else (0, 255, 0)
        x, y = position
        if x == 0 and y == 0 and flag:
            continue
        else:
            flag = False
        if DEBUG:
            x = x * 256 / 1024
            y = y * 256 / 640
        draw.ellipse([x-2, y-2, x+2, y+2], fill=color)
        
        if prev_x is not None:
            #prev_x, prev_y = previous_actions[i-1][1]
            draw.line([prev_x, prev_y, x, y], fill=color, width=1)
        prev_x, prev_y = x, y
    
    return np.array(pil_image)

# Initialize the model at the start of your application
model = initialize_model("config_csllm.yaml", "yuntian-deng/computer-model")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def load_initial_images(width, height):
    initial_images = []
    for i in range(7):
        initial_images.append(np.zeros((height, width, 3), dtype=np.uint8))
        #image_path = f"image_{i}.png"
        #if os.path.exists(image_path):
        #    img = Image.open(image_path).resize((width, height))
        #    initial_images.append(np.array(img))
        #else:
        #    print(f"Warning: {image_path} not found. Using blank image instead.")
        #    initial_images.append(np.zeros((height, width, 3), dtype=np.uint8))
    return initial_images

def normalize_images(images, target_range=(-1, 1)):
    images = np.stack(images).astype(np.float32)
    if target_range == (-1, 1):
        return images / 127.5 - 1
    elif target_range == (0, 1):
        return images / 255.0
    else:
        raise ValueError(f"Unsupported target range: {target_range}")

def denormalize_image(image, source_range=(-1, 1)):
    if source_range == (-1, 1):
        return ((image + 1) * 127.5).clip(0, 255).astype(np.uint8)
    elif source_range == (0, 1):
        return (image * 255).clip(0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported source range: {source_range}")
        
def format_action(action_str, is_padding=False):
    if is_padding:
        return "N N N N N : N N N N N"
    
    # Split the x~y coordinates
    x, y = map(int, action_str.split('~'))
    
    # Convert numbers to padded strings and add spaces between digits
    x_str = f"{abs(x):04d}"
    y_str = f"{abs(y):04d}"
    x_spaced = ' '.join(x_str)
    y_spaced = ' '.join(y_str)
    
    # Format with sign and proper spacing
    return f"{'+ ' if x >= 0 else '- '}{x_spaced} : {'+ ' if y >= 0 else '- '}{y_spaced}"
    
def predict_next_frame(previous_frames: List[np.ndarray], previous_actions: List[Tuple[str, List[int]]]) -> np.ndarray:
    width, height = 256, 256
    initial_images = load_initial_images(width, height)

    # Prepare the image sequence for the model
    image_sequence = previous_frames[-7:]  # Take the last 7 frames
    while len(image_sequence) < 7:
        image_sequence.insert(0, initial_images[len(image_sequence)])

    # Convert the image sequence to a tensor and concatenate in the channel dimension
    image_sequence_tensor = torch.from_numpy(normalize_images(image_sequence, target_range=(-1, 1)))
    image_sequence_tensor = image_sequence_tensor.to(device)
    
    # Prepare the prompt based on the previous actions
    action_descriptions = []
    initial_actions = ['901:604', '901:604', '901:604', '901:604', '901:604', '901:604', '901:604', '921:604']
    initial_actions = ['0:0'] * 7
    #initial_actions = ['N N N N N : N N N N N'] * 7
    def unnorm_coords(x, y):
        return int(x), int(y) #int(x - (1920 - 256) / 2), int(y - (1080 - 256) / 2)
    
    # Process initial actions if there are not enough previous actions
    while len(previous_actions) < 8:
        x, y = map(int, initial_actions.pop(0).split(':'))
        previous_actions.insert(0, ("move", unnorm_coords(x, y)))
    prev_x = 0
    prev_y = 0
    for action_type, pos in previous_actions: #[-8:]:
        if action_type == "move":
            x, y = pos
            norm_x = int(round(x / 256 * 1024)) #x + (1920 - 256) / 2
            norm_y = int(round(y / 256 * 640)) #y + (1080 - 256) / 2
            if DEBUG:
                norm_x = x
                norm_y = y
            #action_descriptions.append(f"{(norm_x-prev_x):.0f}~{(norm_y-prev_y):.0f}")
            action_descriptions.append(format_action(f'{norm_x-prev_x:.0f}~{norm_y-prev_y:.0f}'), pos=='0~0')
            prev_x = norm_x
            prev_y = norm_y
        elif action_type == "left_click":
            action_descriptions.append("left_click")
        elif action_type == "right_click":
            action_descriptions.append("right_click")
    
    prompt = " ".join(action_descriptions[-8:])
    #prompt = ''
    #prompt = "1~1 0~0 0~0 0~0 0~0 0~0 0~0 0~0"
    print(prompt)
    
    # Generate the next frame
    new_frame = sample_frame(model, prompt, image_sequence_tensor)
    
    # Convert the generated frame to the correct format
    new_frame = new_frame.transpose(1, 2, 0)
    print (new_frame.max(), new_frame.min())
    new_frame_denormalized = denormalize_image(new_frame, source_range=(-1, 1))
    
    # Draw the trace of previous actions
    new_frame_with_trace = draw_trace(new_frame_denormalized, previous_actions)
    
    return new_frame_with_trace, new_frame_denormalized

# WebSocket endpoint for continuous user interaction
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = id(websocket)  # Use a unique identifier for each connection
    print(f"New WebSocket connection: {client_id}")
    await websocket.accept()
    previous_frames = []
    previous_actions = []
    positions = ['815~335', '787~342', '787~342', '749~345', '703~346', '703~346', '654~347', '654~347', '604~349', '555~353', '555~353', '509~357', '509~357', '468~362', '431~368', '431~368']
    #positions = ['815~335', '787~342', '749~345', '703~346', '703~346', '654~347', '654~347', '604~349', '555~353', '555~353', '509~357', '509~357', '468~362', '431~368', '431~368']

#positions = positions[:4]
    try:
        while True:
            try:
                # Receive user input with a timeout
                #data = await asyncio.wait_for(websocket.receive_json(), timeout=90000.0)
                data = await websocket.receive_json()

                
                if data.get("type") == "heartbeat":
                    await websocket.send_json({"type": "heartbeat_response"})
                    continue
                
                action_type = data.get("action_type")
                mouse_position = data.get("mouse_position")
                
                # Store the actions
                if DEBUG:
                    position = positions[0]
                    #positions = positions[1:]
                    mouse_position = position.split('~')
                    mouse_position = [int(item) for item in mouse_position]
                    #mouse_position = '+ 0 8 1 5 : + 0 3 3 5'
                    
                #previous_actions.append((action_type, mouse_position))
                previous_actions = [(action_type, mouse_position))]
                
                # Log the start time
                start_time = time.time()
                
                # Predict the next frame based on the previous frames and actions
                next_frame, next_frame_append = predict_next_frame(previous_frames, previous_actions)
                # Load and append the corresponding ground truth image instead of model output
                img = Image.open(f"image_{len(previous_frames)%7}.png")
                #previous_frames.append(np.array(img))
                
                # Convert the numpy array to a base64 encoded image
                img = Image.fromarray(next_frame)
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Log the processing time
                processing_time = time.time() - start_time
                print(f"Frame processing time: {processing_time:.2f} seconds")
                
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
