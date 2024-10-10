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
    draw = ImageDraw.Draw(pil_image)
    
    for i, (action_type, position) in enumerate(previous_actions):
        color = (255, 0, 0) if action_type == "move" else (0, 255, 0)
        x, y = position
        draw.ellipse([x-2, y-2, x+2, y+2], fill=color)
        
        if i > 0:
            prev_x, prev_y = previous_actions[i-1][1]
            draw.line([prev_x, prev_y, x, y], fill=color, width=1)
    
    return np.array(pil_image)

# Initialize the model at the start of your application
model = initialize_model("config_csllm.yaml", "yuntian-deng/computer-model")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
def predict_next_frame(previous_frames: List[np.ndarray], previous_actions: List[Tuple[str, List[int]]]) -> np.ndarray:
    width, height = 256, 256
    
    # Prepare the image sequence for the model
    image_sequence = previous_frames[-7:]  # Take the last 7 frames
    while len(image_sequence) < 7:
        image_sequence.insert(0, np.zeros((height, width, 3), dtype=np.uint8))
    
    # Convert the image sequence to a tensor and concatenate in the channel dimension
    image_sequence_tensor = torch.from_numpy(np.stack(image_sequence)).float() / 127.5 - 1
    image_sequence_tensor = image_sequence_tensor.to(device)
    
    
    # Prepare the prompt based on the previous actions
    #action_descriptions = [f"{pos[0]}:{pos[1]}" for _, pos in previous_actions[-7:]]
    #prompt = " ".join(action_descriptions)
    action_descriptions = []
    for action_type, pos in previous_actions[-7:]:
        if action_type == "move":
            action_descriptions.append(f"{pos[0]}:{pos[1]}")
        elif action_type == "left_click":
            action_descriptions.append("left_click")
        elif action_type == "right_click":
            action_descriptions.append("right_click")
    
    prompt = " ".join(action_descriptions)
    
    # Generate the next frame
    new_frame = sample_frame(model, prompt, image_sequence_tensor)
    
    # Convert the generated frame to the correct format
    new_frame = (new_frame * 255).astype(np.uint8).transpose(1, 2, 0)
    
    # Resize the frame to 256x256 if necessary
    if new_frame.shape[:2] != (height, width):
        new_frame = np.array(Image.fromarray(new_frame).resize((width, height)))
    
    # Draw the trace of previous actions
    new_frame_with_trace = draw_trace(new_frame, previous_actions)
    
    return new_frame_with_trace

# WebSocket endpoint for continuous user interaction
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = id(websocket)  # Use a unique identifier for each connection
    print(f"New WebSocket connection: {client_id}")
    await websocket.accept()
    previous_frames = []
    previous_actions = []
    
    try:
        while True:
            try:
                # Receive user input with a timeout
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                
                if data.get("type") == "heartbeat":
                    await websocket.send_json({"type": "heartbeat_response"})
                    continue
                
                action_type = data.get("action_type")
                mouse_position = data.get("mouse_position")
                
                # Store the actions
                previous_actions.append((action_type, mouse_position))
                
                # Predict the next frame based on the previous frames and actions
                next_frame = predict_next_frame(previous_frames, previous_actions)
                previous_frames.append(next_frame)
                
                # Convert the numpy array to a base64 encoded image
                img = Image.fromarray(next_frame)
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Send the generated frame back to the client
                await websocket.send_json({"image": img_str})
            
            except asyncio.TimeoutError:
                print("WebSocket connection timed out")
                break
            
            except WebSocketDisconnect:
                print("WebSocket disconnected")
                break

    except Exception as e:
        print(f"Error in WebSocket connection {client_id}: {e}")
    
    finally:
        print(f"WebSocket connection closed: {client_id}")
        # Remove the explicit websocket.close() call here
