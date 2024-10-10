from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw
import base64
import io

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

def predict_next_frame(previous_frames: List[np.ndarray], previous_actions: List[Tuple[str, List[int]]]) -> np.ndarray:
    width, height = 800, 600
    
    if not previous_frames or previous_actions[-1][0] == "move":
        # Generate a new random image when there's no previous frame or the mouse moves
        new_frame = generate_random_image(width, height)
    else:
        # Use the last frame if it exists and the action is not a mouse move
        new_frame = previous_frames[-1].copy()
    
    # Draw the trace of previous actions
    new_frame_with_trace = draw_trace(new_frame, previous_actions)
    
    return new_frame_with_trace

# WebSocket endpoint for continuous user interaction
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    previous_frames = []
    previous_actions = []
    
    try:
        while True:
            # Receive user input (mouse movement, click, etc.)
            data = await websocket.receive_json()
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

    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()
