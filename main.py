from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import numpy as np

app = FastAPI()

# Mount the static directory to serve HTML, JavaScript, and CSS files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the index.html file at the root URL
@app.get("/")
async def get():
    return HTMLResponse(open("static/index.html").read())

# Simulate your diffusion model (placeholder)
def predict_next_frame(previous_frames: List[np.ndarray], previous_actions: List[str]) -> np.ndarray:
    return np.zeros((800, 600, 3), dtype=np.uint8)

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
            
            # Send the generated frame back to the client (encoded as base64 or similar)
            await websocket.send_text("Next frame generated")  # Replace with real image sending logic

    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()
