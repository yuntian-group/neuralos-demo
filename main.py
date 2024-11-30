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

DEBUG = False
DEBUG_TEACHER_FORCING = True
app = FastAPI()

# Mount the static directory to serve HTML, JavaScript, and CSS files
app.mount("/static", StaticFiles(directory="static"), name="static")


def parse_action_string(action_str):
    """Convert formatted action string to x, y coordinates
    Args:
        action_str: String like 'N N N N N : N N N N N' or '+ 0 2 1 3 : + 0 3 8 3'
    Returns:
        tuple: (x, y) coordinates or None if action is padding
    """
    action_type = action_str[0]
    action_str = action_str[1:].strip()
    if 'N' in action_str:
        return (None, None, None)
        
    # Split into x and y parts
    action_str = action_str.replace(' ', '')
    x_part, y_part = action_str.split(':')
    
    # Parse x: remove sign, join digits, convert to int, apply sign
    
    x = int(x_part)
    
    # Parse y: remove sign, join digits, convert to int, apply sign
    y = int(y_part)
    
    return x, y, action_type

def create_position_and_click_map(pos,action_type, image_height=48, image_width=64, original_width=512, original_height=384):
    """Convert cursor position to a binary position map
    Args:
        x, y: Original cursor positions
        image_size: Size of the output position map (square)
        original_width: Original screen width (1024)
        original_height: Original screen height (640)
    Returns:
        torch.Tensor: Binary position map of shape (1, image_size, image_size)
    """
    x, y = pos
    if x is None:
        return torch.zeros((1, image_height, image_width)), torch.zeros((1, image_height, image_width)), None, None
    # Scale the positions to new size
    #x_scaled = int((x / original_width) * image_size)
    #y_scaled = int((y / original_height) * image_size)
    #screen_width, screen_height = 512, 384
    #video_width, video_height = 512, 384
        
    #x_scaled = x - (screen_width / 2 - video_width / 2)
    #y_scaled = y - (screen_height / 2 - video_height / 2)
    x_scaled = int(x / original_width * image_width)
    y_scaled = int(y / original_height * image_height)
    
    # Clamp values to ensure they're within bounds
    x_scaled = max(0, min(x_scaled, image_width - 1))
    y_scaled = max(0, min(y_scaled, image_height - 1))
    
    # Create binary position map
    pos_map = torch.zeros((1, image_height, image_width))
    pos_map[0, y_scaled, x_scaled] = 1.0

    leftclick_map = torch.zeros((1, image_height, image_width))
    if action_type == 'L':
        leftclick_map[0, y_scaled, x_scaled] = 1.0
    
    
    return pos_map, leftclick_map, x_scaled, y_scaled
    
# Serve the index.html file at the root URL
@app.get("/")
async def get():
    return HTMLResponse(open("static/index.html").read())

def generate_random_image(width: int, height: int) -> np.ndarray:
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

def draw_trace(image: np.ndarray, previous_actions: List[Tuple[str, List[int]]], x_scaled=-1, y_scaled=-1) -> np.ndarray:
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
        #if DEBUG:
        #    x = x * 256 / 1024
        #    y = y * 256 / 640
        #draw.ellipse([x-2, y-2, x+2, y+2], fill=color)
        
        
        #if prev_x is not None:
        #    #prev_x, prev_y = previous_actions[i-1][1]
        #    draw.line([prev_x, prev_y, x, y], fill=color, width=1)
        prev_x, prev_y = x, y
    draw.ellipse([x_scaled*8-2, y_scaled*8-2, x_scaled*8+2, y_scaled*8+2], fill=(0, 255, 0))
    #pil_image = pil_image.convert("RGB")
    
    return np.array(pil_image)

# Initialize the model at the start of your application
#model = initialize_model("config_csllm.yaml", "yuntian-deng/computer-model")
model = initialize_model("pssearch_bsz64_acc1_lr8e5_512_leftclick_histpos_512_384.yaml", "yuntian-deng/computer-model")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def load_initial_images(width, height):
    initial_images = []
    if DEBUG_TEACHER_FORCING:
        # Load the previous 7 frames for image_81
        for i in range(209-7, 209):  # Load images 74-80
            img = Image.open(f"record_100/image_{i}.png").resize((width, height))
            initial_images.append(np.array(img))
    else:
        #assert False
        for i in range(7):
            initial_images.append(np.zeros((height, width, 3), dtype=np.uint8))
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
        
def format_action(action_str, is_padding=False, is_leftclick=False):
    if is_padding:
        return "N N N N N N : N N N N N"
    
    # Split the x~y coordinates
    x, y = map(int, action_str.split('~'))
    prefix = 'N'
    if is_leftclick:
        prefix = 'L'
    # Convert numbers to padded strings and add spaces between digits
    x_str = f"{abs(x):04d}"
    y_str = f"{abs(y):04d}"
    x_spaced = ' '.join(x_str)
    y_spaced = ' '.join(y_str)
    
    # Format with sign and proper spacing
    return prefix + " " + f"{'+ ' if x >= 0 else '- '}{x_spaced} : {'+ ' if y >= 0 else '- '}{y_spaced}"
    
def predict_next_frame(previous_frames: List[np.ndarray], previous_actions: List[Tuple[str, List[int]]]) -> np.ndarray:
    width, height = 512, 384
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
    #initial_actions = ['901:604', '901:604', '901:604', '901:604', '901:604', '901:604', '901:604', '921:604']
    initial_actions = ['0:0'] * 7
    #initial_actions = ['N N N N N : N N N N N'] * 7
    def unnorm_coords(x, y):
        return int(x), int(y) #int(x - (1920 - 256) / 2), int(y - (1080 - 256) / 2)
    
    # Process initial actions if there are not enough previous actions
    while len(previous_actions) < 8:
        x, y = map(int, initial_actions.pop(0).split(':'))
        previous_actions.insert(0, ("N", unnorm_coords(x, y)))
    prev_x = 0
    prev_y = 0
    #print ('here')

    
    
    for action_type, pos in previous_actions: #[-8:]:
        print ('here3', action_type, pos)
        if action_type == 'move':
            action_type = 'N'
        if action_type == 'left_click':
            action_type = 'L'
        if action_type == "N":
            x, y = pos
            #norm_x = int(round(x / 256 * 1024)) #x + (1920 - 256) / 2
            #norm_y = int(round(y / 256 * 640)) #y + (1080 - 256) / 2
            #norm_x = x + (1920 - 512) / 2
            #norm_y = y + (1080 - 512) / 2
            norm_x = x
            norm_y = y
            if False and DEBUG_TEACHER_FORCING:
                norm_x = x
                norm_y = y
            #action_descriptions.append(f"{(norm_x-prev_x):.0f}~{(norm_y-prev_y):.0f}")
            #action_descriptions.append(format_action(f'{norm_x-prev_x:.0f}~{norm_y-prev_y:.0f}', x==0 and y==0))
            action_descriptions.append(format_action(f'{norm_x:.0f}~{norm_y:.0f}', x==0 and y==0))
            prev_x = norm_x
            prev_y = norm_y
        elif action_type == "L":
            x, y = pos
            #norm_x = int(round(x / 256 * 1024)) #x + (1920 - 256) / 2
            #norm_y = int(round(y / 256 * 640)) #y + (1080 - 256) / 2
            #norm_x = x + (1920 - 512) / 2
            #norm_y = y + (1080 - 512) / 2
            norm_x = x
            norm_y = y
            if False and DEBUG_TEACHER_FORCING:
                norm_x = x #+ (1920 - 512) / 2
                norm_y = y #+ (1080 - 512) / 2
            #if DEBUG:
            #    norm_x = x
            #    norm_y = y
            #action_descriptions.append(f"{(norm_x-prev_x):.0f}~{(norm_y-prev_y):.0f}")
            #action_descriptions.append(format_action(f'{norm_x-prev_x:.0f}~{norm_y-prev_y:.0f}', x==0 and y==0))
            action_descriptions.append(format_action(f'{norm_x:.0f}~{norm_y:.0f}', x==0 and y==0, True))
        elif action_type == "right_click":
            assert False
            action_descriptions.append("right_click")
        else:
            assert False
    
    prompt = " ".join(action_descriptions[-8:])
    print(prompt)
    #prompt = "N N N N N : N N N N N N N N N N : N N N N N N N N N N : N N N N N N N N N N : N N N N N N N N N N : N N N N N N N N N N : N N N N N N N N N N : N N N N N + 0 3 0 7 : + 0 3 7 5"
    #x, y, action_type = parse_action_string(action_descriptions[-1])
    #pos_map, leftclick_map, x_scaled, y_scaled = create_position_and_click_map((x, y), action_type)
    leftclick_maps = []
    pos_maps = []
    for j in range(1, 9):
        print ('fsfs', action_descriptions[-j])
        x, y, action_type = parse_action_string(action_descriptions[-j])
        pos_map_j, leftclick_map_j, x_scaled_j, y_scaled_j = create_position_and_click_map((x, y), action_type)
        leftclick_maps.append(leftclick_map_j)
        pos_maps.append(pos_map_j)
        if j == 1:
            x_scaled = x_scaled_j
            y_scaled = y_scaled_j
    
    #prompt = ''
    #prompt = "1~1 0~0 0~0 0~0 0~0 0~0 0~0 0~0"
    print(prompt)
    
    # Generate the next frame
    new_frame = sample_frame(model, prompt, image_sequence_tensor, pos_maps=pos_maps, leftclick_maps=leftclick_maps)
    
    # Convert the generated frame to the correct format
    new_frame = new_frame.transpose(1, 2, 0)
    print (new_frame.max(), new_frame.min())
    new_frame_denormalized = denormalize_image(new_frame, source_range=(-1, 1))
    
    # Draw the trace of previous actions
    new_frame_with_trace = draw_trace(new_frame_denormalized, previous_actions, x_scaled, y_scaled)
    
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
    positions = ['307~375']
    positions = ['815~335']
    #positions = ['787~342']
    positions = ['300~800']

    if DEBUG_TEACHER_FORCING:
        #print ('here2')
        # Use the predefined actions for image_81
        debug_actions = [
            'N + 0 8 5 3 : + 0 4 5 0', 'N + 0 8 7 1 : + 0 4 6 3',
            'N + 0 8 9 0 : + 0 4 7 5', 'N + 0 9 0 8 : + 0 4 8 8',
            'N + 0 9 2 7 : + 0 5 0 1', 'N + 0 9 2 7 : + 0 5 0 1',
            'N + 0 9 2 7 : + 0 5 0 1', 'N + 0 9 2 7 : + 0 5 0 1',
            'N + 0 9 2 7 : + 0 5 0 1', 'N + 0 9 2 7 : + 0 5 0 1',
            'L + 0 9 2 7 : + 0 5 0 1', 'N + 0 9 2 7 : + 0 5 0 1',
            'L + 0 9 2 7 : + 0 5 0 1', 'N + 0 9 2 7 : + 0 5 0 1',
            'N + 0 9 2 7 : + 0 5 0 1', #'N + 0 9 2 7 : + 0 5 0 1'
        ]
        debug_actions = [
            'N + 1 1 6 5 : + 0 4 4 3', 'N + 1 1 7 0 : + 0 4 1 8', 
            'N + 1 1 7 5 : + 0 3 9 4', 'N + 1 1 8 1 : + 0 3 7 0', 
            'N + 1 1 8 4 : + 0 3 5 8', 'N + 1 1 8 9 : + 0 3 3 3', 
            'N + 1 1 9 4 : + 0 3 0 9', 'N + 1 1 9 7 : + 0 2 9 7', 
            'N + 1 1 9 7 : + 0 2 9 7', 'N + 1 1 9 7 : + 0 2 9 7', 
            'N + 1 1 9 7 : + 0 2 9 7', 'N + 1 1 9 7 : + 0 2 9 7', 
            'L + 1 1 9 7 : + 0 2 9 7', 'N + 1 1 9 7 : + 0 2 9 7', 
            'N + 1 1 9 7 : + 0 2 9 7'
        ]
        debug_actions = [
            'N + 1 1 6 5 : + 0 4 4 3', 'N + 1 1 7 0 : + 0 4 1 8', 
            'N + 1 1 7 5 : + 0 3 9 4', 'N + 1 1 8 1 : + 0 3 7 0', 
            'N + 1 1 8 4 : + 0 3 5 8', 'N + 1 1 8 9 : + 0 3 3 3', 
            'N + 1 1 9 4 : + 0 3 0 9', 'N + 1 1 9 7 : + 0 2 9 7', 
            'N + 1 1 9 7 : + 0 2 9 7', 'N + 1 1 9 7 : + 0 2 9 7', 
            'N + 1 1 9 7 : + 0 2 9 7', 'N + 1 1 9 7 : + 0 2 9 7', 
            'N + 1 1 9 7 : + 0 2 9 7', 'N + 1 1 9 7 : + 0 2 9 7', 
            'N + 1 1 9 7 : + 0 2 9 7'
        ]
        debug_actions = ['N + 0 0 4 0 : + 0 2 0 4', 'N + 0 1 3 8 : + 0 1 9 0', 
                         'N + 0 2 7 4 : + 0 3 8 3', 'N + 0 5 0 1 : + 0 1 7 3', 
                         'L + 0 4 7 3 : + 0 0 8 7', 'N + 0 1 0 9 : + 0 3 4 4', 
                         'N + 0 0 5 2 : + 0 1 9 4', 'N + 0 3 6 5 : + 0 2 3 2', 
                         'N + 0 3 8 9 : + 0 2 4 5', 'N + 0 0 2 0 : + 0 0 5 9', 
                         'N + 0 4 7 3 : + 0 1 5 7', 'L + 0 1 9 1 : + 0 0 8 7', 
                         'L + 0 1 9 1 : + 0 0 8 7', 'N + 0 3 4 3 : + 0 2 6 3', ]
                         #'N + 0 2 0 5 : + 0 1 3 3']
        previous_actions = []
        for action in debug_actions[-8:]:
            action = action.replace('1 1', '0 4')
            x, y, action_type = parse_action_string(action)
            previous_actions.append((action_type, (x, y)))
        positions = [
            'N + 0 9 2 7 : + 0 5 0 1', 'N + 0 9 1 8 : + 0 4 9 2', 
            'N + 0 9 0 8 : + 0 4 8 3', 'N + 0 8 9 8 : + 0 4 7 4', 
            'N + 0 8 8 9 : + 0 4 6 5', 'N + 0 8 8 0 : + 0 4 5 6', 
            'N + 0 8 7 0 : + 0 4 4 7', 'N + 0 8 6 0 : + 0 4 3 8', 
            'N + 0 8 5 1 : + 0 4 2 9', 'N + 0 8 4 2 : + 0 4 2 0', 
            'N + 0 8 3 2 : + 0 4 1 1', 'N + 0 8 3 2 : + 0 4 1 1'
        ]
        positions = [
            #'L + 1 1 9 7 : + 0 2 9 7', 'N + 1 1 9 7 : + 0 2 9 7', 
            'N + 1 1 9 7 : + 0 2 9 7', 'N + 1 1 9 7 : + 0 2 9 7', 
            'N + 1 1 7 9 : + 0 3 0 3', 'N + 1 1 4 2 : + 0 3 1 4', 
            'N + 1 1 0 6 : + 0 3 2 6', 'N + 1 0 6 9 : + 0 3 3 7', 
            'N + 1 0 5 1 : + 0 3 4 3', 'N + 1 0 1 4 : + 0 3 5 4', 
            'N + 0 9 7 8 : + 0 3 6 5', 'N + 0 9 4 2 : + 0 3 7 7', 
            'N + 0 9 0 5 : + 0 3 8 8', 'N + 0 8 6 8 : + 0 4 0 0', 
            'N + 0 8 3 2 : + 0 4 1 1'
        ]
        positions = ['N + 0 2 0 5 : + 0 1 3 3', 'N + 0 0 7 6 : + 0 3 4 5', 
                     'N + 0 3 1 8 : + 0 3 3 3', 'N + 0 2 5 4 : + 0 2 9 0', 
                     'N + 0 1 0 6 : + 0 1 6 4', 'N + 0 0 7 4 : + 0 2 8 4', 
                     'N + 0 0 2 4 : + 0 0 4 1', 'N + 0 1 5 0 : + 0 3 8 3', 
                     'N + 0 4 0 5 : + 0 1 6 8', 'N + 0 0 5 4 : + 0 3 2 4', 
                     'N + 0 2 9 0 : + 0 1 4 1', 'N + 0 4 0 2 : + 0 0 0 9', 
                     'N + 0 3 0 7 : + 0 3 3 2', 'N + 0 2 2 0 : + 0 3 7 1', 
                     'N + 0 0 8 2 : + 0 1 5 1']
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
                    #mouse_position = position.split('~')
                    #mouse_position = [int(item) for item in mouse_position]
                    #mouse_position = '+ 0 8 1 5 : + 0 3 3 5'
                if True and DEBUG_TEACHER_FORCING:
                    position = positions[0]
                    positions = positions[1:]
                    x, y, action_type = parse_action_string(position)
                    mouse_position = (x, y)
                if False:
                    previous_actions.append((action_type, mouse_position))
                #previous_actions = [(action_type, mouse_position)]
                
                # Log the start time
                start_time = time.time()
                
                # Predict the next frame based on the previous frames and actions
                if DEBUG_TEACHER_FORCING:
                    print ('predicting', f"record_10003/image_{117+len(previous_frames)}.png")

                next_frame, next_frame_append = predict_next_frame(previous_frames, previous_actions)
                # Load and append the corresponding ground truth image instead of model output
                print ('here4', len(previous_frames))
                if True and DEBUG_TEACHER_FORCING:
                    img = Image.open(f"record_10003/image_{117+len(previous_frames)}.png")
                    previous_frames.append(img)
                elif True:
                    previous_frames.append(next_frame_append)
                #previous_frames = []
                
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
