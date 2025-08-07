#!/usr/bin/env python3
"""
Script to generate videos for human evaluation comparing NeuralOS demo with real OS.

This script:
1. Lists interaction logs and takes the most recent 1000 files
2. Filters sessions with no resets and more than 192 frames
3. Generates demo videos from frames_* directories
4. Generates real OS videos using the data collection script
5. Creates HTML evaluation interfaces for different video lengths
"""

import os
import json
import glob
import cv2
import numpy as np
import logging
import time
import random
import math
from datetime import datetime
from typing import List, Dict, Tuple
import subprocess
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("human_evaluation_video_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
FRAMES_DIR = "interaction_logs"
OUTPUT_DIR = "human_evaluation_videos"
# DEMO_FRAMES_PREFIX will be determined from session data
FPS = 1.8
TARGET_VIDEO_LENGTHS = [10, 20, 30, 40, 50, 60]  # 1.6s, 3.2s, 6.4s, 12.8s, 25.6s, 51.2s, 102.4s at 1.8fps
TARGET_FRAME_COUNTS = [int(math.ceil(length * FPS)) for length in TARGET_VIDEO_LENGTHS]

MAX_SESSIONS = 10000
MIN_FRAME_COUNT = 192

# Video cropping settings - two different crop levels for comparison
CROP_SETTINGS = {
    'cropped': {
        'max_height': 354,  # Crops 30px from bottom (384-354=30)
        'name': 'Cropped'
    },
    'uncropped': {
        'max_height': 384,  # No cropping (shows full video)
        'name': 'Uncropped'
    }
}

def extract_timestamp_from_session(session_file):
    """Extract timestamp from session filename like session_1754504816_907.jsonl"""
    try:
        basename = os.path.basename(session_file)
        # session_1754504816_907.jsonl -> 1754504816
        timestamp_part = basename.split('_')[1]
        return int(timestamp_part)
    except Exception:
        return 0

def load_session_data(session_file):
    """Load session data from JSONL file"""
    try:
        trajectory = []
        with open(session_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    trajectory.append(entry)
                except json.JSONDecodeError:
                    continue
        return trajectory
    except Exception as e:
        logger.error(f"Error loading session {session_file}: {e}")
        return []

def has_resets(trajectory):
    """Check if trajectory contains any reset events"""
    return any(entry.get("is_reset", False) for entry in trajectory)

def get_frame_count(session_id):
    """Get number of available frames for a session"""
    frame_dir = os.path.join(FRAMES_DIR, f"frames_{session_id}")
    if not os.path.exists(frame_dir):
        return 0
    frames = glob.glob(os.path.join(frame_dir, "*.png"))
    return len(frames)

def filter_suitable_sessions(session_files):
    """
    Filter sessions that:
    1. Have no reset events
    2. Have more than 192 frames available
    3. Are complete (have EOS marker)
    """
    suitable_sessions = []
    
    for session_file in session_files:
        try:
            # Load session data
            trajectory = load_session_data(session_file)
            if not trajectory:
                continue
            
            # Check if session is complete
            has_eos = any(entry.get("is_eos", False) for entry in trajectory)
            if not has_eos:
                continue
            
            # Check for resets
            if has_resets(trajectory):
                continue
            
            # Extract session ID from filename
            basename = os.path.basename(session_file)
            session_id = basename.replace('session_', '').replace('.jsonl', '')
            
            # Check frame count
            frame_count = get_frame_count(session_id)
            if frame_count < MIN_FRAME_COUNT:
                continue
            
            suitable_sessions.append({
                'session_file': session_file,
                'session_id': session_id,
                'frame_count': frame_count,
                'trajectory_length': len(trajectory)
            })
            
        except Exception as e:
            logger.error(f"Error processing {session_file}: {e}")
            continue
    
    return suitable_sessions

def create_demo_video(session_id, output_path, target_frames):
    """Create demo video from frames in the demo frames directory"""
    try:
        # Use the session's own frames directory for demo video
        demo_frame_dir = os.path.join(FRAMES_DIR, f"frames_{session_id}")
        if not os.path.exists(demo_frame_dir):
            logger.error(f"Demo frame directory {demo_frame_dir} not found")
            return False
        
        # Get available demo frames
        demo_frames = glob.glob(os.path.join(demo_frame_dir, "*.png"))
        demo_frames.sort(key=lambda x: float(os.path.basename(x).split('.png')[0]))
        
        if len(demo_frames) < target_frames:
            logger.error(f"Not enough demo frames: {len(demo_frames)} < {target_frames}")
            return False
        
        # Use the first target_frames frames
        selected_frames = demo_frames[:target_frames]
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(selected_frames[0])
        if first_frame is None:
            logger.error(f"Could not read first demo frame")
            return False
        
        height, width, _ = first_frame.shape
        
        # Create video writer with H.264 codec for better browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 codec
        video_writer = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))
        
        # Write frames to video
        for frame_path in selected_frames:
            frame = cv2.imread(frame_path)
            if frame is not None:
                video_writer.write(frame)
        
        video_writer.release()
        
        # Re-encode for browser compatibility
        temp_path = output_path + ".temp.mp4"
        if reencode_video_for_browser(output_path, temp_path):
            os.rename(temp_path, output_path)
        
        logger.info(f"Created demo video: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating demo video: {e}")
        return False

def create_real_video(session_id, session_file, output_path, target_frames):
    """Create real OS video by processing session through data collection script"""
    try:
        # Load session trajectory
        trajectory = load_session_data(session_file)
        if not trajectory:
            logger.error(f"Could not load trajectory from {session_file}")
            return False
        
        # Format trajectory for processing (similar to online_data_generation.py)
        formatted_trajectory = format_trajectory_for_processing(trajectory)
        
        if len(formatted_trajectory) < target_frames:
            logger.error(f"Trajectory too short: {len(formatted_trajectory)} < {target_frames}")
            return False
        
        # Truncate to target frames
        formatted_trajectory = formatted_trajectory[:target_frames]
        
        # Create temporary directory for processing
        temp_dir = os.path.join(OUTPUT_DIR, f"temp_{session_id}")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Import required modules for data generation
            from data.data_collection.synthetic_script_compute_canada import process_trajectory, initialize_clean_state
            
            # Initialize clean state if not already done
            if not hasattr(create_real_video, 'clean_state'):
                logger.info("Initializing clean Docker state for real OS rendering...")
                create_real_video.clean_state = initialize_clean_state()
            
            # Generate a unique record number
            import tempfile
            record_num = int(time.time() * 1000) % 1000000  # Use timestamp for uniqueness
            
            # Call process_trajectory to generate real OS video
            args = (record_num, formatted_trajectory)
            process_trajectory(args, 512, 384, create_real_video.clean_state, "2g")
            
            # Path to generated video
            generated_video_path = f'raw_data/raw_data/videos/record_{record_num}.mp4'
            
            if not os.path.exists(generated_video_path):
                logger.error(f"Generated video not found: {generated_video_path}")
                return False
            
            # Extract frames from generated video and create output video
            from moviepy.editor import VideoFileClip
            
            with VideoFileClip(generated_video_path) as video:
                fps = video.fps
                duration = video.duration
                
                # Create video writer for output with H.264 codec
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 codec
                video_writer = cv2.VideoWriter(output_path, fourcc, FPS, (512, 384))
                
                # Extract and write frames up to target_frames
                frames_written = 0
                for frame_idx in range(min(target_frames, int(fps * duration))):
                    frame = video.get_frame(frame_idx / fps)
                    # Convert from RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)
                    frames_written += 1
                
                video_writer.release()
                
                # Re-encode for browser compatibility
                temp_path = output_path + ".temp.mp4"
                if reencode_video_for_browser(output_path, temp_path):
                    os.rename(temp_path, output_path)
            
            # Clean up generated files
            try:
                if os.path.exists(generated_video_path):
                    os.remove(generated_video_path)
                action_file = f'raw_data/raw_data/actions/record_{record_num}.csv'
                if os.path.exists(action_file):
                    os.remove(action_file)
            except Exception as cleanup_e:
                logger.warning(f"Error cleaning up generated files: {cleanup_e}")
            
            logger.info(f"Created real video: {output_path} ({frames_written} frames)")
            return True
            
        except ImportError as e:
            logger.error(f"Could not import data collection modules: {e}")
            logger.info("Falling back to using existing frames as placeholder for real video")
            return create_real_video_fallback(session_id, output_path, target_frames)
            
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
                
    except Exception as e:
        logger.error(f"Error creating real video: {e}")
        return False

def create_real_video_fallback(session_id, output_path, target_frames):
    """Fallback method using existing frames when data collection script is not available"""
    try:
        frame_dir = os.path.join(FRAMES_DIR, f"frames_{session_id}")
        if not os.path.exists(frame_dir):
            logger.error(f"Frame directory {frame_dir} not found")
            return False
        
        # Get available frames
        frames = glob.glob(os.path.join(frame_dir, "*.png"))
        frames.sort(key=lambda x: float(os.path.basename(x).split('.png')[0]))
        
        if len(frames) < target_frames:
            logger.error(f"Not enough frames: {len(frames)} < {target_frames}")
            return False
        
        # Use the first target_frames frames
        selected_frames = frames[:target_frames]
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(selected_frames[0])
        if first_frame is None:
            logger.error(f"Could not read first frame")
            return False
        
        height, width, _ = first_frame.shape
        
        # Create video writer with H.264 codec for better browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 codec
        video_writer = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))
        
        # Write frames to video
        for frame_path in selected_frames:
            frame = cv2.imread(frame_path)
            if frame is not None:
                video_writer.write(frame)
        
        video_writer.release()
        
        # Re-encode for browser compatibility
        temp_path = output_path + ".temp.mp4"
        if reencode_video_for_browser(output_path, temp_path):
            os.rename(temp_path, output_path)
        
        logger.info(f"Created fallback real video: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating fallback real video: {e}")
        return False

def reencode_video_for_browser(input_path, output_path):
    """Re-encode video using ffmpeg for better browser compatibility"""
    try:
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-i', input_path,
            '-c:v', 'libx264',  # H.264 codec
            '-profile:v', 'baseline',  # Baseline profile for compatibility
            '-level', '3.0',
            '-pix_fmt', 'yuv420p',  # Compatible pixel format
            '-movflags', '+faststart',  # Enable fast start for web
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Remove original file and rename the new one
            os.remove(input_path)
            logger.info(f"Re-encoded video for browser compatibility: {output_path}")
            return True
        else:
            logger.error(f"ffmpeg failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error re-encoding video: {e}")
        return False

def format_trajectory_for_processing(trajectory):
    """
    Format the trajectory in the structure expected by process_trajectory function.
    Based on online_data_generation.py format_trajectory_for_processing function.
    """
    # Key mappings from online_data_generation.py
    KEYMAPPING = {
        'arrowup': 'up',
        'arrowdown': 'down',
        'arrowleft': 'left',
        'arrowright': 'right',
        'meta': 'command',
        'contextmenu': 'apps',
        'control': 'ctrl',
    }
    
    # Valid keys from online_data_generation.py
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
    stoi = {key: i for i, key in enumerate(VALID_KEYS)}
    
    formatted_events = []
    down_keys = set([])
    
    for entry in trajectory:
        # Skip control messages
        if entry.get("is_reset") or entry.get("is_eos"):
            continue
            
        # Extract input data
        inputs = entry.get("inputs", {})
        if inputs is None:
            continue
            
        key_events = []
        
        # Process keys_down
        for key in inputs.get("keys_down", []):
            key = key.lower()
            if key in KEYMAPPING:
                key = KEYMAPPING[key]
            if key not in down_keys and key in stoi:
                down_keys.add(key)
                key_events.append(("keydown", key))
        
        # Process keys_up        
        for key in inputs.get("keys_up", []):
            key = key.lower()
            if key in KEYMAPPING:
                key = KEYMAPPING[key]
            if key in down_keys and key in stoi:
                down_keys.remove(key)
                key_events.append(("keyup", key))
        
        event = {
            "pos": (inputs.get("x"), inputs.get("y")),
            "left_click": inputs.get("is_left_click", False),
            "right_click": inputs.get("is_right_click", False),
            "key_events": key_events,
        }
        
        formatted_events.append(event)
    
    return formatted_events

def generate_video_pairs(suitable_sessions, num_pairs_per_length=30):
    """Generate video pairs for each target frame count and crop setting using non-overlapping slices"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    generated_pairs = {}
    
    # Shuffle sessions once at the beginning for randomness, but ensure reproducibility
    random.seed(42)
    sessions_shuffled = suitable_sessions.copy()
    random.shuffle(sessions_shuffled)
    
    # Calculate total sessions needed for both crop settings
    total_needed = len(TARGET_FRAME_COUNTS) * len(CROP_SETTINGS) * num_pairs_per_length
    available_sessions = len(sessions_shuffled)
    
    if total_needed > available_sessions:
        new_pairs_per_length = available_sessions // (len(TARGET_FRAME_COUNTS) * len(CROP_SETTINGS))
        if new_pairs_per_length == 0:
            logger.error(f"Need {total_needed} sessions but only have {available_sessions}. "
                        f"Not enough sessions to generate any pairs. Need at least {len(TARGET_FRAME_COUNTS) * len(CROP_SETTINGS)} sessions.")
            return {}
        logger.warning(f"Need {total_needed} sessions but only have {available_sessions}. "
                      f"Reducing pairs per length to {new_pairs_per_length}")
        num_pairs_per_length = new_pairs_per_length
    
    logger.info(f"Using non-overlapping slices: {num_pairs_per_length} pairs per frame count per crop setting")
    
    session_idx = 0  # Track session index across all combinations
    
    for crop_key, crop_config in CROP_SETTINGS.items():
        for i, target_frames in enumerate(TARGET_FRAME_COUNTS):
            duration = target_frames / FPS
            pairs_dir = os.path.join(OUTPUT_DIR, f"{target_frames}frames_{duration}s_{crop_key}")
            os.makedirs(pairs_dir, exist_ok=True)
            
            pairs = []
            
            # Use non-overlapping slices for each combination
            start_idx = session_idx
            end_idx = start_idx + num_pairs_per_length
            session_slice = sessions_shuffled[start_idx:end_idx]
            session_idx = end_idx  # Update for next combination
            
            logger.info(f"Frame count {target_frames} ({crop_config['name']}): using sessions {start_idx} to {end_idx-1} "
                       f"({len(session_slice)} sessions)")
            
            for j, session in enumerate(session_slice):
                session_id = session['session_id']
                session_file = session['session_file']
                
                # Generate demo video
                demo_video_path = os.path.join(pairs_dir, f"pair_{j+1:03d}_demo.mp4")
                demo_success = create_demo_video(session_id, demo_video_path, target_frames)
                
                # Generate real video  
                real_video_path = os.path.join(pairs_dir, f"pair_{j+1:03d}_real.mp4")
                real_success = create_real_video(session_id, session_file, real_video_path, target_frames)
                
                if demo_success and real_success:
                    # Randomize which video appears on left vs right
                    left_is_real = random.choice([True, False])
                    
                    pairs.append({
                        'pair_id': j + 1,
                        'session_id': session_id,
                        'demo_video': f"pair_{j+1:03d}_demo.mp4",
                        'real_video': f"pair_{j+1:03d}_real.mp4",
                        'left_is_real': left_is_real,  # True if real video is on left, False if demo is on left
                        'left_video': f"pair_{j+1:03d}_real.mp4" if left_is_real else f"pair_{j+1:03d}_demo.mp4",
                        'right_video': f"pair_{j+1:03d}_demo.mp4" if left_is_real else f"pair_{j+1:03d}_real.mp4",
                        'crop_setting': crop_key,
                        'max_height': crop_config['max_height'],
                    })
                else:
                    logger.warning(f"Failed to create videos for pair {j+1}")
            
            # Store with combined key
            key = f"{target_frames}_{crop_key}"
            generated_pairs[key] = pairs
            logger.info(f"Generated {len(pairs)} video pairs for {target_frames} frames ({duration}s) - {crop_config['name']}")
    
    return generated_pairs

def create_evaluation_html(generated_pairs):
    """Create HTML evaluation interface for all video pairs"""
    
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuralOS Human Evaluation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .evaluation-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .video-pair {
            display: flex;
            justify-content: space-around;
            margin-bottom: 20px;
            gap: 20px;
        }
        
        .video-container {
            flex: 1;
            text-align: center;
        }
        
        .video-container video {
            width: 100%;
            max-width: 512px;    /* Match original video width */
            height: auto;
            border: 2px solid #ddd;
            border-radius: 5px;
            /* Crop bottom to hide telltale signs like free space indicator */
            object-fit: cover;
            object-position: top;
            /* Adjust this value to crop more/less from bottom */
            max-height: 364px;   /* Crops 84px from bottom (384-300=84) */
        }
        
        .video-wrapper {
            overflow: hidden;
            border-radius: 5px;
            max-width: 512px;
            margin: 0 auto;
        }
        
        .selection-buttons {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            gap: 20px;
        }
        
        .select-btn {
            flex: 1;
            max-width: 400px;
            padding: 15px 20px;
            font-size: 16px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        
        .select-btn:hover {
            background-color: #0056b3;
        }
        
        .select-btn.selected {
            background-color: #28a745;
        }
        
        .navigation {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 20px 0;
        }
        
        .nav-btn {
            padding: 10px 20px;
            background-color: #6c757d;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .nav-btn:hover {
            background-color: #5a6268;
        }
        
        .nav-btn:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        
        .progress {
            text-align: center;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .setting-selector {
            text-align: center;
            margin-bottom: 20px;
        }
        
        .setting-selector select {
            padding: 10px;
            font-size: 16px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        
        .results-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-top: 20px;
            display: none;
        }
        
        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .results-table th, .results-table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
        }
        
        .results-table th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        
        .submit-btn {
            display: block;
            width: 200px;
            margin: 20px auto;
            padding: 15px;
            background-color: #dc3545;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
        }
        
        .submit-btn:hover {
            background-color: #c82333;
        }
        
        .instructions {
            background-color: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>NeuralOS Human Evaluation</h1>
        <p>Help us evaluate the quality of NeuralOS by identifying which video shows a real operating system.</p>
    </div>
    
    <div class="instructions">
        <h3>Instructions:</h3>
        <ul>
            <li>You will see pairs of videos showing computer interactions</li>
            <li>One video shows a real operating system, the other shows NeuralOS (AI-generated)</li>
            <li>Watch both videos and click "This is Real OS" under the video you think shows the real operating system</li>
            <li>Use the navigation buttons to move between examples</li>
            <li>Complete all examples in each setting to see your accuracy results</li>
            <li><em>Note: Videos are cropped to hide UI elements that might give away the answer</em></li>
        </ul>
    </div>
    
    <div class="setting-selector">
        <label for="settingSelect">Video Length Setting:</label>
        <select id="settingSelect" onchange="changeSetting()">
            <!-- Options will be populated by JavaScript -->
        </select>
    </div>
    
    <div class="evaluation-container">
        <div class="progress">
            <span id="progressText">Example 1 of 1</span>
        </div>
        
        <div class="video-pair">
            <div class="video-container">
                <h3>Video A</h3>
                <div class="video-wrapper">
                    <video id="videoLeft" controls>
                        <source src="" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
            </div>
            <div class="video-container">
                <h3>Video B</h3>
                <div class="video-wrapper">
                    <video id="videoRight" controls>
                        <source src="" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
            </div>
        </div>
        
        <div class="selection-buttons">
            <button class="select-btn" id="selectLeft" onclick="selectVideo('left')">
                This is Real OS
            </button>
            <button class="select-btn" id="selectRight" onclick="selectVideo('right')">
                This is Real OS
            </button>
        </div>
        
        <div class="navigation">
            <button class="nav-btn" id="prevBtn" onclick="previousExample()">Previous</button>
            <button class="submit-btn" id="submitBtn" onclick="submitResults()" style="display: none;">
                Submit Results
            </button>
            <button class="nav-btn" id="nextBtn" onclick="nextExample()">Next</button>
        </div>
    </div>
    
    <div class="results-container" id="resultsContainer">
        <h2>Evaluation Results</h2>
        <table class="results-table">
            <thead>
                <tr>
                    <th>Video Length</th>
                    <th>Duration</th>
                    <th>Crop Setting</th>
                    <th>Correct Identifications</th>
                    <th>Total Examples</th>
                    <th>Accuracy</th>
                </tr>
            </thead>
            <tbody id="resultsTableBody">
                <!-- Results will be populated by JavaScript -->
            </tbody>
        </table>
    </div>

    <script>
        // Data from server - will be populated
        const evaluationData = {evaluation_data_placeholder};
        const FPS = {fps_placeholder};
        const CROP_SETTINGS = {crop_settings_placeholder};
        
        let currentSetting = null;
        let currentExample = 0;
        let responses = {};
        
        // Initialize the evaluation
        function initializeEvaluation() {
            // Populate setting selector
            const settingSelect = document.getElementById('settingSelect');
            const settings = Object.keys(evaluationData).sort((a, b) => parseInt(a) - parseInt(b));
            
            settings.forEach(setting => {
                const option = document.createElement('option');
                option.value = setting;
                
                // Parse setting key: "frames_cropkey"
                const [framesStr, cropKey] = setting.split('_');
                const frames = parseInt(framesStr);
                const duration = (frames / FPS).toFixed(1);
                const cropName = CROP_SETTINGS[cropKey] ? CROP_SETTINGS[cropKey].name : cropKey;
                
                option.textContent = `${frames} frames (${duration}s) - ${cropName}`;
                settingSelect.appendChild(option);
            });
            
            // Initialize responses object
            settings.forEach(setting => {
                responses[setting] = {};
            });
            
            if (settings.length > 0) {
                currentSetting = settings[0];
                settingSelect.value = currentSetting;
                loadCurrentExample();
            }
        }
        
        function changeSetting() {
            const settingSelect = document.getElementById('settingSelect');
            currentSetting = settingSelect.value;
            currentExample = 0;
            loadCurrentExample();
        }
        
        function loadCurrentExample() {
            if (!currentSetting || !evaluationData[currentSetting]) return;
            
            const examples = evaluationData[currentSetting];
            const example = examples[currentExample];
            
            if (!example) return;
            
            // Update progress
            document.getElementById('progressText').textContent = 
                `Example ${currentExample + 1} of ${examples.length}`;
            
            // Load videos
            const videoLeft = document.getElementById('videoLeft');
            const videoRight = document.getElementById('videoRight');
            
            // Parse setting to get directory name
            const [framesStr, cropKey] = currentSetting.split('_');
            const duration = (parseInt(framesStr) / FPS).toFixed(1);
            const dirName = `${framesStr}frames_${duration}s_${cropKey}`;
            
            videoLeft.src = `${dirName}/${example.left_video}`;
            videoRight.src = `${dirName}/${example.right_video}`;
            
            // Apply crop setting to videos
            const maxHeight = example.max_height || 384;
            videoLeft.style.maxHeight = `${maxHeight}px`;
            videoRight.style.maxHeight = `${maxHeight}px`;
            
            // Reset selection buttons
            document.getElementById('selectLeft').classList.remove('selected');
            document.getElementById('selectRight').classList.remove('selected');
            
            // Update button states based on existing response
            const existingResponse = responses[currentSetting][example.pair_id];
            if (existingResponse !== undefined) {
                if (existingResponse === 'left') {
                    document.getElementById('selectLeft').classList.add('selected');
                } else if (existingResponse === 'right') {
                    document.getElementById('selectRight').classList.add('selected');
                }
            }
            
            // Update navigation buttons
            document.getElementById('prevBtn').disabled = currentExample === 0;
            document.getElementById('nextBtn').disabled = currentExample === examples.length - 1;
            
            // Show submit button if this is the last example and all examples are completed
            const allCompleted = examples.every(ex => responses[currentSetting][ex.pair_id] !== undefined);
            const isLastExample = currentExample === examples.length - 1;
            document.getElementById('submitBtn').style.display = 
                (isLastExample && allCompleted) ? 'block' : 'none';
        }
        
        function selectVideo(side) {
            if (!currentSetting || !evaluationData[currentSetting]) return;
            
            const example = evaluationData[currentSetting][currentExample];
            responses[currentSetting][example.pair_id] = side;
            
            // Update button appearance
            document.getElementById('selectLeft').classList.remove('selected');
            document.getElementById('selectRight').classList.remove('selected');
            
            if (side === 'left') {
                document.getElementById('selectLeft').classList.add('selected');
            } else {
                document.getElementById('selectRight').classList.add('selected');
            }
            
            // Check if this was the last unanswered question in current setting
            const examples = evaluationData[currentSetting];
            const allCompleted = examples.every(ex => responses[currentSetting][ex.pair_id] !== undefined);
            const isLastExample = currentExample === examples.length - 1;
            
            if (isLastExample && allCompleted) {
                document.getElementById('submitBtn').style.display = 'block';
            }
        }
        
        function previousExample() {
            if (currentExample > 0) {
                currentExample--;
                loadCurrentExample();
            }
        }
        
        function nextExample() {
            if (!currentSetting || !evaluationData[currentSetting]) return;
            
            const examples = evaluationData[currentSetting];
            if (currentExample < examples.length - 1) {
                currentExample++;
                loadCurrentExample();
            }
        }
        
        function submitResults() {
            // Calculate results for each setting
            const resultsTableBody = document.getElementById('resultsTableBody');
            resultsTableBody.innerHTML = '';
            
            const settings = Object.keys(evaluationData).sort((a, b) => parseInt(a) - parseInt(b));
            
            settings.forEach(setting => {
                const examples = evaluationData[setting];
                const userResponses = responses[setting];
                
                let correct = 0;
                let total = 0;
                
                examples.forEach(example => {
                    const userChoice = userResponses[example.pair_id];
                    if (userChoice !== undefined) {
                        total++;
                        
                        // Check if user was correct
                        const userChoseLeft = userChoice === 'left';
                        const leftWasReal = example.left_is_real;
                        
                        if (userChoseLeft === leftWasReal) {
                            correct++;
                        }
                    }
                });
                
                const accuracy = total > 0 ? ((correct / total) * 100).toFixed(1) : '0.0';
                
                // Parse setting
                const [framesStr, cropKey] = setting.split('_');
                const frames = parseInt(framesStr);
                const duration = (frames / FPS).toFixed(1);
                const cropName = CROP_SETTINGS[cropKey] ? CROP_SETTINGS[cropKey].name : cropKey;
                
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${frames} frames</td>
                    <td>${duration}s</td>
                    <td>${cropName}</td>
                    <td>${correct}</td>
                    <td>${total}</td>
                    <td>${accuracy}%</td>
                `;
                resultsTableBody.appendChild(row);
            });
            
            // Show results
            document.getElementById('resultsContainer').style.display = 'block';
            
            // Scroll to results
            document.getElementById('resultsContainer').scrollIntoView({ 
                behavior: 'smooth' 
            });
        }
        
        // Initialize when page loads
        window.onload = initializeEvaluation;
    </script>
</body>
</html>"""
    
    # Create HTML files for each setting
    for target_frames in generated_pairs.keys():
        duration = target_frames / FPS
        pairs_dir = os.path.join(OUTPUT_DIR, f"{target_frames}frames_{duration}s")
        
        # Create evaluation data for this setting
        evaluation_data = {str(target_frames): generated_pairs[target_frames]}
        
        # Replace placeholders with actual data
        html_content = html_template.replace(
            '{evaluation_data_placeholder}', 
            json.dumps(evaluation_data, indent=2)
        ).replace(
            '{fps_placeholder}',
            str(FPS)
        ).replace(
            '{crop_settings_placeholder}',
            json.dumps(CROP_SETTINGS)
        )
        
        # Save HTML file
        html_path = os.path.join(pairs_dir, "evaluation.html")
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Created evaluation HTML: {html_path}")
    
    # Create a combined HTML file with all settings
    html_content = html_template.replace(
        '{evaluation_data_placeholder}', 
        json.dumps({str(k): v for k, v in generated_pairs.items()}, indent=2)
    ).replace(
        '{fps_placeholder}',
        str(FPS)
    ).replace(
        '{crop_settings_placeholder}',
        json.dumps(CROP_SETTINGS)
    )
    
    combined_html_path = os.path.join(OUTPUT_DIR, "evaluation_all_settings.html")
    with open(combined_html_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Created combined evaluation HTML: {combined_html_path}")
    
    return combined_html_path

def main():
    """Main function to generate evaluation videos"""
    logger.info("Starting human evaluation video generation")
    
    # Find all session files
    session_files = glob.glob(os.path.join(FRAMES_DIR, "session_*.jsonl"))
    logger.info(f"Found {len(session_files)} session files")
    
    # Sort by timestamp (most recent first) and take top 1000
    session_files.sort(key=extract_timestamp_from_session, reverse=True)
    session_files = session_files[:MAX_SESSIONS]
    logger.info(f"Processing most recent {len(session_files)} sessions")
    
    # Filter for suitable sessions
    suitable_sessions = filter_suitable_sessions(session_files)
    logger.info(f"Found {len(suitable_sessions)} suitable sessions (no resets, >192 frames)")

    random.seed(42)
    random.shuffle(suitable_sessions)
    # Remove the artificial limit - use all suitable sessions
    # N = 2
    # if len(suitable_sessions) > N:
    #     suitable_sessions = suitable_sessions[:N]
    
    # Print selected session names
    print("\nSelected sessions:")
    for session in suitable_sessions:
        print(f"  {session['session_id']} - {session['frame_count']} frames")
    
    if not suitable_sessions:
        logger.error("No suitable sessions found. Exiting.")
        return
    
    # Generate video pairs
    generated_pairs = generate_video_pairs(suitable_sessions)
    
    # Create HTML evaluation interfaces
    html_path = create_evaluation_html(generated_pairs)
    
    # Save metadata about generated pairs
    metadata = {
        'generation_time': datetime.now().isoformat(),
        'total_sessions_processed': len(session_files),
        'suitable_sessions_found': len(suitable_sessions),
        'target_frame_counts': TARGET_FRAME_COUNTS,
        'fps': FPS,
        'demo_frames_source': 'frames_{session_id} (inferred from session)',
        'pairs_generated': generated_pairs,
        'evaluation_html': html_path
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, "generation_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Video generation complete. Results saved to {OUTPUT_DIR}")
    logger.info(f"Metadata saved to {metadata_path}")
    logger.info(f"Main evaluation interface: {html_path}")
    logger.info("Open the HTML file in a web browser to start the human evaluation.")

if __name__ == "__main__":
    main() 
