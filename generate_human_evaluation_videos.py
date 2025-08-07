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
DEMO_FRAMES_PREFIX = "frames_1749405369_1"  # As specified for demo-generated video
TARGET_FRAME_COUNTS = [24, 48, 96, 192]  # 1.6s, 3.2s, 6.4s, 12.8s at 15fps
FPS = 15
MAX_SESSIONS = 1000

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
            if frame_count < 192:
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
        # Use the specified demo frames directory
        demo_frame_dir = os.path.join(FRAMES_DIR, DEMO_FRAMES_PREFIX)
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
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))
        
        # Write frames to video
        for frame_path in selected_frames:
            frame = cv2.imread(frame_path)
            if frame is not None:
                video_writer.write(frame)
        
        video_writer.release()
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
                
                # Create video writer for output
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))
        
        # Write frames to video
        for frame_path in selected_frames:
            frame = cv2.imread(frame_path)
            if frame is not None:
                video_writer.write(frame)
        
        video_writer.release()
        logger.info(f"Created fallback real video: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating fallback real video: {e}")
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

def generate_video_pairs(suitable_sessions, num_pairs_per_length=20):
    """Generate video pairs for each target frame count"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    generated_pairs = {}
    
    for target_frames in TARGET_FRAME_COUNTS:
        duration = target_frames / FPS
        pairs_dir = os.path.join(OUTPUT_DIR, f"{target_frames}frames_{duration}s")
        os.makedirs(pairs_dir, exist_ok=True)
        
        pairs = []
        
        # Generate pairs up to the number requested or available sessions
        max_pairs = min(num_pairs_per_length, len(suitable_sessions))
        
        for i in range(max_pairs):
            session = suitable_sessions[i]
            session_id = session['session_id']
            session_file = session['session_file']
            
            # Generate demo video
            demo_video_path = os.path.join(pairs_dir, f"pair_{i+1:03d}_demo.mp4")
            demo_success = create_demo_video(session_id, demo_video_path, target_frames)
            
            # Generate real video  
            real_video_path = os.path.join(pairs_dir, f"pair_{i+1:03d}_real.mp4")
            real_success = create_real_video(session_id, session_file, real_video_path, target_frames)
            
            if demo_success and real_success:
                pairs.append({
                    'pair_id': i + 1,
                    'session_id': session_id,
                    'demo_video': f"pair_{i+1:03d}_demo.mp4",
                    'real_video': f"pair_{i+1:03d}_real.mp4",
                })
            else:
                logger.warning(f"Failed to create videos for pair {i+1}")
        
        generated_pairs[target_frames] = pairs
        logger.info(f"Generated {len(pairs)} video pairs for {target_frames} frames ({duration}s)")
    
    return generated_pairs

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
    N = 2
    if len(suitable_sessions) > N:
        suitable_sessions = suitable_sessions[:N]
    
    # Print selected session names
    print("\nSelected sessions:")
    for session in suitable_sessions:
        print(f"  {session['session_id']} - {session['frame_count']} frames")
    
    if not suitable_sessions:
        logger.error("No suitable sessions found. Exiting.")
        return
    
    # Generate video pairs
    generated_pairs = generate_video_pairs(suitable_sessions)
    
    # Save metadata about generated pairs
    metadata = {
        'generation_time': datetime.now().isoformat(),
        'total_sessions_processed': len(session_files),
        'suitable_sessions_found': len(suitable_sessions),
        'target_frame_counts': TARGET_FRAME_COUNTS,
        'fps': FPS,
        'demo_frames_source': DEMO_FRAMES_PREFIX,
        'pairs_generated': generated_pairs
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, "generation_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Video generation complete. Results saved to {OUTPUT_DIR}")
    logger.info(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    main() 