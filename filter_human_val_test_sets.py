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
import sys
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
        logging.FileHandler("filter_human_evaluation_video_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
SCREEN_WIDTH = 512
SCREEN_HEIGHT = 384
MEMORY_LIMIT = "2g"
MAX_SESSIONS = 2000
MIN_FRAME_COUNT = 192
N = 100
SAMPLES = 3
OUTPUT_DIR = "evaluation_frames"

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
    frame_dir = os.path.join('../neuralos-demo/interaction_logs', f"frames_{session_id}")
    if not os.path.exists(frame_dir):
        assert False, 'no frames found'
        #return 0
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

def create_real_video(session_id, session_file, output_path, output_csv_path):
    """Create real OS video by processing session through data collection script"""
    try:
        # Load session trajectory
        trajectory = load_session_data(session_file)
        logger.info(f"processing trajectory from {session_file}")
        if not trajectory:
            logger.error(f"Could not load trajectory from {session_file}")
            return False
        
        # Format trajectory for processing (similar to online_data_generation.py)
        formatted_trajectory = format_trajectory_for_processing(trajectory)
        
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
            process_trajectory(args, SCREEN_WIDTH, SCREEN_HEIGHT, create_real_video.clean_state, MEMORY_LIMIT)
            
            # Path to generated video
            generated_video_path = f'raw_data/raw_data/videos/record_{record_num}.mp4'
            action_file = f'raw_data/raw_data/actions/record_{record_num}.csv'
            
            if not os.path.exists(generated_video_path):
                logger.error(f"Generated video not found: {generated_video_path}")
                assert False
            shutil.copy(generated_video_path, output_path)
            shutil.copy(action_file, output_csv_path)
            # Clean up generated files
            try:
                if os.path.exists(generated_video_path):
                    os.remove(generated_video_path)
                if os.path.exists(action_file):
                    os.remove(action_file)
            except Exception as cleanup_e:
                logger.warning(f"Error cleaning up generated files: {cleanup_e}")
            
            logger.info(f"Created real video: {output_path}")
            return True, formatted_trajectory
            
        except ImportError as e:
            logger.error(f"Could not import data collection modules: {e}")
            raise e
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
                
    except Exception as e:
        logger.error(f"Error creating real video: {e}")
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

def main():
    """Main function to generate evaluation videos"""
    logger.info("Starting human evaluation video generation")

    all_settings = {
        'train': '../neuralos-demo-datagen_aug15_used_for_realdatagen/interaction_logs',
        'test': '../neuralos-demo-datagen_aug15_used_for_realdatagen/new_interaction_logs',
    }
    all_sessions = {}
    for setting in all_settings:
        frames_dir = all_settings[setting]
        print (setting, frames_dir)
        # Find all session files
        session_files = glob.glob(os.path.join(frames_dir, "session_*.jsonl"))
        logger.info(f"Found {len(session_files)} session files")
        
        # Sort by timestamp (most recent first) and take top 2000
        session_files.sort(key=extract_timestamp_from_session, reverse=True)
        session_files = session_files[:MAX_SESSIONS]
        logger.info(f"Processing most recent {len(session_files)} sessions")
        
        # Filter for suitable sessions
        suitable_sessions = filter_suitable_sessions(session_files)
        logger.info(f"Found {len(suitable_sessions)} suitable sessions (no resets, >192 frames)")

        random.seed(42)
        random.shuffle(suitable_sessions)
        a = []
        for session in suitable_sessions:
            session_file = session['session_file']
            t = load_session_data(session_file)
            #import pdb; pdb.set_trace()
            if len(t) >= 368 and len(t) <= 371:
                import pdb; pdb.set_trace()
                continue
            a.append(session)
        suitable_sessions = a
        assert len(suitable_sessions) > N, len(suitable_sessions)
        if len(suitable_sessions) > N:
            suitable_sessions = suitable_sessions[:N]
        all_sessions[setting] = suitable_sessions

    for sample_id in range(SAMPLES):
        for setting in all_settings:
            if setting == 'train' and sample_id == 0:
                print ('skipping')
                continue
            suitable_sessions = all_sessions[setting]
            video_dir = os.path.join(OUTPUT_DIR, setting)
            os.makedirs(video_dir, exist_ok=True)
            for session in suitable_sessions:
                session_id = session['session_id']
                session_file = session['session_file']
                video_path = os.path.join(video_dir, f"{sample_id}_{session_id}.mp4")
                #import pdb; pdb.set_trace()
                csv_path = os.path.join(video_dir, f"{sample_id}_{session_id}.csv")
                real_success, formatted_trajectory = create_real_video(session_id, session_file, video_path, csv_path)
                assert real_success

if __name__ == "__main__":
    main() 
