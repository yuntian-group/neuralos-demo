#!/usr/bin/env python3
import os
import json
import glob
import time
import sqlite3
import logging
import cv2
import numpy as np
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Tuple
from omegaconf import OmegaConf
from computer.util import load_model_from_config
from PIL import Image
import io
import torch
from einops import rearrange
import webdataset as wds
import pandas as pd
import ast
import pickle
from moviepy.editor import VideoFileClip

# Import the existing functions
from data.data_collection.synthetic_script_compute_canada import process_trajectory, initialize_clean_state

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trajectory_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
DB_FILE = "trajectory_processor.db"
FRAMES_DIR = "interaction_logs"
OUTPUT_DIR = 'train_dataset_encoded_online'
os.makedirs(OUTPUT_DIR, exist_ok=True)
SCREEN_WIDTH = 512
SCREEN_HEIGHT = 384
MEMORY_LIMIT = "2g"

# load autoencoder
config = OmegaConf.load('../computer/autoencoder/config_kl4_lr4.5e6_load_acc1_512_384_mar10_keyboard_init_16_contmar15_acc1.yaml')
autoencoder = load_model_from_config(config, '../computer/autoencoder/saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_512_384_mar10_keyboard_init_16_cont_mar15_acc1_cont_1e6_cont_2e7_cont/model-2076000.ckpt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder = autoencoder.to(device)

def initialize_database():
    """Initialize the SQLite database if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS processed_sessions (
        id INTEGER PRIMARY KEY,
        log_file TEXT UNIQUE,
        client_id TEXT,
        processed_time TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS processed_segments (
        id INTEGER PRIMARY KEY,
        log_file TEXT,
        client_id TEXT,
        segment_index INTEGER,
        start_time REAL,
        end_time REAL,
        processed_time TIMESTAMP,
        trajectory_id INTEGER,
        UNIQUE(log_file, segment_index)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS config (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    ''')
    
    # Initialize next_id if not exists
    cursor.execute("SELECT value FROM config WHERE key = 'next_id'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO config (key, value) VALUES ('next_id', '1')")
    
    conn.commit()
    conn.close()


def is_session_complete(log_file):
    """Check if a session is complete (has an EOS marker)."""
    try:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("is_eos", False):
                        return True
                except json.JSONDecodeError:
                    continue
        return False
    except Exception as e:
        logger.error(f"Error checking if session {log_file} is complete: {e}")
        return False


def is_session_valid(log_file):
    """
    Check if a session is valid (has more than just an EOS entry).
    Returns True if the log file has at least one non-EOS entry.
    """
    try:
        entry_count = 0
        has_non_eos = False
        
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entry_count += 1
                    if not entry.get("is_eos", False) and not entry.get("is_reset", False):
                        has_non_eos = True
                except json.JSONDecodeError:
                    continue
        
        # Valid if there's at least one entry and at least one non-EOS entry
        return entry_count > 0 and has_non_eos
    
    except Exception as e:
        logger.error(f"Error checking if session {log_file} is valid: {e}")
        return False


def load_trajectory(log_file):
    """Load a trajectory from a log file."""
    trajectory = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    trajectory.append(entry)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line in {log_file}")
                    continue
        return trajectory
    
    except Exception as e:
        logger.error(f"Error loading trajectory from {log_file}: {e}")
        return []

@torch.no_grad()
def process_session_file(log_file, clean_state):
    """Process a session file, splitting into multiple trajectories at reset points."""
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("BEGIN TRANSACTION")  # Explicit transaction
        cursor = conn.cursor()
        
        # Ensure output directory exists
        os.makedirs("generated_videos", exist_ok=True)
        
        # Get session details
        trajectory = load_trajectory(log_file)
        if not trajectory:
            logger.error(f"Empty trajectory for {log_file}, skipping")
            return []
        
        client_id = trajectory[0].get("client_id", "unknown")
        
        # Find all reset points and EOS
        reset_indices = []
        has_eos = False
        
        for i, entry in enumerate(trajectory):
            if entry.get("is_reset", False):
                reset_indices.append(i)
            if entry.get("is_eos", False):
                has_eos = True
        
        # If no resets and no EOS, this is incomplete - skip
        if not reset_indices and not has_eos:
            logger.warning(f"Session {log_file} has no resets and no EOS, may be incomplete")
            return []
        
        # Split trajectory at reset points
        sub_trajectories = []
        start_idx = 0
        
        # Add all segments between resets
        for reset_idx in reset_indices:
            if reset_idx > start_idx:  # Only add non-empty segments
                sub_trajectories.append(trajectory[start_idx:reset_idx])
            start_idx = reset_idx + 1  # Start new segment after the reset
        
        # Add the final segment if it's not empty
        if start_idx < len(trajectory):
            sub_trajectories.append(trajectory[start_idx:])
        
        # Process each sub-trajectory
        processed_ids = []
        
        for i, sub_traj in enumerate(sub_trajectories):
            # Skip segments with no interaction data (just control messages)
            if not any(not entry.get("is_reset", False) and not entry.get("is_eos", False) for entry in sub_traj):
                continue
            
            # Get the next ID for this sub-trajectory
            cursor.execute("SELECT value FROM config WHERE key = 'next_id'")
            next_id = int(cursor.fetchone()[0])
            
            # Find timestamps for this segment
            start_time = sub_traj[0]["timestamp"]
            end_time = sub_traj[-1]["timestamp"]
            
            # STEP 1: Generate a video from the original frames
            segment_label = f"segment_{i+1}_of_{len(sub_trajectories)}"
            video_path = os.path.join("generated_videos", f"trajectory_{next_id}_{segment_label}.mp4")
            
            # Generate video from original frames for comparison
            success, frame_count = generate_comparison_video(
                client_id, 
                sub_traj,
                video_path,
                start_time,
                end_time
            )
            
            if not success:
                logger.warning(f"Failed to generate comparison video for segment {i+1}, but continuing with processing")
            
            # STEP 2: Process with Docker for training data generation
            try:
                logger.info(f"Processing segment {i+1}/{len(sub_trajectories)} from {log_file} as trajectory {next_id}")
                
                # Format the trajectory as needed by process_trajectory function
                formatted_trajectory = format_trajectory_for_processing(sub_traj)
                record_num = next_id
                
                # Call the external process_trajectory function
                args = (record_num, formatted_trajectory)
                process_trajectory(args, SCREEN_WIDTH, SCREEN_HEIGHT, clean_state, MEMORY_LIMIT)

                # Prepare training data format
                video_file = f'raw_data/raw_data/videos/record_{record_num}.mp4'
                action_file = f'raw_data/raw_data/actions/record_{record_num}.csv'
                mouse_data = pd.read_csv(action_file)
                mapping_dict = {}
                target_data = []
                # remove the existing tar file if exists
                if os.path.exists(os.path.join(OUTPUT_DIR, f'record_{record_num}.tar')):
                    logger.info(f"Removing existing tar file {os.path.join(OUTPUT_DIR, f'record_{record_num}.tar')}")
                    os.remove(os.path.join(OUTPUT_DIR, f'record_{record_num}.tar'))
                sink = wds.TarWriter(os.path.join(OUTPUT_DIR, f'record_{record_num}.tar'))
                with VideoFileClip(video_file) as video:
                    fps = video.fps
                    assert fps == 15, f"Expected 15 FPS, got {fps}"
                    duration = video.duration
                    down_keys = set([])
                    for image_num in range(int(fps*duration)):
                        action_row = mouse_data.iloc[image_num]
                        x = int(action_row['X'])
                        y = int(action_row['Y'])
                        left_click = True if action_row['Left Click'] == 1 else False
                        right_click = True if action_row['Right Click'] == 1 else False
                        key_events = ast.literal_eval(action_row['Key Events'])
                        for key_state, key in key_events:
                            if key_state == "keydown":
                                down_keys.add(key)
                            elif key_state == "keyup":
                                down_keys.remove(key)
                            else:
                                raise ValueError(f"Unknown key event type: {key_state}")
                        mapping_dict[(record_num, image_num)] = (x, y, left_click, right_click, list(down_keys))
                        target_data.append((record_num, image_num))
                        frame = video.get_frame(image_num / fps)

                        # Normalize to [-1, 1]
                        image_array = (frame / 127.5 - 1.0).astype(np.float32)

                        # Convert to torch tensor
                        images_tensor = torch.tensor(image_array).unsqueeze(0)
                        images_tensor = rearrange(images_tensor, 'b h w c -> b c h w')

                        # Move to device for inference
                        images_tensor = images_tensor.to(device)

                        # Encode images
                        posterior = autoencoder.encode(images_tensor)
                        latents = posterior.sample()  # Sample from the posterior

                        # Move back to CPU for saving
                        latents = latents.cpu()

                        # Save each latent to the tar file
                        latent = latents[0]
                        keys = [str(image_num)]
                        key = keys[0]
                        
                        # Convert latent to bytes
                        latent_bytes = io.BytesIO()
                        np.save(latent_bytes, latent.numpy())
                        latent_bytes.seek(0)

                        # Write to tar
                        sample = {
                            "__key__": key,
                            "npy": latent_bytes.getvalue(),
                        }
                        sink.write(sample)
                        debug = True
                        # Debug first batch if requested
                        if debug:
                            debug_dir = os.path.join(OUTPUT_DIR, 'debug')
                            os.makedirs(debug_dir, exist_ok=True)

                            # Decode latents back to images
                            reconstructions = autoencoder.decode(latents.to(device))

                            # Save original and reconstructed images side by side
                            for idx, (orig, recon) in enumerate(zip(images_tensor, reconstructions)):
                                # Convert to numpy and move to CPU
                                orig = orig.cpu().numpy()
                                recon = recon.cpu().numpy()

                                # Denormalize from [-1,1] to [0,255]
                                orig = (orig + 1.0) * 127.5
                                recon = (recon + 1.0) * 127.5

                                # Clip values to valid range
                                orig = np.clip(orig, 0, 255).astype(np.uint8)
                                recon = np.clip(recon, 0, 255).astype(np.uint8)

                                # Rearrange from CHW to HWC
                                orig = np.transpose(orig, (1,2,0))
                                recon = np.transpose(recon, (1,2,0))

                                # Create side-by-side comparison
                                comparison = np.concatenate([orig, recon], axis=1)

                                # Save comparison image
                                Image.fromarray(comparison).save(
                                    os.path.join(debug_dir, f'debug_{video_file}_{idx}_{keys[idx]}.png')
                                )
                            print(f"\nDebug visualizations saved to {debug_dir}")
                sink.close()
                # merge with existing mapping_dict if exists, otherwise create new one
                if os.path.exists(os.path.join(OUTPUT_DIR, 'image_action_mapping_with_key_states.pkl')):
                    with open(os.path.join(OUTPUT_DIR, 'image_action_mapping_with_key_states.pkl'), 'rb') as f:
                        existing_mapping_dict = pickle.load(f)
                    for key, value in existing_mapping_dict.items():
                        if key not in mapping_dict:
                            mapping_dict[key] = value
                # save the mapping_dict in an atomic way
                temp_path = os.path.join(OUTPUT_DIR, 'image_action_mapping_with_key_states.pkl.temp')
                with open(temp_path, 'wb') as f:
                    pickle.dump(mapping_dict, f)
                os.rename(temp_path, os.path.join(OUTPUT_DIR, 'image_action_mapping_with_key_states.pkl'))

                # merge with existing target_data if exists, otherwise create new one
                target_data = pd.DataFrame(target_data, columns=['record_num', 'image_num'])
                if os.path.exists(os.path.join(OUTPUT_DIR, 'train_dataset.target_frames.csv')):
                    existing_target_data = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_dataset.target_frames.csv'))
                    target_data = pd.concat([existing_target_data, target_data])
                # deduplicate
                target_data = target_data.drop_duplicates()
                # save the target_data in an atomic way
                temp_path = os.path.join(OUTPUT_DIR, 'train_dataset.target_frames.csv.temp')
                target_data.to_csv(temp_path, index=False)
                os.rename(temp_path, os.path.join(OUTPUT_DIR, 'train_dataset.target_frames.csv'))


                # Mark this segment as processed
                cursor.execute(
                    """INSERT INTO processed_segments 
                       (log_file, client_id, segment_index, start_time, end_time, 
                        processed_time, trajectory_id) 
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (log_file, client_id, i, start_time, end_time, 
                     datetime.now().isoformat(), next_id)
                )
                
                # Increment the next ID
                cursor.execute("UPDATE config SET value = ? WHERE key = 'next_id'", (str(next_id + 1),))
                conn.commit()
                
                processed_ids.append(next_id)
                logger.info(f"Successfully processed segment {i+1}/{len(sub_trajectories)} from {log_file}")
                
            except Exception as e:
                logger.error(f"Failed to process segment {i+1}/{len(sub_trajectories)} from {log_file}: {e}")
                continue
        
        # Mark the entire session as processed only if at least one segment succeeded
        if processed_ids:
            try:
                cursor.execute(
                    "INSERT INTO processed_sessions (log_file, client_id, processed_time) VALUES (?, ?, ?)",
                    (log_file, client_id, datetime.now().isoformat())
                )
                conn.commit()
            except sqlite3.IntegrityError:
                # This can happen if we're re-processing a file that had some segments fail
                pass
        
        # Commit only at the end if everything succeeds
        conn.commit()
        return processed_ids
    except Exception as e:
        logger.error(f"Error processing session {log_file}: {e}")
        if conn:
            conn.rollback()  # Roll back on error
        return []
    finally:
        if conn:
            conn.close()  # Always close connection


def format_trajectory_for_processing(trajectory):
    """
    Format the trajectory in the structure expected by process_trajectory function.
    
    The exact format will depend on what your process_trajectory function expects.
    This is a placeholder - modify based on the actual requirements.
    """
    formatted_events = []
    
    for entry in trajectory:
        # Skip control messages
        if entry.get("is_reset") or entry.get("is_eos"):
            continue
            
        # Extract input data
        inputs = entry.get("inputs", {})
        key_events = []
        for key in inputs.get("keys_down", []):
            key_events.append(("keydown", key))
        for key in inputs.get("keys_up", []):
            key_events.append(("keyup", key))
        event = {
            "pos": (inputs.get("x"), inputs.get("y")),
            "left_click": inputs.get("is_left_click", False),
            "right_click": inputs.get("is_right_click", False),
            "key_events": key_events,
        }
        
        formatted_events.append(event)
    
    return formatted_events


def generate_comparison_video(client_id, trajectory, output_file, start_time, end_time):
    """
    Generate a video from the original frames for comparison purposes.
    
    Args:
        client_id: Client ID for frame lookup
        trajectory: List of interaction log entries for this segment
        output_file: Path to save the output video
        start_time: Start timestamp for this segment
        end_time: End timestamp for this segment
        
    Returns:
        (bool, int): (success status, frame count)
    """
    try:
        # Get frame files for this client
        frame_dir = os.path.join(FRAMES_DIR, f"frames_{client_id}")
        if not os.path.exists(frame_dir):
            logger.warning(f"No frame directory found for client {client_id}")
            return False, 0
        
        all_frames = glob.glob(os.path.join(frame_dir, "*.png"))
        # Sort frames by timestamp in filename
        all_frames.sort(key=lambda x: float(os.path.basename(x).split('.png')[0]))
        
        if not all_frames:
            logger.error(f"No frames found for client {client_id}")
            return False, 0
        
        # Filter frames to the time range of this segment
        # Frame filenames are timestamps, so we can use them for filtering
        segment_frames = [
            f for f in all_frames 
            if start_time <= float(os.path.basename(f).split('.png')[0]) <= end_time
        ]
        
        if not segment_frames:
            logger.error(f"No frames found in time range for segment {start_time}-{end_time}")
            return False, 0
            
        # Read the first frame to get dimensions
        first_frame = cv2.imread(segment_frames[0])
        if first_frame is None:
            logger.error(f"Could not read first frame {segment_frames[0]}")
            return False, 0
            
        height, width, channels = first_frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_file, fourcc, 10.0, (width, height))
        
        # Process each frame
        for frame_file in segment_frames:
            frame = cv2.imread(frame_file)
            if frame is not None:
                video.write(frame)
        
        # Release the video writer
        video.release()
        
        logger.info(f"Created comparison video {output_file} with {len(segment_frames)} frames")
        return True, len(segment_frames)
        
    except Exception as e:
        logger.error(f"Error generating comparison video: {e}")
        return False, 0


def main():
    """Main function to run the data processing pipeline."""
    # Initialize database
    initialize_database()
    
    # Initialize clean Docker state once
    logger.info("Initializing clean container state...")
    clean_state = initialize_clean_state()
    logger.info(f"Clean state initialized: {clean_state}")
    
    # Find all log files
    log_files = glob.glob(os.path.join(FRAMES_DIR, "session_*.jsonl"))
    logger.info(f"Found {len(log_files)} log files")
    
    # Filter for complete sessions
    complete_sessions = [f for f in log_files if is_session_complete(f)]
    logger.info(f"Found {len(complete_sessions)} complete sessions")
    
    # Filter for sessions not yet processed
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT log_file FROM processed_sessions")
    processed_files = set(row[0] for row in cursor.fetchall())
    conn.close()
    
    new_sessions = [f for f in complete_sessions if f not in processed_files]
    logger.info(f"Found {len(new_sessions)} new sessions to process")
    
    # Filter for valid sessions
    valid_sessions = [f for f in new_sessions if is_session_valid(f)]
    logger.info(f"Found {len(valid_sessions)} valid new sessions to process")
    
    # Process each valid session
    total_trajectories = 0
    for log_file in valid_sessions:
        logger.info(f"Processing session file: {log_file}")
        processed_ids = process_session_file(log_file, clean_state)
        total_trajectories += len(processed_ids)
    
    # Get next ID for reporting
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM config WHERE key = 'next_id'")
    next_id = int(cursor.fetchone()[0])
    conn.close()
    
    logger.info(f"Processing complete. Generated {total_trajectories} trajectories.")
    logger.info(f"Next ID will be {next_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
