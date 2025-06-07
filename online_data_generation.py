#!/usr/bin/env python3
import os
import json
import glob
import time
import sqlite3
import logging
import cv2
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("video_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
DB_FILE = "trajectory_processor.db"
OUTPUT_DIR = "generated_videos"
FRAMES_DIR = "interaction_logs"


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
        video_path TEXT,
        frame_count INTEGER,
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


def get_frame_files(client_id):
    """Get all frame files for a client ID, sorted by timestamp."""
    frame_dir = os.path.join(FRAMES_DIR, f"frames_{client_id}")
    
    if not os.path.exists(frame_dir):
        logger.warning(f"No frame directory found for client {client_id}")
        return []
    
    frames = glob.glob(os.path.join(frame_dir, "*.png"))
    # Sort frames by timestamp in filename
    frames.sort(key=lambda x: float(os.path.basename(x).split('.png')[0]))
    return frames


def process_trajectory(trajectory, output_file):
    """
    Process a trajectory and create a video file.
    
    Args:
        trajectory: List of interaction log entries
        output_file: Path to save the output video
        
    Returns:
        (bool, int): (success status, frame count)
    """
    if not trajectory:
        logger.error("Cannot process empty trajectory")
        return False, 0
    
    try:
        # Extract client_id from the first entry
        client_id = trajectory[0].get("client_id")
        if not client_id:
            logger.error("Trajectory missing client_id")
            return False, 0
        
        # Get all frame files for this client
        frame_files = get_frame_files(client_id)
        if not frame_files:
            logger.error(f"No frames found for client {client_id}")
            return False, 0
            
        # Read the first frame to get dimensions
        first_frame = cv2.imread(frame_files[0])
        if first_frame is None:
            logger.error(f"Could not read first frame {frame_files[0]}")
            return False, 0
            
        height, width, channels = first_frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_file, fourcc, 10.0, (width, height))
        
        # Process each frame
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            if frame is not None:
                video.write(frame)
        
        # Release the video writer
        video.release()
        
        logger.info(f"Successfully created video {output_file} with {len(frame_files)} frames")
        return True, len(frame_files)
        
    except Exception as e:
        logger.error(f"Error processing trajectory: {e}")
        return False, 0


def get_next_id():
    """Get the next available ID from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT value FROM config WHERE key = 'next_id'")
    result = cursor.fetchone()
    next_id = int(result[0])
    
    conn.close()
    return next_id


def increment_next_id():
    """Increment the next ID in the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("UPDATE config SET value = value + 1 WHERE key = 'next_id'")
    conn.commit()
    
    conn.close()


def is_session_processed(log_file):
    """Check if a session has already been processed."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT 1 FROM processed_sessions WHERE log_file = ?", (log_file,))
    result = cursor.fetchone() is not None
    
    conn.close()
    return result


def mark_session_processed(log_file, client_id, video_path, frame_count):
    """Mark a session as processed in the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO processed_sessions (log_file, client_id, processed_time, video_path, frame_count) VALUES (?, ?, ?, ?, ?)",
        (log_file, client_id, datetime.now().isoformat(), video_path, frame_count)
    )
    
    conn.commit()
    conn.close()


def process_session_file(log_file):
    """
    Process a session file, splitting into multiple trajectories at reset points.
    Returns a list of successfully processed trajectory IDs.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
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
        
        # Define output path
        segment_label = f"segment_{i+1}_of_{len(sub_trajectories)}"
        output_file = os.path.join(OUTPUT_DIR, f"trajectory_{next_id:06d}_{segment_label}.mp4")
        
        # Find timestamps for this segment to get corresponding frames
        start_time = sub_traj[0]["timestamp"]
        end_time = sub_traj[-1]["timestamp"]
        
        # Process this sub-trajectory
        success, frame_count = process_trajectory_segment(
            client_id, 
            sub_traj, 
            output_file,
            start_time,
            end_time
        )
        
        if success:
            # Mark this segment as processed
            cursor.execute(
                """INSERT INTO processed_segments 
                   (log_file, client_id, segment_index, start_time, end_time, 
                    processed_time, video_path, frame_count, trajectory_id) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (log_file, client_id, i, start_time, end_time, 
                 datetime.now().isoformat(), output_file, frame_count, next_id)
            )
            
            # Increment the next ID
            cursor.execute("UPDATE config SET value = ? WHERE key = 'next_id'", (str(next_id + 1),))
            conn.commit()
            
            processed_ids.append(next_id)
            logger.info(f"Successfully processed segment {i+1}/{len(sub_trajectories)} from {log_file}")
        else:
            logger.error(f"Failed to process segment {i+1}/{len(sub_trajectories)} from {log_file}")
    
    # Mark the entire session as processed
    cursor.execute(
        "INSERT INTO processed_sessions (log_file, client_id, processed_time) VALUES (?, ?, ?)",
        (log_file, client_id, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    
    return processed_ids


def process_trajectory_segment(client_id, trajectory, output_file, start_time, end_time):
    """
    Process a segment of a trajectory between timestamps and create a video.
    
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
        all_frames = get_frame_files(client_id)
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
        
        logger.info(f"Created video {output_file} with {len(segment_frames)} frames")
        return True, len(segment_frames)
        
    except Exception as e:
        logger.error(f"Error processing trajectory segment: {e}")
        return False, 0


def main():
    """Main function to run the data processing pipeline."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize database
    initialize_database()
    
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
        processed_ids = process_session_file(log_file)
        total_trajectories += len(processed_ids)
    
    # Get next ID for reporting
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM config WHERE key = 'next_id'")
    next_id = int(cursor.fetchone()[0])
    conn.close()
    
    logger.info(f"Processing complete. Generated {total_trajectories} trajectory videos.")
    logger.info(f"Next ID will be {next_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)