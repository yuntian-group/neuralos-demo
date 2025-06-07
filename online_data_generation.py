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

# Import the existing functions
from latent_diffusion.ldm.data.data_collection import process_trajectory, initialize_clean_state

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
SCREEN_WIDTH = 512
SCREEN_HEIGHT = 384
MEMORY_LIMIT = "2g"

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


def process_session_file(log_file, clean_state):
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
        conn.close()
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
        conn.close()
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
        
        # Process this sub-trajectory using the external function
        try:
            logger.info(f"Processing segment {i+1}/{len(sub_trajectories)} from {log_file} as trajectory {next_id}")
            
            # Format the trajectory as needed by process_trajectory function
            formatted_trajectory = format_trajectory_for_processing(sub_traj)
            
            # Call the external process_trajectory function
            args = (next_id, formatted_trajectory)
            process_trajectory(args, SCREEN_WIDTH, SCREEN_HEIGHT, clean_state, MEMORY_LIMIT)
            
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
    
    conn.close()
    return processed_ids


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