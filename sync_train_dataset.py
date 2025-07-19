#!/usr/bin/env python3
import os
import sys
import time
import logging
import paramiko
import hashlib
import tempfile
from datetime import datetime
import sqlite3
import re
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_transfer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
REMOTE_HOST = "neural-os.com"
REMOTE_USER = "root"  # Replace with your actual username
REMOTE_KEY_PATH = "~/.ssh/id_rsa"  # Replace with path to your SSH key
REMOTE_DATA_DIR = "/root/neuralos-demo-datagen/train_dataset_encoded_online"  # Replace with actual path
LOCAL_DATA_DIR = "./train_dataset_encoded_online"  # Local destination
DB_FILE = "transfer_state.db"
POLL_INTERVAL = 300  # Check for new files every 5 minutes
STABILITY_WAIT = 5   # Reduced from 30 seconds to 5 seconds
MAX_PARALLEL_TRANSFERS = 3  # Number of parallel file transfers

# Ensure local directories exist
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

# Thread lock for database operations
db_lock = threading.Lock()


def initialize_database():
    """Create and initialize the SQLite database to track transferred files."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transferred_files (
        id INTEGER PRIMARY KEY,
        filename TEXT UNIQUE,
        remote_size INTEGER,
        remote_mtime REAL,
        transfer_time TIMESTAMP,
        checksum TEXT
    )
    ''')
    
    # Table for tracking last successful CSV/PKL transfer
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transfer_state (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    ''')
    
    conn.commit()
    conn.close()


def is_file_transferred(filename, remote_size, remote_mtime):
    """Check if a file has already been transferred with the same size and mtime."""
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT 1 FROM transferred_files WHERE filename = ? AND remote_size = ? AND remote_mtime = ?",
            (filename, remote_size, remote_mtime)
        )
        result = cursor.fetchone() is not None
        
        conn.close()
        return result


def mark_file_transferred(filename, remote_size, remote_mtime, checksum):
    """Mark a file as successfully transferred."""
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute(
            """INSERT OR REPLACE INTO transferred_files 
               (filename, remote_size, remote_mtime, transfer_time, checksum) 
               VALUES (?, ?, ?, ?, ?)""",
            (filename, remote_size, remote_mtime, datetime.now().isoformat(), checksum)
        )
        
        conn.commit()
        conn.close()


def update_transfer_state(key, value):
    """Update the transfer state for a key."""
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT OR REPLACE INTO transfer_state (key, value) VALUES (?, ?)",
            (key, value)
        )
        
        conn.commit()
        conn.close()


def get_transfer_state(key):
    """Get the transfer state for a key."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT value FROM transfer_state WHERE key = ?", (key,))
    result = cursor.fetchone()
    
    conn.close()
    return result[0] if result else None


def calculate_checksum(file_path):
    """Calculate MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)
    return md5.hexdigest()


def create_ssh_client():
    """Create and return an SSH client connected to the remote server."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    # Expand the key path
    key_path = os.path.expanduser(REMOTE_KEY_PATH)
    
    try:
        key = paramiko.RSAKey.from_private_key_file(key_path)
        client.connect(
            hostname=REMOTE_HOST,
            username=REMOTE_USER,
            pkey=key
        )
        logger.info(f"Successfully connected to {REMOTE_USER}@{REMOTE_HOST}")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to {REMOTE_HOST}: {str(e)}")
        raise


def safe_transfer_file(sftp, remote_path, local_path):
    """
    Transfer a file safely using a temporary file and rename.
    Returns the checksum of the transferred file.
    """
    # Create a temporary file for download
    temp_file = local_path + ".tmp"
    
    try:
        # Transfer to temporary file
        sftp.get(remote_path, temp_file)
        
        # Calculate checksum
        checksum = calculate_checksum(temp_file)
        
        # Rename to final destination
        os.rename(temp_file, local_path)
        logger.info(f"Successfully transferred {remote_path} to {local_path}")
        
        return checksum
    except Exception as e:
        logger.error(f"Error transferring {remote_path}: {str(e)}")
        # Clean up temp file if it exists
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise


def is_file_stable(sftp, remote_path, wait_time=STABILITY_WAIT):
    """
    Check if a file is stable (not being written to) by comparing its size
    before and after a short wait period.
    """
    try:
        # Get initial stats
        initial_stat = sftp.stat(remote_path)
        initial_size = initial_stat.st_size
        
        # Wait a bit
        time.sleep(wait_time)
        
        # Get updated stats
        updated_stat = sftp.stat(remote_path)
        updated_size = updated_stat.st_size
        
        # File is stable if size hasn't changed
        is_stable = initial_size == updated_size
        
        if not is_stable:
            logger.info(f"File {remote_path} is still being written to (size changed from {initial_size} to {updated_size})")
        
        return is_stable, updated_stat
    except Exception as e:
        logger.error(f"Error checking if {remote_path} is stable: {str(e)}")
        return False, None


def transfer_tar_files(sftp):
    """Transfer all record_*.tar files that haven't been transferred yet."""
    transferred_count = 0
    
    try:
        # List all tar files
        tar_pattern = re.compile(r'record_.*\.tar$')
        remote_files = sftp.listdir(REMOTE_DATA_DIR)
        tar_files = [f for f in remote_files if tar_pattern.match(f)]
        
        logger.info(f"Found {len(tar_files)} TAR files on remote server")
        
        for tar_file in tar_files:
            remote_path = os.path.join(REMOTE_DATA_DIR, tar_file)
            local_path = os.path.join(LOCAL_DATA_DIR, tar_file)
            
            # Get file stats
            try:
                stat = sftp.stat(remote_path)
            except FileNotFoundError:
                logger.warning(f"File {remote_path} disappeared, skipping")
                continue
                
            # Skip if already transferred with same size and mtime
            if is_file_transferred(tar_file, stat.st_size, stat.st_mtime):
                logger.debug(f"Skipping already transferred file: {tar_file}")
                continue
                
            # Check if file is stable (not being written to)
            is_stable, updated_stat = is_file_stable(sftp, remote_path)
            if not is_stable:
                logger.info(f"Skipping unstable file: {tar_file}")
                continue
                
            # Transfer the file
            try:
                checksum = safe_transfer_file(sftp, remote_path, local_path)
                mark_file_transferred(tar_file, updated_stat.st_size, updated_stat.st_mtime, checksum)
                transferred_count += 1
            except Exception as e:
                logger.error(f"Failed to transfer {tar_file}: {str(e)}")
                continue
                
        logger.info(f"Transferred {transferred_count} new TAR files")
        return transferred_count
    except Exception as e:
        logger.error(f"Error in transfer_tar_files: {str(e)}")
        return 0


def transfer_pkl_file(sftp):
    """Transfer the PKL file if it hasn't been transferred yet or has changed."""
    pkl_file = "image_action_mapping_with_key_states.pkl"
    remote_path = os.path.join(REMOTE_DATA_DIR, pkl_file)
    local_path = os.path.join(LOCAL_DATA_DIR, pkl_file)
    
    try:
        # Check if file exists
        try:
            stat = sftp.stat(remote_path)
        except FileNotFoundError:
            logger.warning(f"PKL file {remote_path} not found")
            return False
            
        # Skip if already transferred with same size and mtime
        if is_file_transferred(pkl_file, stat.st_size, stat.st_mtime):
            logger.debug(f"Skipping already transferred PKL file (unchanged)")
            return True
            
        # Check if file is stable
        is_stable, updated_stat = is_file_stable(sftp, remote_path)
        if not is_stable:
            logger.info(f"PKL file is still being written to, skipping")
            return False
            
        # Transfer the file
        checksum = safe_transfer_file(sftp, remote_path, local_path)
        mark_file_transferred(pkl_file, updated_stat.st_size, updated_stat.st_mtime, checksum)
        
        # Update state
        update_transfer_state("last_pkl_transfer", datetime.now().isoformat())
        
        logger.info(f"Successfully transferred PKL file")
        return True
    except Exception as e:
        logger.error(f"Error transferring PKL file: {str(e)}")
        return False


def transfer_csv_file(sftp):
    """Transfer the CSV file if it hasn't been transferred yet or has changed."""
    csv_file = "train_dataset.target_frames.csv"
    remote_path = os.path.join(REMOTE_DATA_DIR, csv_file)
    local_path = os.path.join(LOCAL_DATA_DIR, csv_file)
    
    try:
        # Check if file exists
        try:
            stat = sftp.stat(remote_path)
        except FileNotFoundError:
            logger.warning(f"CSV file {remote_path} not found")
            return False
            
        # Skip if already transferred with same size and mtime
        if is_file_transferred(csv_file, stat.st_size, stat.st_mtime):
            logger.debug(f"Skipping already transferred CSV file (unchanged)")
            return True
            
        # Check if file is stable
        is_stable, updated_stat = is_file_stable(sftp, remote_path)
        if not is_stable:
            logger.info(f"CSV file is still being written to, skipping")
            return False
            
        # Transfer the file
        checksum = safe_transfer_file(sftp, remote_path, local_path)
        mark_file_transferred(csv_file, updated_stat.st_size, updated_stat.st_mtime, checksum)
        
        # Update state
        update_transfer_state("last_csv_transfer", datetime.now().isoformat())
        
        logger.info(f"Successfully transferred CSV file")
        return True
    except Exception as e:
        logger.error(f"Error transferring CSV file: {str(e)}")
        return False


def transfer_padding_file(sftp):
    """Transfer the padding.npy file if it hasn't been transferred yet or has changed."""
    padding_file = "padding.npy"
    remote_path = os.path.join(REMOTE_DATA_DIR, padding_file)
    local_path = os.path.join(LOCAL_DATA_DIR, padding_file)
    
    try:
        # Check if file exists
        try:
            stat = sftp.stat(remote_path)
        except FileNotFoundError:
            logger.warning(f"Padding file {remote_path} not found")
            return False
            
        # Skip if already transferred with same size and mtime
        if is_file_transferred(padding_file, stat.st_size, stat.st_mtime):
            logger.debug(f"Skipping already transferred padding file (unchanged)")
            return True
            
        # Check if file is stable
        is_stable, updated_stat = is_file_stable(sftp, remote_path)
        if not is_stable:
            logger.info(f"Padding file is still being written to, skipping")
            return False
            
        # Transfer the file
        checksum = safe_transfer_file(sftp, remote_path, local_path)
        mark_file_transferred(padding_file, updated_stat.st_size, updated_stat.st_mtime, checksum)
        
        # Update state
        update_transfer_state("last_padding_transfer", datetime.now().isoformat())
        
        logger.info(f"Successfully transferred padding.npy file")
        return True
    except Exception as e:
        logger.error(f"Error transferring padding file: {str(e)}")
        return False


def run_transfer_cycle():
    """Run a complete transfer cycle with time-based consistency."""
    client = None
    try:
        # Connect to the remote server
        client = create_ssh_client()
        sftp = client.open_sftp()
        
        # Step 0: Transfer CSV file FIRST (get current state before snapshot)
        csv_file = "train_dataset.target_frames.csv"
        csv_file_path = os.path.join(REMOTE_DATA_DIR, csv_file)
        csv_success = False
        
        try:
            csv_stat = sftp.stat(csv_file_path)
            # Only transfer if needed
            if not is_file_transferred(csv_file, csv_stat.st_size, csv_stat.st_mtime):
                is_stable, updated_csv_stat = is_file_stable(sftp, csv_file_path)
                if is_stable:
                    local_path = os.path.join(LOCAL_DATA_DIR, csv_file)
                    checksum = safe_transfer_file(sftp, csv_file_path, local_path)
                    mark_file_transferred(csv_file, updated_csv_stat.st_size, updated_csv_stat.st_mtime, checksum)
                    update_transfer_state("last_csv_transfer", datetime.now().isoformat())
                    logger.info("Successfully transferred CSV file (before snapshot)")
                    csv_success = True
                else:
                    logger.warning("CSV file is still being written, skipping")
                    csv_success = False
            else:
                logger.debug("CSV file unchanged, skipping")
                csv_success = True
        except FileNotFoundError:
            logger.warning("CSV file not found on remote server")
            csv_success = False
        except Exception as e:
            logger.error(f"Error checking CSV file: {str(e)}")
            csv_success = False
        
        # Step 1: NOW take snapshot of all files (after CSV transfer)
        # This ensures TAR files in snapshot include everything referenced by the CSV
        logger.info("Taking snapshot of remote directory state (after CSV transfer)")
        remote_files = {}
        for filename in sftp.listdir(REMOTE_DATA_DIR):
            try:
                file_path = os.path.join(REMOTE_DATA_DIR, filename)
                stat = sftp.stat(file_path)
                remote_files[filename] = {
                    'size': stat.st_size,
                    'mtime': stat.st_mtime,
                    'path': file_path
                }
            except Exception as e:
                logger.warning(f"Could not stat file {filename}: {str(e)}")
                
        logger.info(f"Found {len(remote_files)} files in remote directory snapshot")
        
        # Step 2: Transfer padding.npy file if needed
        if "padding.npy" in remote_files:
            file_info = remote_files["padding.npy"]
            if not is_file_transferred("padding.npy", file_info['size'], file_info['mtime']):
                # Check stability
                is_stable, updated_stat = is_file_stable(sftp, file_info['path'])
                if is_stable:
                    local_path = os.path.join(LOCAL_DATA_DIR, "padding.npy")
                    checksum = safe_transfer_file(sftp, file_info['path'], local_path)
                    mark_file_transferred("padding.npy", updated_stat.st_size, updated_stat.st_mtime, checksum)
                    logger.info("Successfully transferred padding.npy file")
                else:
                    logger.warning("Padding file is still being written, skipping")
        else:
            logger.warning("padding.npy not found in remote directory")

        # Step 3: Transfer TAR files from the snapshot 
        # (Snapshot taken AFTER CSV, so includes all TAR files referenced by CSV)
        tar_pattern = re.compile(r'record_.*\.tar$')
        tar_files = {name: info for name, info in remote_files.items() if tar_pattern.match(name)}
        logger.info(f"Found {len(tar_files)} TAR files in snapshot")
        
        # Sort tar files numerically by the record number
        def extract_record_number(filename):
            match = re.search(r'record_(\d+)\.tar$', filename)
            return int(match.group(1)) if match else float('inf')
        
        sorted_tar_files = sorted(tar_files.items(), key=lambda x: extract_record_number(x[0]))
        
        # Filter files that need to be transferred (quick check only)
        files_to_check = []
        for tar_file, file_info in sorted_tar_files:
            # Skip if already transferred with same size and mtime
            if is_file_transferred(tar_file, file_info['size'], file_info['mtime']):
                logger.debug(f"Skipping already transferred file: {tar_file}")
                continue
                
            files_to_check.append((tar_file, file_info))
        
        logger.info(f"Found {len(files_to_check)} TAR files to check and transfer")
        
        # Transfer files in parallel (including stability check)
        tar_count = 0
        if files_to_check:
            def transfer_single_file(args):
                tar_file, file_info = args
                thread_client = None
                try:
                    # Create a new SFTP connection for this thread
                    thread_client = create_ssh_client()
                    thread_sftp = thread_client.open_sftp()
                    
                    # Check if file is stable (now done in parallel)
                    is_stable, updated_stat = is_file_stable(thread_sftp, file_info['path'])
                    if not is_stable:
                        logger.info(f"Skipping unstable file: {tar_file}")
                        return False
                    
                    local_path = os.path.join(LOCAL_DATA_DIR, tar_file)
                    checksum = safe_transfer_file(thread_sftp, file_info['path'], local_path)
                    mark_file_transferred(tar_file, updated_stat.st_size, updated_stat.st_mtime, checksum)
                    
                    logger.info(f"Successfully transferred {tar_file}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to transfer {tar_file}: {str(e)}")
                    return False
                finally:
                    # Always close the connection if it was created
                    if thread_client is not None:
                        try:
                            thread_client.close()
                        except Exception as e:
                            logger.warning(f"Error closing connection for {tar_file}: {str(e)}")
            
            # Use ThreadPoolExecutor for parallel stability checks and transfers
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_TRANSFERS) as executor:
                results = list(executor.map(transfer_single_file, files_to_check))
                tar_count = sum(results)
                
        logger.info(f"Transferred {tar_count} new TAR files from snapshot")
        
        # Step 4: Transfer PKL file from the snapshot
        pkl_file = "image_action_mapping_with_key_states.pkl"
        if pkl_file in remote_files:
            file_info = remote_files[pkl_file]
            
            # Only transfer if needed
            if not is_file_transferred(pkl_file, file_info['size'], file_info['mtime']):
                is_stable, updated_stat = is_file_stable(sftp, file_info['path'])
                if is_stable:
                    local_path = os.path.join(LOCAL_DATA_DIR, pkl_file)
                    checksum = safe_transfer_file(sftp, file_info['path'], local_path)
                    mark_file_transferred(pkl_file, updated_stat.st_size, updated_stat.st_mtime, checksum)
                    update_transfer_state("last_pkl_transfer", datetime.now().isoformat())
                    logger.info("Successfully transferred PKL file from snapshot")
                    pkl_success = True
                else:
                    logger.warning("PKL file is still being written, skipping")
                    pkl_success = False
            else:
                logger.debug("PKL file unchanged, skipping")
                pkl_success = True
        else:
            logger.warning("PKL file not found in snapshot")
            pkl_success = False
        
        
        
        return tar_count > 0 or pkl_success or csv_success
    except Exception as e:
        logger.error(f"Error in transfer cycle: {str(e)}")
        return False
    finally:
        if client:
            client.close()


def main():
    """Main function for the data transfer script."""
    logger.info("Starting data transfer script")
    
    # Initialize the database
    initialize_database()
    
    try:
        while True:
            logger.info("Starting new transfer cycle")
            
            changes = run_transfer_cycle()
            
            if changes:
                logger.info("Transfer cycle completed with new files transferred")
            else:
                logger.info("Transfer cycle completed with no changes")
                
            logger.info(f"Sleeping for {POLL_INTERVAL} seconds before next check")
            time.sleep(POLL_INTERVAL)
            
    except KeyboardInterrupt:
        logger.info("Script terminated by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        raise


if __name__ == "__main__":
    main()
