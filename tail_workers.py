#!/usr/bin/env python3
"""
Script to tail all worker log files simultaneously.
Usage: python tail_workers.py [--num-gpus N]
"""

import argparse
import os
import time
import sys
from typing import Dict

def tail_all_workers(num_gpus: int):
    """Tail all worker log files simultaneously"""
    print(f"Tailing logs for {num_gpus} GPU workers...")
    print("=" * 60)
    
    # Keep track of file positions
    log_positions: Dict[int, int] = {}
    for i in range(num_gpus):
        log_positions[i] = 0
    
    try:
        while True:
            has_new_output = False
            
            for i in range(num_gpus):
                log_file = f"worker_gpu_{i}.log"
                
                try:
                    if os.path.exists(log_file):
                        with open(log_file, 'r') as f:
                            f.seek(log_positions[i])
                            new_lines = f.readlines()
                            
                            if new_lines:
                                has_new_output = True
                                for line in new_lines:
                                    timestamp = time.strftime("%H:%M:%S")
                                    print(f"[{timestamp}] [GPU {i}] {line.rstrip()}")
                                
                            log_positions[i] = f.tell()
                    else:
                        # File doesn't exist yet, check if we should show a message
                        if log_positions[i] == 0:
                            print(f"[INFO] Waiting for {log_file} to be created...")
                            log_positions[i] = -1  # Mark as checked
                            
                except Exception as e:
                    print(f"[ERROR] Error reading {log_file}: {e}")
            
            # Only sleep if there was no new output to keep it responsive
            if not has_new_output:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\nStopping log monitoring...")

def main():
    parser = argparse.ArgumentParser(description="Tail all worker log files")
    parser.add_argument("--num-gpus", type=int, default=2, 
                       help="Number of GPU workers to monitor (default: 2)")
    
    args = parser.parse_args()
    
    if args.num_gpus < 1:
        print("Error: Number of GPUs must be at least 1")
        sys.exit(1)
    
    tail_all_workers(args.num_gpus)

if __name__ == "__main__":
    main() 