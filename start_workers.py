#!/usr/bin/env python3
"""
Script to start multiple GPU workers for the neural OS demo.
Usage: python start_workers.py --num-gpus 4
"""

import argparse
import subprocess
import time
import sys
import signal
import os
from typing import List

class WorkerManager:
    def __init__(self, num_gpus: int, dispatcher_url: str = "http://localhost:8000"):
        self.num_gpus = num_gpus
        self.dispatcher_url = dispatcher_url
        self.processes: List[subprocess.Popen] = []
        
    def start_workers(self):
        """Start all worker processes"""
        print(f"Starting {self.num_gpus} GPU workers...")
        
        for gpu_id in range(self.num_gpus):
            try:
                port = 8001 + gpu_id
                print(f"Starting worker for GPU {gpu_id} on port {port}...")
                
                # Start worker process
                cmd = [
                    sys.executable, "worker.py",
                    "--gpu-id", str(gpu_id),
                    "--dispatcher-url", self.dispatcher_url
                ]
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                self.processes.append(process)
                print(f"✓ Started worker {gpu_id} (PID: {process.pid})")
                
                # Small delay between starts
                time.sleep(1)
                
            except Exception as e:
                print(f"✗ Failed to start worker for GPU {gpu_id}: {e}")
                self.cleanup()
                return False
        
        print(f"\n✓ All {self.num_gpus} workers started successfully!")
        print("Workers are running on ports:", [8001 + i for i in range(self.num_gpus)])
        return True
    
    def monitor_workers(self):
        """Monitor worker processes and print their output"""
        print("\nMonitoring workers (Ctrl+C to stop)...")
        print("-" * 50)
        
        try:
            while True:
                # Check if any process has died
                for i, process in enumerate(self.processes):
                    if process.poll() is not None:
                        print(f"⚠️  Worker {i} (PID: {process.pid}) has died!")
                        # Optionally restart it
                        
                # Print output from processes
                for i, process in enumerate(self.processes):
                    if process.stdout and process.stdout.readable():
                        try:
                            line = process.stdout.readline()
                            if line:
                                print(f"[GPU {i}] {line.strip()}")
                        except:
                            pass
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\nReceived interrupt signal, shutting down workers...")
            self.cleanup()
    
    def cleanup(self):
        """Clean up all worker processes"""
        print("Stopping all workers...")
        
        for i, process in enumerate(self.processes):
            if process.poll() is None:  # Process is still running
                print(f"Stopping worker {i} (PID: {process.pid})...")
                try:
                    process.terminate()
                    # Wait for graceful shutdown
                    process.wait(timeout=5)
                    print(f"✓ Worker {i} stopped gracefully")
                except subprocess.TimeoutExpired:
                    print(f"⚠️  Force killing worker {i}...")
                    process.kill()
                    process.wait()
                except Exception as e:
                    print(f"Error stopping worker {i}: {e}")
        
        print("✓ All workers stopped")

def main():
    parser = argparse.ArgumentParser(description="Start multiple GPU workers")
    parser.add_argument("--num-gpus", type=int, required=True, 
                       help="Number of GPU workers to start")
    parser.add_argument("--dispatcher-url", type=str, default="http://localhost:8000",
                       help="URL of the dispatcher service")
    parser.add_argument("--no-monitor", action="store_true",
                       help="Start workers but don't monitor them")
    
    args = parser.parse_args()
    
    if args.num_gpus < 1:
        print("Error: Number of GPUs must be at least 1")
        sys.exit(1)
    
    # Check if worker.py exists
    if not os.path.exists("worker.py"):
        print("Error: worker.py not found in current directory")
        sys.exit(1)
    
    manager = WorkerManager(args.num_gpus, args.dispatcher_url)
    
    # Set up signal handlers for clean shutdown
    def signal_handler(sig, frame):
        print(f"\nReceived signal {sig}, shutting down...")
        manager.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start workers
    if not manager.start_workers():
        sys.exit(1)
    
    if not args.no_monitor:
        manager.monitor_workers()
    else:
        print("Workers started. Use 'ps aux | grep worker.py' to check status.")
        print("To stop workers, use: pkill -f 'python.*worker.py'")

if __name__ == "__main__":
    main() 