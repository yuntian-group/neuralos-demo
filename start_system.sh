#!/bin/bash

# Multi-GPU Neural OS Startup Script

# Default values
NUM_GPUS=2
DISPATCHER_PORT=8000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --port)
            DISPATCHER_PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--num-gpus N] [--port PORT]"
            echo "  --num-gpus N    Number of GPU workers to start (default: 2)"
            echo "  --port PORT     Dispatcher port (default: 8000)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down system..."
    
    # Kill dispatcher
    if [[ -n $DISPATCHER_PID ]]; then
        echo "Stopping dispatcher (PID: $DISPATCHER_PID)..."
        kill $DISPATCHER_PID 2>/dev/null
        wait $DISPATCHER_PID 2>/dev/null
    fi
    
    # Kill workers by finding their processes
    echo "Stopping workers..."
    pkill -f "python.*worker.py.*--gpu-id" 2>/dev/null || true
    sleep 2
    # Force kill if any are still running
    pkill -9 -f "python.*worker.py.*--gpu-id" 2>/dev/null || true
    
    echo "✅ System stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo "🚀 Starting Multi-GPU Neural OS System"
echo "========================================"
echo "📊 Number of GPUs: $NUM_GPUS"
echo "🌐 Dispatcher port: $DISPATCHER_PORT" 
echo "💻 Worker ports: $(seq -s', ' 8001 $((8000 + NUM_GPUS)))"
echo ""

# Check if required files exist
if [[ ! -f "dispatcher.py" ]]; then
    echo "❌ Error: dispatcher.py not found"
    exit 1
fi

if [[ ! -f "worker.py" ]]; then
    echo "❌ Error: worker.py not found"
    exit 1
fi

if [[ ! -f "start_workers.py" ]]; then
    echo "❌ Error: start_workers.py not found"
    exit 1
fi

# Start dispatcher
echo "🎯 Starting dispatcher..."
python dispatcher.py --port $DISPATCHER_PORT > dispatcher.log 2>&1 &
DISPATCHER_PID=$!

# Wait a bit for dispatcher to start
sleep 3

# Check if dispatcher started successfully
if ! kill -0 $DISPATCHER_PID 2>/dev/null; then
    echo "❌ Failed to start dispatcher. Check dispatcher.log for errors."
    exit 1
fi

echo "✅ Dispatcher started (PID: $DISPATCHER_PID)"

# Start workers
echo "🔧 Starting $NUM_GPUS GPU workers..."
python start_workers.py --num-gpus $NUM_GPUS --no-monitor > workers.log 2>&1
WORKER_START_EXIT_CODE=$?

# Wait a bit for workers to register
sleep 3

# Check if workers started successfully by checking the exit code and log
if [ $WORKER_START_EXIT_CODE -ne 0 ]; then
    echo "❌ Failed to start workers. Check workers.log for errors."
    cleanup
    exit 1
fi

# Check if workers are actually running by looking for their processes
RUNNING_WORKERS=$(ps aux | grep -c "python.*worker.py.*--gpu-id" || echo "0")
if [ "$RUNNING_WORKERS" -lt "$NUM_GPUS" ]; then
    echo "❌ Not all workers are running. Expected $NUM_GPUS, found $RUNNING_WORKERS. Check workers.log for errors."
    cleanup
    exit 1
fi

echo "✅ Workers started successfully ($RUNNING_WORKERS workers running)"
echo ""
echo "🎉 System is ready!"
echo "================================"
echo "🌍 Web interface: http://localhost:$DISPATCHER_PORT"
echo "📊 Dispatcher health: http://localhost:$DISPATCHER_PORT"
echo "🔧 Worker health checks:"
for ((i=0; i<NUM_GPUS; i++)); do
    echo "   GPU $i: http://localhost:$((8001 + i))/health"
done
echo ""
echo "📋 Log files:"
echo "   Dispatcher: dispatcher.log"
echo "   Workers summary: workers.log"
for ((i=0; i<NUM_GPUS; i++)); do
    echo "   GPU $i worker: worker_gpu_$i.log"
done
echo ""
echo "Press Ctrl+C to stop the system"
echo "================================"

# Keep the script running and wait for interrupt
while true; do
    # Check if processes are still running
    if ! kill -0 $DISPATCHER_PID 2>/dev/null; then
        echo "⚠️  Dispatcher process died unexpectedly"
        cleanup
        exit 1
    fi
    
    # Check if workers are still running
    CURRENT_WORKERS=$(ps aux | grep -c "python.*worker.py.*--gpu-id" || echo "0")
    if [ "$CURRENT_WORKERS" -lt "$NUM_GPUS" ]; then
        echo "⚠️  Some workers died unexpectedly. Expected $NUM_GPUS, found $CURRENT_WORKERS"
        cleanup
        exit 1
    fi
    
    sleep 5
done 