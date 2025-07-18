#!/bin/bash

# Multi-GPU Neural OS Startup Script

# Function to detect number of GPUs automatically
detect_gpu_count() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        # Use nvidia-smi to count GPUs
        local gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
        if [ "$gpu_count" -gt 0 ]; then
            echo "$gpu_count"
            return 0
        fi
    fi
    
    # If nvidia-smi fails, try alternative methods
    if [ -d "/proc/driver/nvidia/gpus" ]; then
        local gpu_count=$(ls -d /proc/driver/nvidia/gpus/*/information 2>/dev/null | wc -l)
        if [ "$gpu_count" -gt 0 ]; then
            echo "$gpu_count"
            return 0
        fi
    fi
    
    # Default fallback
    echo "1"
    return 1
}

# Detect GPU count automatically
DETECTED_GPUS=$(detect_gpu_count)
GPU_DETECTION_SUCCESS=$?

# Default values
NUM_GPUS=$DETECTED_GPUS
DISPATCHER_PORT=7860

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
            echo "  --num-gpus N    Number of GPU workers to start (default: auto-detected)"
            echo "  --port PORT     Dispatcher port (default: 7860)"
            echo ""
            echo "GPU Detection:"
            echo "  Automatically detects available GPUs using nvidia-smi"
            echo "  Currently detected: $DETECTED_GPUS GPU(s)"
            if [ $GPU_DETECTION_SUCCESS -ne 0 ]; then
                echo "  ⚠️  GPU detection failed - using fallback of 1 GPU"
            fi
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
    pkill -f "python.*worker.py.*--worker-address" 2>/dev/null || true
    sleep 2
    # Force kill if any are still running
    pkill -9 -f "python.*worker.py.*--worker-address" 2>/dev/null || true
    
    echo "✅ System stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo "🚀 Starting Multi-GPU Neural OS System"
echo "========================================"
echo "🔍 GPU Detection: $DETECTED_GPUS GPU(s) detected"
if [ $GPU_DETECTION_SUCCESS -ne 0 ]; then
    echo "⚠️  GPU detection failed - using fallback count"
elif command -v nvidia-smi >/dev/null 2>&1; then
    echo "💎 Detected GPUs:"
    nvidia-smi -L 2>/dev/null | sed 's/^/   /'
fi
echo "📊 Number of GPUs: $NUM_GPUS"
echo "🌐 Dispatcher port: $DISPATCHER_PORT" 
echo "💻 Worker ports: $(seq -s', ' 8001 $((8000 + NUM_GPUS)))"
echo "📈 Analytics logging: system_analytics_$(date +%Y%m%d_%H%M%S).log"
echo ""

# Validate that we're not trying to start more workers than GPUs
if [ "$NUM_GPUS" -gt "$DETECTED_GPUS" ]; then
    echo "⚠️  Warning: Trying to start $NUM_GPUS workers but only $DETECTED_GPUS GPU(s) detected"
    echo "   This may cause GPU sharing or errors. Consider using --num-gpus $DETECTED_GPUS"
    echo ""
fi

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
python start_workers.py --num-gpus $NUM_GPUS --dispatcher-url "http://localhost:$DISPATCHER_PORT" --no-monitor > workers.log 2>&1
WORKER_START_EXIT_CODE=$?

# Wait for workers to fully load models and register (60 seconds)
echo "⏳ Waiting 60 seconds for workers to load models and register..."
sleep 60

# Check if workers started successfully by checking the exit code and log
if [ $WORKER_START_EXIT_CODE -ne 0 ]; then
    echo "❌ Failed to start workers. Check workers.log for errors."
    cleanup
    exit 1
fi

# Check if workers are actually running by looking for their processes (updated for new --worker-address format)
RUNNING_WORKERS=$(ps aux | grep -c "python.*worker.py.*--worker-address" || echo "0")
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
echo "   📊 Analytics (human-readable): system_analytics_*.log"
echo "   🖥️  GPU metrics (JSON): gpu_metrics_*.jsonl"
echo "   🔗 Connection events (JSON): connection_events_*.jsonl"
echo "   📝 Queue metrics (JSON): queue_metrics_*.jsonl" 
echo "   🌍 IP statistics (JSON): ip_stats_*.jsonl"
echo "   🎯 Dispatcher: dispatcher.log"
echo "   🔧 Workers summary: workers.log"
for ((i=0; i<NUM_GPUS; i++)); do
    echo "   🖥️  GPU $i worker: worker_gpu_$i.log"
done
echo ""
echo "💡 Real-time monitoring:"
echo "   Human-readable: tail -f system_analytics_*.log"
echo "   GPU utilization: tail -f gpu_metrics_*.jsonl"
echo "   Connection events: tail -f connection_events_*.jsonl"
echo ""
echo "📈 Data analysis:"
echo "   Summary report: python analyze_analytics.py"
echo "   Last 6 hours: python analyze_analytics.py --since 6"
echo "   GPU analysis only: python analyze_analytics.py --type gpu"
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
    CURRENT_WORKERS=$(ps aux | grep -c "python.*worker.py.*--worker-address" || echo "0")
    if [ "$CURRENT_WORKERS" -lt "$NUM_GPUS" ]; then
        echo "⚠️  Some workers died unexpectedly. Expected $NUM_GPUS, found $CURRENT_WORKERS"
        echo "🔄 System will continue operating with reduced capacity"
        echo "💡 Check worker logs for error details"
        # Don't exit - keep system running with remaining workers
    fi
    
    sleep 5
done 