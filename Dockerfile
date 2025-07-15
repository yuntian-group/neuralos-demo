# Use the official Python 3.9 image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9


RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 curl -y

# Set the working directory to /code
WORKDIR /code

# Copy the current directory contents into the container at /code
COPY ./requirements.txt /code/requirements.txt

# Install requirements.txt 

RUN pip install pip==24.0

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
# Switch to the "user" user
USER user
# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# Create a startup script for HF Spaces using echo commands
RUN echo '#!/bin/bash' > start_hf_spaces.sh && \
    echo 'set -e' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo 'echo "🚀 Starting Neural OS for HF Spaces"' >> start_hf_spaces.sh && \
    echo 'echo "===================================="' >> start_hf_spaces.sh && \
    echo 'echo "📁 Current directory: $(pwd)"' >> start_hf_spaces.sh && \
    echo 'echo "📋 Files in current directory:"' >> start_hf_spaces.sh && \
    echo 'ls -la' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Check if required files exist' >> start_hf_spaces.sh && \
    echo 'if [[ ! -f "dispatcher.py" ]]; then' >> start_hf_spaces.sh && \
    echo '    echo "❌ Error: dispatcher.py not found"' >> start_hf_spaces.sh && \
    echo '    exit 1' >> start_hf_spaces.sh && \
    echo 'fi' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo 'if [[ ! -f "worker.py" ]]; then' >> start_hf_spaces.sh && \
    echo '    echo "❌ Error: worker.py not found"' >> start_hf_spaces.sh && \
    echo '    exit 1' >> start_hf_spaces.sh && \
    echo 'fi' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo 'if [[ ! -f "static/index.html" ]]; then' >> start_hf_spaces.sh && \
    echo '    echo "❌ Error: static/index.html not found"' >> start_hf_spaces.sh && \
    echo '    exit 1' >> start_hf_spaces.sh && \
    echo 'fi' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo 'echo "✅ All required files found"' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Start dispatcher in background' >> start_hf_spaces.sh && \
    echo 'echo "🎯 Starting dispatcher..."' >> start_hf_spaces.sh && \
    echo 'python dispatcher.py --port 7860 > dispatcher.log 2>&1 &' >> start_hf_spaces.sh && \
    echo 'DISPATCHER_PID=$!' >> start_hf_spaces.sh && \
    echo 'echo "📊 Dispatcher PID: $DISPATCHER_PID"' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Wait for dispatcher to start and check if it is running' >> start_hf_spaces.sh && \
    echo 'echo "⏳ Waiting for dispatcher to initialize..."' >> start_hf_spaces.sh && \
    echo 'sleep 5' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo 'if ! kill -0 $DISPATCHER_PID 2>/dev/null; then' >> start_hf_spaces.sh && \
    echo '    echo "❌ Dispatcher failed to start"' >> start_hf_spaces.sh && \
    echo '    echo "📋 Dispatcher log:"' >> start_hf_spaces.sh && \
    echo '    cat dispatcher.log' >> start_hf_spaces.sh && \
    echo '    exit 1' >> start_hf_spaces.sh && \
    echo 'fi' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Test if dispatcher is responding' >> start_hf_spaces.sh && \
    echo 'echo "🔍 Testing dispatcher health..."' >> start_hf_spaces.sh && \
    echo 'curl -f http://localhost:7860/ > /dev/null 2>&1' >> start_hf_spaces.sh && \
    echo 'if [ $? -eq 0 ]; then' >> start_hf_spaces.sh && \
    echo '    echo "✅ Dispatcher is responding to HTTP requests"' >> start_hf_spaces.sh && \
    echo 'else' >> start_hf_spaces.sh && \
    echo '    echo "⚠️ Dispatcher HTTP test failed, but continuing..."' >> start_hf_spaces.sh && \
    echo 'fi' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Start single worker' >> start_hf_spaces.sh && \
    echo 'echo "🔧 Starting worker..."' >> start_hf_spaces.sh && \
    echo 'python worker.py --worker-address localhost:8001 --dispatcher-url http://localhost:7860 > worker.log 2>&1 &' >> start_hf_spaces.sh && \
    echo 'WORKER_PID=$!' >> start_hf_spaces.sh && \
    echo 'echo "📊 Worker PID: $WORKER_PID"' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Wait for worker to initialize' >> start_hf_spaces.sh && \
    echo 'echo "⏳ Waiting for worker to initialize..."' >> start_hf_spaces.sh && \
    echo 'sleep 30' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Check if worker is still running' >> start_hf_spaces.sh && \
    echo 'if ! kill -0 $WORKER_PID 2>/dev/null; then' >> start_hf_spaces.sh && \
    echo '    echo "❌ Worker failed to start"' >> start_hf_spaces.sh && \
    echo '    echo "📋 Worker log:"' >> start_hf_spaces.sh && \
    echo '    cat worker.log' >> start_hf_spaces.sh && \
    echo '    echo "📋 Dispatcher log:"' >> start_hf_spaces.sh && \
    echo '    cat dispatcher.log' >> start_hf_spaces.sh && \
    echo '    exit 1' >> start_hf_spaces.sh && \
    echo 'fi' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo 'echo "✅ System ready!"' >> start_hf_spaces.sh && \
    echo 'echo "🌍 Web interface: http://localhost:7860"' >> start_hf_spaces.sh && \
    echo 'echo "📊 Dispatcher PID: $DISPATCHER_PID"' >> start_hf_spaces.sh && \
    echo 'echo "📊 Worker PID: $WORKER_PID"' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Function to cleanup' >> start_hf_spaces.sh && \
    echo 'cleanup() {' >> start_hf_spaces.sh && \
    echo '    echo "🛑 Shutting down..."' >> start_hf_spaces.sh && \
    echo '    kill $DISPATCHER_PID $WORKER_PID 2>/dev/null || true' >> start_hf_spaces.sh && \
    echo '    exit 0' >> start_hf_spaces.sh && \
    echo '}' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo 'trap cleanup SIGINT SIGTERM' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Keep the script running by following the dispatcher log' >> start_hf_spaces.sh && \
    echo 'echo "📋 Following dispatcher log (Ctrl+C to stop):"' >> start_hf_spaces.sh && \
    echo 'tail -f dispatcher.log &' >> start_hf_spaces.sh && \
    echo 'TAIL_PID=$!' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Wait for dispatcher (main process)' >> start_hf_spaces.sh && \
    echo 'wait $DISPATCHER_PID' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Clean up tail process' >> start_hf_spaces.sh && \
    echo 'kill $TAIL_PID 2>/dev/null || true' >> start_hf_spaces.sh && \
    chmod +x start_hf_spaces.sh

CMD ["bash", "start_hf_spaces.sh"]