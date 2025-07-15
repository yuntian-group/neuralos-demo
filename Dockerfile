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

# Create startup script using uvicorn like the working version
RUN echo '#!/bin/bash' > start_hf_spaces.sh && \
    echo 'set -e' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo 'echo "🚀 Starting Neural OS for HF Spaces with uvicorn"' >> start_hf_spaces.sh && \
    echo 'echo "===================================="' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Start dispatcher directly with uvicorn' >> start_hf_spaces.sh && \
    echo 'echo "🎯 Starting dispatcher with uvicorn..."' >> start_hf_spaces.sh && \
    echo 'uvicorn dispatcher:app --host 0.0.0.0 --port 7860 > dispatcher.log 2>&1 &' >> start_hf_spaces.sh && \
    echo 'DISPATCHER_PID=$!' >> start_hf_spaces.sh && \
    echo 'echo "📊 Dispatcher PID: $DISPATCHER_PID"' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Wait for dispatcher to start' >> start_hf_spaces.sh && \
    echo 'echo "⏳ Waiting for dispatcher to initialize..."' >> start_hf_spaces.sh && \
    echo 'sleep 5' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Start worker' >> start_hf_spaces.sh && \
    echo 'echo "🔧 Starting worker..."' >> start_hf_spaces.sh && \
    echo 'python worker.py --worker-address localhost:8001 --dispatcher-url http://localhost:7860 > worker.log 2>&1 &' >> start_hf_spaces.sh && \
    echo 'WORKER_PID=$!' >> start_hf_spaces.sh && \
    echo 'echo "📊 Worker PID: $WORKER_PID"' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Wait for worker to initialize' >> start_hf_spaces.sh && \
    echo 'echo "⏳ Waiting for worker to initialize..."' >> start_hf_spaces.sh && \
    echo 'sleep 30' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo 'echo "✅ System ready!"' >> start_hf_spaces.sh && \
    echo 'echo "🌍 Web interface: http://localhost:7860"' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Keep running' >> start_hf_spaces.sh && \
    echo 'tail -f dispatcher.log worker.log &' >> start_hf_spaces.sh && \
    echo 'wait $DISPATCHER_PID' >> start_hf_spaces.sh && \
    chmod +x start_hf_spaces.sh

CMD ["bash", "start_hf_spaces.sh"]