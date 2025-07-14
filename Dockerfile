# Use the official Python 3.9 image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9


RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Set the working directory to /code
WORKDIR /code

# Copy the current directory contents into the container at /code
COPY ./requirements.txt /code/requirements.txt

# Install requirements.txt 

RUN pip install pip==24.0

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#RUN git clone https://github.com/da03/latent-diffusion.git

# Install latent-diffusion in editable mode
#RUN pip install -e ./latent-diffusion

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

# Create a startup script for HF Spaces
COPY --chown=user <<EOF $HOME/app/start_hf_spaces.sh
#!/bin/bash
set -e

echo "🚀 Starting Neural OS for HF Spaces"
echo "===================================="

# Start dispatcher in background
echo "🎯 Starting dispatcher..."
python dispatcher.py --port 7860 > dispatcher.log 2>&1 &
DISPATCHER_PID=\$!

# Wait for dispatcher to start
sleep 3

# Start single worker (HF Spaces typically has 1 GPU or CPU)
echo "🔧 Starting worker..."
python worker.py --worker-address localhost:8001 --dispatcher-url http://localhost:7860 > worker.log 2>&1 &
WORKER_PID=\$!

# Wait for worker to initialize
echo "⏳ Waiting for worker to initialize..."
sleep 30

echo "✅ System ready!"
echo "🌍 Web interface: http://localhost:7860"

# Function to cleanup
cleanup() {
    echo "🛑 Shutting down..."
    kill \$DISPATCHER_PID \$WORKER_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for dispatcher (main process)
wait \$DISPATCHER_PID
EOF

RUN chmod +x $HOME/app/start_hf_spaces.sh

CMD ["bash", "start_hf_spaces.sh"]