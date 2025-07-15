# Use the official Python 3.9 image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 curl nginx -y

# Set the working directory to /code
WORKDIR /code

# Copy the current directory contents into the container at /code
COPY ./requirements.txt /code/requirements.txt

# Install requirements.txt 
RUN pip install pip==24.0
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Create directories for nginx that user can write to
RUN mkdir -p /home/user/nginx/logs /home/user/nginx/cache /home/user/nginx/tmp && \
    chown -R user:user /home/user/nginx

# Switch to the "user" user
USER user
# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# Create custom nginx.conf for non-root operation
RUN echo 'pid /home/user/nginx/nginx.pid;' > /home/user/nginx.conf && \
    echo 'error_log /home/user/nginx/logs/error.log;' >> /home/user/nginx.conf && \
    echo 'worker_processes 1;' >> /home/user/nginx.conf && \
    echo '' >> /home/user/nginx.conf && \
    echo 'events {' >> /home/user/nginx.conf && \
    echo '    worker_connections 1024;' >> /home/user/nginx.conf && \
    echo '}' >> /home/user/nginx.conf && \
    echo '' >> /home/user/nginx.conf && \
    echo 'http {' >> /home/user/nginx.conf && \
    echo '    access_log /home/user/nginx/logs/access.log;' >> /home/user/nginx.conf && \
    echo '    client_body_temp_path /home/user/nginx/tmp/client_body;' >> /home/user/nginx.conf && \
    echo '    proxy_temp_path /home/user/nginx/tmp/proxy;' >> /home/user/nginx.conf && \
    echo '    fastcgi_temp_path /home/user/nginx/tmp/fastcgi;' >> /home/user/nginx.conf && \
    echo '    uwsgi_temp_path /home/user/nginx/tmp/uwsgi;' >> /home/user/nginx.conf && \
    echo '    scgi_temp_path /home/user/nginx/tmp/scgi;' >> /home/user/nginx.conf && \
    echo '' >> /home/user/nginx.conf && \
    echo '    server {' >> /home/user/nginx.conf && \
    echo '        listen 7860;' >> /home/user/nginx.conf && \
    echo '        server_name localhost;' >> /home/user/nginx.conf && \
    echo '' >> /home/user/nginx.conf && \
    echo '        # WebSocket support' >> /home/user/nginx.conf && \
    echo '        location /ws {' >> /home/user/nginx.conf && \
    echo '            proxy_pass http://localhost:8080/ws;' >> /home/user/nginx.conf && \
    echo '            proxy_http_version 1.1;' >> /home/user/nginx.conf && \
    echo '            proxy_set_header Upgrade $http_upgrade;' >> /home/user/nginx.conf && \
    echo '            proxy_set_header Connection "upgrade";' >> /home/user/nginx.conf && \
    echo '            proxy_set_header Host $host;' >> /home/user/nginx.conf && \
    echo '            proxy_set_header X-Real-IP $remote_addr;' >> /home/user/nginx.conf && \
    echo '            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;' >> /home/user/nginx.conf && \
    echo '            proxy_set_header X-Forwarded-Proto $scheme;' >> /home/user/nginx.conf && \
    echo '            proxy_read_timeout 86400;' >> /home/user/nginx.conf && \
    echo '            proxy_send_timeout 86400;' >> /home/user/nginx.conf && \
    echo '        }' >> /home/user/nginx.conf && \
    echo '' >> /home/user/nginx.conf && \
    echo '        # Regular HTTP requests' >> /home/user/nginx.conf && \
    echo '        location / {' >> /home/user/nginx.conf && \
    echo '            proxy_pass http://localhost:8080;' >> /home/user/nginx.conf && \
    echo '            proxy_set_header Host $host;' >> /home/user/nginx.conf && \
    echo '            proxy_set_header X-Real-IP $remote_addr;' >> /home/user/nginx.conf && \
    echo '            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;' >> /home/user/nginx.conf && \
    echo '            proxy_set_header X-Forwarded-Proto $scheme;' >> /home/user/nginx.conf && \
    echo '        }' >> /home/user/nginx.conf && \
    echo '    }' >> /home/user/nginx.conf && \
    echo '}' >> /home/user/nginx.conf

# Create necessary temp directories
RUN mkdir -p /home/user/nginx/tmp/{client_body,proxy,fastcgi,uwsgi,scgi}

# Create startup script
RUN echo '#!/bin/bash' > start_hf_spaces.sh && \
    echo 'set -e' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo 'echo "🚀 Starting Neural OS for HF Spaces with nginx proxy"' >> start_hf_spaces.sh && \
    echo 'echo "===================================="' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Start nginx as user' >> start_hf_spaces.sh && \
    echo 'echo "🌐 Starting nginx proxy..."' >> start_hf_spaces.sh && \
    echo 'nginx -c /home/user/nginx.conf -t' >> start_hf_spaces.sh && \
    echo 'nginx -c /home/user/nginx.conf &' >> start_hf_spaces.sh && \
    echo 'NGINX_PID=$!' >> start_hf_spaces.sh && \
    echo 'echo "📊 Nginx PID: $NGINX_PID"' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Start dispatcher' >> start_hf_spaces.sh && \
    echo 'echo "🎯 Starting dispatcher..."' >> start_hf_spaces.sh && \
    echo 'python dispatcher.py --port 8080 > dispatcher.log 2>&1 &' >> start_hf_spaces.sh && \
    echo 'DISPATCHER_PID=$!' >> start_hf_spaces.sh && \
    echo 'echo "📊 Dispatcher PID: $DISPATCHER_PID"' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Wait for dispatcher to start' >> start_hf_spaces.sh && \
    echo 'echo "⏳ Waiting for dispatcher to initialize..."' >> start_hf_spaces.sh && \
    echo 'sleep 5' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Test if everything is working' >> start_hf_spaces.sh && \
    echo 'echo "🔍 Testing system health..."' >> start_hf_spaces.sh && \
    echo 'curl -f http://localhost:7860/ > /dev/null 2>&1' >> start_hf_spaces.sh && \
    echo 'if [ $? -eq 0 ]; then' >> start_hf_spaces.sh && \
    echo '    echo "✅ System is responding"' >> start_hf_spaces.sh && \
    echo 'else' >> start_hf_spaces.sh && \
    echo '    echo "❌ System health check failed"' >> start_hf_spaces.sh && \
    echo '    exit 1' >> start_hf_spaces.sh && \
    echo 'fi' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Start worker' >> start_hf_spaces.sh && \
    echo 'echo "🔧 Starting worker..."' >> start_hf_spaces.sh && \
    echo 'python worker.py --worker-address localhost:8001 --dispatcher-url http://localhost:8080 > worker.log 2>&1 &' >> start_hf_spaces.sh && \
    echo 'WORKER_PID=$!' >> start_hf_spaces.sh && \
    echo 'echo "📊 Worker PID: $WORKER_PID"' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Wait for worker to initialize' >> start_hf_spaces.sh && \
    echo 'echo "⏳ Waiting for worker to initialize..."' >> start_hf_spaces.sh && \
    echo 'sleep 30' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Final health check' >> start_hf_spaces.sh && \
    echo 'if kill -0 $DISPATCHER_PID 2>/dev/null && kill -0 $WORKER_PID 2>/dev/null; then' >> start_hf_spaces.sh && \
    echo '    echo "✅ System ready!"' >> start_hf_spaces.sh && \
    echo '    echo "🌍 Web interface: http://localhost:7860"' >> start_hf_spaces.sh && \
    echo '    echo "📊 Dispatcher: http://localhost:8080"' >> start_hf_spaces.sh && \
    echo 'else' >> start_hf_spaces.sh && \
    echo '    echo "❌ System failed to start properly"' >> start_hf_spaces.sh && \
    echo '    echo "📋 Dispatcher log:"' >> start_hf_spaces.sh && \
    echo '    tail -n 20 dispatcher.log' >> start_hf_spaces.sh && \
    echo '    echo "📋 Worker log:"' >> start_hf_spaces.sh && \
    echo '    tail -n 20 worker.log' >> start_hf_spaces.sh && \
    echo '    exit 1' >> start_hf_spaces.sh && \
    echo 'fi' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Function to cleanup' >> start_hf_spaces.sh && \
    echo 'cleanup() {' >> start_hf_spaces.sh && \
    echo '    echo "🛑 Shutting down..."' >> start_hf_spaces.sh && \
    echo '    kill $NGINX_PID $DISPATCHER_PID $WORKER_PID 2>/dev/null || true' >> start_hf_spaces.sh && \
    echo '    exit 0' >> start_hf_spaces.sh && \
    echo '}' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo 'trap cleanup SIGINT SIGTERM' >> start_hf_spaces.sh && \
    echo '' >> start_hf_spaces.sh && \
    echo '# Keep running' >> start_hf_spaces.sh && \
    echo 'echo "📋 Following logs (Ctrl+C to stop):"' >> start_hf_spaces.sh && \
    echo 'tail -f dispatcher.log worker.log /home/user/nginx/logs/error.log &' >> start_hf_spaces.sh && \
    echo 'wait $DISPATCHER_PID' >> start_hf_spaces.sh && \
    chmod +x start_hf_spaces.sh

CMD ["bash", "start_hf_spaces.sh"]