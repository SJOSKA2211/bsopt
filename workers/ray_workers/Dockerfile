# Stage 1: Base Ray Worker Environment
FROM python:3.11-alpine

# Install necessary system dependencies for Ray
RUN apk add --no-cache 
    build-base 
    libffi-dev 
    openssl-dev 
    && rm -rf /var/cache/apk/*

# Install Ray and its core dependencies
# Specify a version if needed, e.g., ray[default,train,tune]>=2.30.0
RUN pip install --no-cache-dir 
    ray[default,train,tune]>=2.30.0 
    grpcio-tools>=1.78.0 
    # Add other core dependencies if necessary, e.g., for specific ML frameworks

WORKDIR /app

# Copy requirements and install Python dependencies
# Assuming core dependencies are in requirements.txt, and Ray-specific ones are handled above or within requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Ray worker code
COPY workers/ray_workers.py ./workers/
COPY src ./src/
COPY shared/utils/ray_cluster_manager.py ./shared/utils/
COPY shared/utils/distributed.py ./shared/utils/
# Copy any other necessary modules or configurations

# Environment setup
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV RAY_ADDRESS=auto # Or specify a cluster address if needed

# Expose ports if Ray dashboard or other services need it (optional)
# EXPOSE 8265 # Ray Dashboard
# EXPOSE 6379 # Redis for distributed state if used

# Command to run the Ray worker
# This command will depend on how Ray actors are initiated and managed.
# Often, Ray actors are started by a supervisor script or managed by Ray itself.
# For a worker, it might involve running a specific script that initializes Ray.
# Example: This CMD assumes a script that starts the Ray worker processes.
# You might need a supervisor or a specific entrypoint script.
CMD ["ray", "start", "--head", "--port=6379", "--redis-password=your-redis-password", "--block"]
# Note: The CMD above is a placeholder for starting a Ray head node.
# For worker nodes, you'd typically connect to an existing cluster.
# A more specific command might be needed based on how Ray actors are launched.
# Example for a worker connecting to an existing cluster:
# CMD ["python", "workers/ray_workers.py", "--address=auto"]
