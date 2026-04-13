# Stage 1: Base ML Environment
FROM python:3.11-alpine

# Install necessary system dependencies for ML frameworks and tools
RUN apk add --no-cache 
    build-base 
    libffi-dev 
    openssl-dev 
    git 
    curl 
    && rm -rf /var/cache/apk/*

# Install core Python ML dependencies
# This includes PyTorch, Lightning, MLflow, Ray, etc., based on pyproject.toml
RUN pip install --no-cache-dir 
    torch 
    lightning 
    mlflow 
    ray[default,train,tune]>=2.30.0 
    # Add other core ML dependencies as needed

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

# Copy ML specific source code
COPY src/ml/ ./src/ml/
COPY src/shared/utils/distributed.py ./src/shared/utils/ # For Ray orchestration
COPY src/shared/utils/ray_cluster_manager.py ./shared/utils/ # For Ray orchestration
# Copy any other necessary ML modules or configurations

# Copy Rust core if it's part of the ML deployment (assuming it's built separately)
# If the Rust core needs to be included here, adjust the multi-stage build accordingly.
# COPY --from=builder /build/wheels/*.whl /tmp/
# RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Environment setup
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI="http://mlflow:5000" # Example: Point to MLflow tracking server

# Expose ports if ML services need them (e.g., Ray dashboard, MLflow UI)
# EXPOSE 8265 # Ray Dashboard
# EXPOSE 5000 # MLflow Tracking Server

# Default command to run an ML task or service
# This will likely need to be a specific script or entrypoint for ML training/inference.
# Example: CMD ["python", "src/ml/trainer.py", "--config", "config/ml/trainer.yaml"]
# Example for MLflow Watchdog: CMD ["python", "scripts/mlflow_watchdog.py"]
CMD ["echo", "ML component Dockerfile created. Specify entrypoint command as needed."]
