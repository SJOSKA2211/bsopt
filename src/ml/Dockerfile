# Stage 1: Base ML Environment
FROM python:3.12.13-alpine AS base

# Install necessary system dependencies for ML frameworks and tools
RUN apk add --no-cache \
    build-base \
    libffi-dev \
    openssl-dev \
    git \
    curl

# Install core Python ML dependencies (PURE CPU ONLY)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch \
    lightning \
    mlflow \
    ray[default,train,tune]>=2.30.0
    # Add other core ML dependencies as needed

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

# Copy ML specific source code
COPY src/ml/ /app/src/ml/
COPY src/shared/utils/distributed.py /app/src/shared/utils/
COPY src/shared/utils/ray_cluster_manager.py /app/src/shared/utils/
# Copy any other necessary ML modules or configurations

# Copy Rust core if it's part of the ML deployment (assuming it's built separately)
# If the Rust core needs to be included here, adjust the multi-stage build accordingly.
# COPY --from=builder /build/wheels/*.whl /tmp/
# RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Environment setup
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI="http://mlflow:5000"

# Expose ports if ML services need them (e.g., Ray dashboard, MLflow UI)
# EXPOSE 8265 # Ray Dashboard
# EXPOSE 5000 # MLflow Tracking Server

# Default command to run an ML task or service
# This will likely need to be a specific script or entrypoint for ML training/inference.
# Example: CMD ["python", "src/ml/trainer.py", "--config", "config/ml/trainer.yaml"]
# Example for MLflow Watchdog: CMD ["python", "scripts/mlflow_watchdog.py"]
CMD ["echo", "ML component Dockerfile created. Specify entrypoint command as needed."]
