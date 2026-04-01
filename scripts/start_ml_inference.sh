#!/bin/bash
# scripts/start_ml_inference.sh - Start the ML Inference (Neural Pricing) Engine

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🧠 Launching ML Inference (Option Pricing ONNX Model)..."

# Load Production# Set environment variables
export ENVIRONMENT="development"
export PORT=5002
export PYTHONPATH=${PYTHONPATH:-}:$(pwd):$(pwd)/src
export BSOPT_ALLOW_WEAK_SECRETS=1
export PGBOUNCER_ADMIN_PASSWORD="change_it_for_development"

# Load Production environment
source scripts/utils_env.sh
load_decrypted_secrets

# Fix hostname resolution when running locally outside of docker
export DATABASE_URL="${DATABASE_URL:-}"
export DATABASE_URL="${DATABASE_URL//pgbouncer:6432/localhost:5435}"
export DATABASE_URL="${DATABASE_URL//pgbouncer/localhost:5435}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI//pgbouncer:6432/localhost:5435}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI//pgbouncer/localhost:5435}"

# Local gRPC and model overrides
export ML_SERVICE_GRPC_URL="0.0.0.0:50051"
export NN_MODEL_PATH="models/latest_pricing.onnx"
export XGB_ONNX_MODEL_PATH="models/latest_pricing.onnx"

# We use port 5002 to avoid conflict with existing services
PORT=5002

if [ "${ENVIRONMENT:-development}" == "production" ]; then
    echo "🏗️ Running in PRODUCTION mode..."
    exec uv run python3 -m uvicorn src.ml.serving.serve:app \
        --host 0.0.0.0 \
        --port $PORT \
        --workers $(nproc) \
        --loop uvloop \
        --no-access-log
else
    echo "🛠️ Running in DEVELOPMENT mode with hot-reload..."
    exec uv run python3 -m uvicorn src.ml.serving.serve:app --port $PORT --host 0.0.0.0 --loop uvloop
fi
