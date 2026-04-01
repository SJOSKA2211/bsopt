#!/bin/bash
# scripts/start_ml_inference.sh - Start the ML Inference (Neural Pricing) Engine

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🧠 Launching ML Inference (Option Pricing ONNX Model)..."

# Load Production environment
source scripts/utils_env.sh
load_decrypted_secrets

export PYTHONPATH=${PYTHONPATH:-}:$(pwd):$(pwd)/src

# We use port 5002 to avoid conflict with existing services
PORT=5002

if [ "${ENVIRONMENT:-development}" == "production" ]; then
    echo "🏗️ Running in PRODUCTION mode..."
    exec python3 -m uvicorn src.ml.serving.serve:app \
        --host 0.0.0.0 \
        --port $PORT \
        --workers $(nproc) \
        --loop uvloop \
        --no-access-log
else
    echo "🛠️ Running in DEVELOPMENT mode with hot-reload..."
    exec python3 -m uvicorn src.ml.serving.serve:app --reload --reload-dir src/ml/serving --port $PORT --host 0.0.0.0 --loop uvloop
fi
