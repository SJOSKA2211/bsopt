#!/bin/bash
# scripts/start_math_kernel.sh - Start the Quantitative Math Kernel Gateway

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🧮 Launching Pricing Math Kernel (BS, Monte Carlo, Greeks)..."

# Load Production environment
source scripts/utils_env.sh
load_decrypted_secrets

export PYTHONPATH=${PYTHONPATH:-}:$(pwd):$(pwd)/src

# We use port 8081 to avoid conflict with existing services
PORT=8081

if [ "${ENVIRONMENT:-development}" == "production" ]; then
    echo "️ Running in PRODUCTION mode..."
    exec python3 -m uvicorn src.math_kernel.main:app \
        --host 0.0.0.0 \
        --port $PORT \
        --workers $(nproc) \
        --loop uvloop \
        --no-access-log
else
    echo "️ Running in DEVELOPMENT mode with hot-reload..."
    exec python3 -m uvicorn src.math_kernel.main:app --reload --reload-dir src/math_kernel --port $PORT --host 0.0.0.0 --loop uvloop
fi
