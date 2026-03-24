#!/bin/bash
# scripts/start_neural_pricing.sh - Institutional Neural Pricing Orchestrator (Zero-Mock)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🧠 Launching Institutional Neural Pricing Manifold..."

# Load institutional environment
source scripts/utils_env.sh
load_decrypted_secrets

# Institutional Runtime Environment
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src

# Production-Grade ASGI Configuration
if [ "${ENVIRONMENT:-development}" == "production" ]; then
    echo "🏗️ Running in PRODUCTION mode..."
    exec python3 -m uvicorn src.math_kernel.pricing.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers $(nproc) \
        --loop uvloop \
        --no-access-log
else
    echo "🛠️ Running in DEVELOPMENT mode with hot-reload..."
    exec python3 -m uvicorn src.math_kernel.pricing.main:app --reload --reload-dir src/math_kernel/pricing --port 8000 --host 0.0.0.0 --loop uvloop
fi
