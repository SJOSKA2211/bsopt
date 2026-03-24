#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 Launching Unified API..."

# Load shared environment utilities for secret derivation
source scripts/utils_env.sh
load_decrypted_secrets

export RAY_IGNORE_UNSTABLE_API_WARNING=1
export RAY_DEDUP_LOGS=0
export PYTHONWARNINGS="ignore::FutureWarning:ray"
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src

# Production-Grade ASGI Configuration
if [ "${ENVIRONMENT:-development}" == "production" ]; then
    echo "🏗️ Running in PRODUCTION mode with multi-worker Granian substrate..."
    exec python3 -m uvicorn src.api.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers $(nproc) \
        --loop uvloop \
        --http h2 \
        --no-access-log \
        --timeout-keep-alive 65
else
    echo "🛠️ Running in DEVELOPMENT mode with hot-reload..."
    exec python3 -m uvicorn src.api.main:app --reload --reload-dir src/api --port 8000 --host 0.0.0.0 --loop uvloop
fi
