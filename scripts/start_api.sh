#!/bin/bash
set -e

echo " Starting API (Local)..."

# Suppress Ray and Unstable API warnings
export RAY_IGNORE_UNSTABLE_API_WARNING=1
export RAY_DEDUP_LOGS=0
export PYTHONWARNINGS="ignore::FutureWarning:ray"

# Override for Local Docker Infra
export DATABASE_URL="postgresql://admin:password@localhost:5432/bsopt"
export REDIS_URL="redis://localhost:6379/0"
export JWT_SECRET="development_secret_high_performance_secure_system_key_manifold_32_char"
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src

# Run Uvicorn with optimized settings
if [ "$ENVIRONMENT" == "prod" ] || [ "$ENVIRONMENT" == "production" ]; then
    echo "Running in PRODUCTION mode with multiple workers..."
    exec python3 -m uvicorn src.api.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers $(nproc) \
        --loop uvloop \
        --http h11 \
        --no-access-log \
        --timeout-keep-alive 65
else
    echo "Running in DEVELOPMENT mode with reload..."
    python3 -m uvicorn src.api.main:app --reload --reload-dir src/api --port 8000 --host 0.0.0.0 --loop uvloop
fi
