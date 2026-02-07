#!/bin/bash
set -e

echo "🥒 Starting API (Local)..."

# Suppress Ray and Unstable API warnings
export RAY_IGNORE_UNSTABLE_API_WARNING=1
export RAY_DEDUP_LOGS=0
export PYTHONWARNINGS="ignore::FutureWarning:ray"

# Override for Local Docker Infra
export DATABASE_URL="postgresql://admin:password@localhost:5432/bsopt"
export REDIS_URL="redis://localhost:6379/0"
export JWT_SECRET="development_secret_pickle_rick_is_the_best_scientist_in_the_multiverse_32_char"
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Activate Virtual Environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run Uvicorn with reload
python3 -m uvicorn src.api.main:app --reload --reload-dir src --port 8000 --host 0.0.0.0
