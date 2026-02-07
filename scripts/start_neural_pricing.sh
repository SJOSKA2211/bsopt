#!/bin/bash
set -e

echo "🥒 Starting Neural Pricing Service (Local)..."

# Setup Environment
export DATABASE_URL="postgresql://admin:password@localhost:5432/bsopt"
export REDIS_URL="redis://localhost:6379/0"
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Activate Virtual Environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run Uvicorn with reload on port 8001 (as per docker-compose mapping 8001:8000)
python3 -m uvicorn src.pricing.main:app --reload --reload-dir src --port 8001 --host 0.0.0.0
