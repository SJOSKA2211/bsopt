#!/bin/bash
set -e

echo "🥒 Starting API (Local)..."

# Override for Local Docker Infra
export DATABASE_URL="postgresql+asyncpg://admin:password@localhost:5432/bsopt"
export REDIS_URL="redis://localhost:6379/0"
export JWT_SECRET="pickle-rick-secret"
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Activate Virtual Environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run Uvicorn with reload
python3 -m uvicorn src.api.main:app --reload --port 8000 --host 0.0.0.0
