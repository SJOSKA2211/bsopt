#!/bin/bash
# High-Performance Engine Runner (Optimized Uvicorn)
export REDIS_PORT=6380
export PYTHONPATH=.
export INSIDE_DOCKER=0
export BSOPT_ALLOW_WEAK_SECRETS=true

# Passwords from bootstrap
export REDIS_PASSWORD="REQUIRED_SET_BY_BOOTSTRAP"
export RABBITMQ_PASSWORD="REQUIRED_SET_BY_BOOTSTRAP"
export MINIO_ROOT_PASSWORD="REQUIRED_SET_BY_BOOTSTRAP"
export PGBOUNCER_ADMIN_PASSWORD="REQUIRED_SET_BY_BOOTSTRAP"
export JWT_SECRET="changeme_in_production_32_chars_long"
export BETTER_AUTH_SECRET="changeme_in_production_32_chars_long"

echo "Starting High-Performance Engine..."
./.venv/bin/python -m uvicorn api.index:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop --http httptools --ws websockets --no-access-log
