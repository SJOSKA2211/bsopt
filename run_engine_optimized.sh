#!/bin/bash
# Optimized Engine Runner (v2026)
export REDIS_PORT=6380
export PYTHONPATH=.
export INSIDE_DOCKER=0
export BSOPT_ALLOW_WEAK_SECRETS=true

# Passwords from bootstrap (placeholder used in this environment)
export REDIS_PASSWORD="REQUIRED_SET_BY_BOOTSTRAP"
export RABBITMQ_PASSWORD="REQUIRED_SET_BY_BOOTSTRAP"
export MINIO_ROOT_PASSWORD="REQUIRED_SET_BY_BOOTSTRAP"
export PGBOUNCER_ADMIN_PASSWORD="REQUIRED_SET_BY_BOOTSTRAP"
export JWT_SECRET="changeme_in_production_32_chars_long"
export BETTER_AUTH_SECRET="changeme_in_production_32_chars_long"

# Standardize on Granian with uvloop for maximum throughput
echo "Starting Optimized Engine with Granian..."
./.venv/bin/python -m granian --interface asgi api.index:app --host 0.0.0.0 --port 8000 --workers 1 --backlog 2048 --loop uvloop
