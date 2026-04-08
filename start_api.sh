#!/bin/bash
# High-Performance Engine Runner (Optimized v2026 - PgBouncer SSL Require)
cd /home/kamau/bsopt
export REDIS_PORT=6380
export PYTHONPATH=.
export INSIDE_DOCKER=0
export BSOPT_ALLOW_WEAK_SECRETS=true
export PGBOUNCER_ENABLED=true
export ENVIRONMENT=dev

# Passwords from bootstrap
export REDIS_PASSWORD="REQUIRED_SET_BY_BOOTSTRAP"
export RABBITMQ_PASSWORD="REQUIRED_SET_BY_BOOTSTRAP"
export MINIO_ROOT_PASSWORD="REQUIRED_SET_BY_BOOTSTRAP"
export PGBOUNCER_ADMIN_PASSWORD="REQUIRED_SET_BY_BOOTSTRAP"
export JWT_SECRET="changeme_in_production_32_chars_long"
export BETTER_AUTH_SECRET="changeme_in_production_32_chars_long"
export RABBITMQ_URL="amqp://bsopt_admin:REQUIRED_SET_BY_BOOTSTRAP@localhost:5673//"

# Connection through PgBouncer with sslmode=require to satisfy PgBouncer config
export DATABASE_URL="postgresql://admin:REQUIRED_SET_BY_BOOTSTRAP@localhost:6432/bsopt?sslmode=require"

# Optimized Uvicorn
echo "Starting Engine with PgBouncer forced to TRUE and SSL Required..."
./.venv/bin/python -m uvicorn api.index:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop --http httptools --ws websockets --no-access-log
