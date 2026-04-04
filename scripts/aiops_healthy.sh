#!/bin/bash
# scripts/aiops_healthy.sh - Consolidated AIops Health & Optimization Loop
# Enforces healthy state and reports comprehensive metrics.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 1. Load Hardened environment with secret decryption
source scripts/utils_env.sh
load_decrypted_secrets

# Host-friendly overrides
export PGBOUNCER_ENABLED=True
export PGBOUNCER_SSLMODE=require
export PGBOUNCER_HOST=127.0.0.1
export PGBOUNCER_PORT=6432
export PGBOUNCER_ADMIN_USER=${POSTGRES_USER:-admin}
export PGBOUNCER_ADMIN_PASSWORD=${POSTGRES_PASSWORD}

# Direct DB connection for view revamping (Port 5435 is mapped to PG directly)
export DATABASE_URL="postgresql://admin:${POSTGRES_PASSWORD}@localhost:5435/bsopt?sslmode=disable"

export REDIS_HOST=localhost
export REDIS_PORT=6380
export REDIS_URL="redis://:${REDIS_PASSWORD}@127.0.0.1:6380/0"

export RABBITMQ_HOST=localhost
export RABBITMQ_PORT=5673
export RABBITMQ_USER=${RABBITMQ_USER:-bsopt_admin}
export RABBITMQ_PASSWORD=${RABBITMQ_PASSWORD}
export RABBITMQ_URL="amqp://${RABBITMQ_USER}:${RABBITMQ_PASSWORD}@${RABBITMQ_HOST}:${RABBITMQ_PORT}//"

export PROMETHEUS_URL="http://localhost:9090"
export PYTHONPATH="."
export BSOPT_ALLOW_WEAK_SECRETS=True

# 2. Start Core Infrastructure
echo "🚀 Ensuring Core Infrastructure is running..."
docker compose -f infrastructure/orchestration/docker-compose.yml up -d postgres pgbouncer redis envoy nginx

# 3. Parallel Initialization (Revamp & SHM)
echo "🔧 Parallelizing AIops Revamp..."
source .venv/bin/activate
(python3 scripts/initialize_shm.py --force > /dev/null 2>&1 && echo "✅ SHM Initialized") &
(python3 scripts/revamp_db_views.py > /dev/null 2>&1 && echo "✅ DB Views Revamped") &
wait

# 4. Autonomous Health Loop (Run until Healthy)
MAX_RETRIES=12
RETRY_COUNT=0
SLEEP_INTERVAL=10

echo "🔍 Entering Sentinel Health Loop (Max $((MAX_RETRIES * SLEEP_INTERVAL))s)..."

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "📊 Health Probe $((RETRY_COUNT + 1))/$MAX_RETRIES..."
    if python3 scripts/system_sentinel.py; then
        echo "✅ AIops HEALTHY: All systems pressurized and operational."
        exit 0
    fi
    
    echo "⚠️ Systems unstable. Retrying in ${SLEEP_INTERVAL}s..."
    RETRY_COUNT=$((RETRY_COUNT + 1))
    sleep $SLEEP_INTERVAL
done

echo "❌ AIops TIMEOUT: System failed to stabilize within the allotted window."
exit 1
