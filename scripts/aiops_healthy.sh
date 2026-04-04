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

# 2. Start Services if needed
echo "🚀 Ensuring Database and Pooler are running..."
docker compose -f infrastructure/orchestration/docker-compose.yml up -d postgres pgbouncer

# 3. Enforce Prerequisites
echo "🔧 Enforcing AIops Prerequisites..."
source .venv/bin/activate

# Warm up PgBouncer pool
echo "☕ Warming up PgBouncer pool..."
PGPASSWORD="${POSTGRES_PASSWORD}" psql -h 127.0.0.1 -p 6432 -U admin -d bsopt -c "SELECT 1" > /dev/null 2>&1 || echo "⚠️ Warmup failed (PgBouncer might still be starting)"

# Initialize Shared Memory
echo "🚀 Initializing Shared Memory..."
python3 scripts/initialize_shm.py --force

# Revamp Database Health Views
echo "🔄 Revamping Database Views..."
python3 scripts/revamp_db_views.py

# 4. Comprehensive Health Report
echo "📊 Running High-Performance System Sentinel..."
python3 scripts/system_sentinel.py

# 5. Detailed AIops Dashboard
echo "🌐 Launching AIops Terminal Dashboard..."
# dashboard is usually TUI, so we might want to skip it in non-interactive mode or run it briefly
# python3 scripts/aiops_dashboard.py --once || echo "⚠️ Dashboard failed"

echo "✅ AIops Health Check Complete."
