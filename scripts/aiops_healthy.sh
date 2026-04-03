#!/bin/bash
# scripts/aiops_healthy.sh - Consolidated AIops Health & Optimization Loop
# Enforces healthy state and reports comprehensive metrics.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 1. Load Host-friendly environment
export PGBOUNCER_ENABLED=True
export PGBOUNCER_SSLMODE=require
export PGBOUNCER_HOST=localhost
export PGBOUNCER_PORT=6432
export DATABASE_URL="postgresql://admin:password@localhost:5435/bsopt?sslmode=disable"

export REDIS_HOST=localhost
export REDIS_PORT=6380
export REDIS_URL="redis://:bsopt_redis_secret@localhost:6380/0"

export RABBITMQ_HOST=localhost
export RABBITMQ_PORT=5673
export RABBITMQ_USER=bsopt_admin
export RABBITMQ_PASSWORD="bsopt_rmq_secret"
export RABBITMQ_URL="amqp://${RABBITMQ_USER}:${RABBITMQ_PASSWORD}@${RABBITMQ_HOST}:${RABBITMQ_PORT}//" # Corrected URL format

export PROMETHEUS_URL="http://localhost:9090"
export PYTHONPATH="." # Ensure project root is in Python path
export BSOPT_ALLOW_WEAK_SECRETS=True

# 2. Enforce Prerequisites
echo "🔧 Enforcing AIops Prerequisites..."
source .venv/bin/activate

# Initialize Shared Memory
echo "🚀 Initializing Shared Memory..."
python3 scripts/initialize_shm.py --force

# Revamp Database Health Views
echo "🔄 Revamping Database Views..."
python3 scripts/revamp_db_views.py

# 3. Comprehensive Health Report
echo "📊 Running High-Performance System Sentinel..."
python3 scripts/system_sentinel.py

# 4. Detailed AIops Dashboard
echo "🌐 Launching AIops Terminal Dashboard..."
# Pass environment variables to the dashboard script if needed, or ensure it picks them up
python3 scripts/aiops_dashboard.py || echo "⚠️ Dashboard failed (Prometheus/other services might be unreachable)"

echo "✅ AIops Health Check Complete."
export RABBITMQ_HOST=localhost
export RABBITMQ_PORT=5673
export RABBITMQ_HOST=localhost
export RABBITMQ_PORT=5673
export RABBITMQ_USER=bsopt_admin
export RABBITMQ_PASSWORD=bsopt_rmq_secret
export PROMETHEUS_URL=http://localhost:9090
export RABBITMQ_HOST=localhost
export RABBITMQ_PORT=5673
export RABBITMQ_USER=bsopt_admin
export RABBITMQ_PASSWORD=bsopt_rmq_secret
export PROMETHEUS_URL=http://localhost:9090
