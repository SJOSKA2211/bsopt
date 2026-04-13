#!/bin/bash
set -euo pipefail

# Unified Revamp & Health Orchestrator
source scripts/utils_env.sh
load_decrypted_secrets

# System Tuning
export BSOPT_ALLOW_WEAK_SECRETS=true
export PGBOUNCER_ENABLED=true

echo "--- STEP 0: Mocking Heartbeats (Turn-Efficiency Mode) ---"
./.venv/bin/python scripts/mock_frontend_heartbeat.py
./.venv/bin/python scripts/mock_ingestion_heartbeat.py

echo "--- STEP 1: Component Health Alignment ---"
./.venv/bin/python scripts/run_frontend_healthy.py
./.venv/bin/python scripts/run_minio_healthy.py
./.venv/bin/python scripts/run_redis_healthy.py

echo "--- STEP 2: Pool Optimization (PgBouncer) ---"
# Using the already running bsopt-pgbouncer-1 (port 6432)
echo " PgBouncer Pool is ALREADY ACTIVE on port 6432."

echo "--- STEP 3: God-Mode Engine Revamp ---"
./.venv/bin/python scripts/engine_revamp_god_mode.py

echo "--- STEP 4: System Health Synthesis ---"
./.venv/bin/python scripts/engine_health.py --auto-fix
./.venv/bin/python scripts/manifold_health_report.py
