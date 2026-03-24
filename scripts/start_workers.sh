#!/bin/bash
# scripts/start_workers.sh - Institutional Celery/Ray Worker Orchestrator (Zero-Mock)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "⚙️ Launching Institutional EquaFlow Worker Substrate..."

# Load institutional environment
source scripts/utils_env.sh
load_decrypted_secrets

# Institutional Runtime Environment
export RAY_IGNORE_UNSTABLE_API_WARNING=1
export PYTHONWARNINGS="ignore"
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src
export RAY_ADDRESS=${RAY_ADDRESS:-auto}

# Execution Standard: Standardize on python3 -m celery
RUN_CELERY="python3 -m celery -A src.workers.tasks.celery_app"

# Start Celery Worker (Institutional Concurrency)
echo "🐝 Starting Celery Worker Substrate..."
$RUN_CELERY worker --loglevel=info --concurrency=${CELERY_CONCURRENCY:-2} -n worker1@%h &
PID_WORKER=$!

# Start Celery Beat (Institutional Scheduler)
echo "📅 Starting Celery Scheduler (Beat)..."
$RUN_CELERY beat --loglevel=info &
PID_BEAT=$!

# Trap signals for graceful institutional shutdown
trap "echo '🛑 Shutting down workers...'; kill $PID_WORKER $PID_BEAT; exit" SIGINT SIGTERM

wait
