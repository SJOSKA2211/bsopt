#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "️ Launching Production Manifold Worker Substrate..."

# Load Production environment
source scripts/utils_env.sh
load_decrypted_secrets

export RAY_IGNORE_UNSTABLE_API_WARNING=1
export PYTHONWARNINGS="ignore"
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src
export RAY_ADDRESS=${RAY_ADDRESS:-auto}

# Execution Standard: Standardize on python3 -m celery
RUN_CELERY="python3 -m celery -A src.workers.tasks.celery_app"

echo " Starting Celery Worker Substrate..."
$RUN_CELERY worker --loglevel=info --concurrency=${CELERY_CONCURRENCY:-2} -n worker1@%h &
PID_WORKER=$!

echo " Starting Celery Scheduler (Beat)..."
$RUN_CELERY beat --loglevel=info &
PID_BEAT=$!

# Trap signals for graceful Production shutdown
trap "echo ' Shutting down workers...'; kill $PID_WORKER $PID_BEAT; exit" SIGINT SIGTERM

wait
