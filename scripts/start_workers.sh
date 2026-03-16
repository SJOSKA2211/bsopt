#!/bin/bash
set -e

echo " Starting Celery Workers (Local)..."

# Suppress warnings
export RAY_IGNORE_UNSTABLE_API_WARNING=1
export PYTHONWARNINGS="ignore"

# Setup Environment
export DATABASE_URL="postgresql://admin:password@localhost:5432/bsopt"
export REDIS_URL="redis://localhost:6379/0"
export RABBITMQ_URL="amqp://guest:guest@localhost:5672//"
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/services
export RAY_ADDRESS="auto"

# Start Celery Worker
if [ -f "/usr/local/bin/celery" ] || command -v celery >/dev/null 2>&1; then
    celery -A services.workers.tasks.celery_app worker --loglevel=info --concurrency=2 -n worker1@%h &
else
    python3 -m celery -A services.workers.tasks.celery_app worker --loglevel=info --concurrency=2 -n worker1@%h &
fi
PID_WORKER=$!

# Start Celery Beat (Scheduler)
if [ -f "/usr/local/bin/celery" ] || command -v celery >/dev/null 2>&1; then
    celery -A services.workers.tasks.celery_app beat --loglevel=info &
else
    python3 -m celery -A services.workers.tasks.celery_app beat --loglevel=info &
fi
PID_BEAT=$!

# Trap signals
trap "kill $PID_WORKER $PID_BEAT; exit" SIGINT SIGTERM

wait
