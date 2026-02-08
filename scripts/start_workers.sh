#!/bin/bash
set -e

echo "🥒 Starting Celery Workers (Local)..."

# Suppress warnings
export RAY_IGNORE_UNSTABLE_API_WARNING=1
export PYTHONWARNINGS="ignore"

# Setup Environment
export DATABASE_URL="postgresql://admin:password@localhost:5432/bsopt"
export REDIS_URL="redis://localhost:6379/0"
export RABBITMQ_URL="amqp://guest:guest@localhost:5672//"
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export RAY_ADDRESS="auto"

# Activate Virtual Environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Start Celery Worker
.venv/bin/celery -A src.tasks.celery_app worker --loglevel=info --concurrency=2 -n worker1@%h &
PID_WORKER=$!

# Start Celery Beat (Scheduler)
.venv/bin/celery -A src.tasks.celery_app beat --loglevel=info &
PID_BEAT=$!

# Trap signals
trap "kill $PID_WORKER $PID_BEAT; exit" SIGINT SIGTERM

wait
