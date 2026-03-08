#!/bin/bash
set -e

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Start Infrastructure Services
echo " Starting Infrastructure (Postgres, Redis, RabbitMQ)..."
docker-compose -f docker-compose.dev.yml up -d postgres redis rabbitmq

echo "⏳ Waiting for database to stabilize..."
MAX_RETRIES=30
RETRY_COUNT=0
until docker-compose -f docker-compose.dev.yml exec -T postgres pg_isready -U admin -d bsopt > /dev/null 2>&1 || [ $RETRY_COUNT -eq $MAX_RETRIES ]; do
    printf "."
    sleep 1
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "\n❌ Database failed to stabilize. Check logs."
    exit 1
fi

# Run BSOpt Verification
echo "🥒 Running God-Mode Manifold Audit (Containerized)..."
docker-compose -f docker-compose.dev.yml run --rm test-runner python3 -m src.database.verify

echo "✅ Infrastructure containers launched and audited."
