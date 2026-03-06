#!/bin/bash
set -e

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Start Infrastructure Services
echo " Starting Infrastructure (Postgres, Redis, RabbitMQ)..."
docker-compose -f docker-compose.dev.yml up -d postgres redis rabbitmq

echo "⌛ Waiting for database to stabilize..."
sleep 5

# Run BSOpt Verification
echo "🥒 Running God-Mode Manifold Audit (Containerized)..."
docker-compose -f docker-compose.dev.yml run --rm test-runner python3 -m src.database.verify

echo "✅ Infrastructure containers launched and audited."
