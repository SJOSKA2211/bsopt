#!/bin/bash
set -e

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Start Infrastructure Services
echo " Starting Infrastructure (Postgres, Redis, RabbitMQ)..."
docker-compose -f docker-compose.dev.yml up -d postgres redis rabbitmq

echo "✅ Infrastructure containers launched."
