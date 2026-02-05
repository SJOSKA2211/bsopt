#!/bin/bash
set -e

# Start Infrastructure Services
echo "🥒 Starting Infrastructure (Postgres, Redis, RabbitMQ)..."
docker compose -f docker-compose.dev.yml up -d postgres redis rabbitmq

echo "✅ Infrastructure is up."
docker compose -f docker-compose.dev.yml ps
