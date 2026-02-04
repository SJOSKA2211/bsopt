#!/bin/bash
set -e

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

echo "🚀 Starting BS-Opt Development Setup..."

# 1. Enable BuildKit
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# 2. Prevent system freezing by limiting build concurrency
export COMPOSE_PARALLEL_LIMIT=1
echo "✅ Build concurrency limited to 1 (Anti-Freeze enabled)"

# 3. Pull latest images from GHCR
echo "📦 Pulling latest pre-built images from GHCR..."
docker compose pull || echo "⚠️ Some images could not be pulled, will attempt to build locally."

# 4. Initialize network and volumes
echo "🌐 Initializing infrastructure..."
docker compose up -d postgres redis rabbitmq zookeeper kafka-1

echo "✨ Setup complete. Run 'docker compose up' to start the full stack."
