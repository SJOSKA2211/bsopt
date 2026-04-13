#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 1. Detect Container Engine
if command -v docker &> /dev/null; then
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
else
    echo " Error: docker is not installed."
    exit 1
fi

echo " Starting Ray Head and Optimized Workers..."

# 2. Start Ray Cluster
$COMPOSE_CMD -f infrastructure/orchestration/docker-compose.yml --profile ml up -d ray-head ray-worker-1 rl-training-worker

# 3. Health Check (Polling Dashboard API)
echo "⏳ Waiting for Ray Dashboard to be ready..."
RETRIES=30
INTERVAL=5
ENDPOINT="http://localhost:8265/api/jobs/"

until curl -s -o /dev/null -w "%{http_code}" "$ENDPOINT" | grep -q "200" || [ $RETRIES -eq 0 ]; do
    echo "   - Dashboard not ready yet (Retries left: $RETRIES). Sleeping ${INTERVAL}s..."
    sleep $INTERVAL
    ((RETRIES--))
done

if [ $RETRIES -eq 0 ]; then
    echo " Fatal: Ray Head failed to reach healthy state within timeout."
    $COMPOSE_CMD -f infrastructure/orchestration/docker-compose.yml --profile ml logs ray-head
    exit 1
fi

echo " Ray Cluster is Healthy."
echo " Revamping and Optimizing Engine..."
# Note: We run this in the ray-head container to ensure it has access to the cluster and 'ray' package
$COMPOSE_CMD -f infrastructure/orchestration/docker-compose.yml --profile ml exec -T ray-head python scripts/engine_revamp_ray_god_mode.py
