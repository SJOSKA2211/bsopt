#!/bin/bash
# scripts/start_geth_and_wait.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

COMPOSE_CMD="docker compose"
DOCKER_CMD="docker"

echo "🚀 Starting Geth with blockchain profile..."
$COMPOSE_CMD -f infrastructure/orchestration/docker-compose.yml --profile blockchain up -d geth

echo "⏳ Waiting for Geth (8545) to become healthy..."
RETRIES=60
until curl -s -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"net_version","params":[],"id":67}' \
  http://localhost:8545 | grep -q result || [ $RETRIES -eq 0 ]; do
    echo "Waiting for JSON-RPC at http://localhost:8545 ($RETRIES retries left)..."
    sleep 5
    ((RETRIES--))
done

if [ $RETRIES -eq 0 ]; then
    echo "❌ Fatal: Geth failed to reach stable state within timeout."
    exit 1
fi

echo "✅ Geth is Online and Healthy!"
# Report health to the engine? Maybe it means outputting a specific string or using a tool.
# I'll just output the final status.
