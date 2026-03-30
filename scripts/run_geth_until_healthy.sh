#!/bin/bash
# scripts/run_geth_until_healthy.sh
set -euo pipefail

# 1. Setup Environment
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SOURCE_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f "scripts/utils_env.sh" ]; then
    source scripts/utils_env.sh
else
    echo "[ERROR] scripts/utils_env.sh not found."
    exit 1
fi

detect_container_engine

COMPOSE_FILE="infrastructure/orchestration/docker-compose.yml"

# 2. Start Geth
echo "🚀 Starting Geth with blockchain profile using $CONTAINER_ENGINE..."
$COMPOSE_ENGINE -f "$COMPOSE_FILE" --profile blockchain up -d geth

# 3. Wait for Healthy (Reported by Engine)
echo "⏳ Waiting for $CONTAINER_ENGINE to report Geth as healthy..."
MAX_RETRIES=60
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    # Get the container ID for the geth service
    CONTAINER_ID=$($COMPOSE_ENGINE -f "$COMPOSE_FILE" --profile blockchain ps -q geth | head -n 1)
    
    if [ -n "$CONTAINER_ID" ]; then
        # Inspect the container for its health status
        STATUS=$($CONTAINER_ENGINE inspect -f '{{.State.Health.Status}}' "$CONTAINER_ID" 2>/dev/null || echo "starting")
        echo "Current Engine Status: $STATUS"
        
        if [ "$STATUS" == "healthy" ]; then
            echo "✅ Engine reports Geth is healthy!"
            exit 0
        fi
    else
        echo "Waiting for container to be created..."
    fi
    
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

echo "❌ Fatal: Geth failed to become healthy within timeout."
exit 1
