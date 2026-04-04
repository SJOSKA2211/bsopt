#!/bin/bash
set -euo pipefail
source scripts/utils_env.sh
load_decrypted_secrets
detect_container_engine

COMPOSE_FILE="infrastructure/orchestration/docker-compose.yml"
SERVICE_NAME="rabbitmq"

echo "Starting $SERVICE_NAME..."
$COMPOSE_ENGINE -f "$COMPOSE_FILE" up -d "$SERVICE_NAME"

echo "Waiting for $SERVICE_NAME to be healthy (engine reporting)..."
MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    CONTAINER_ID=$($COMPOSE_ENGINE -f "$COMPOSE_FILE" ps -q "$SERVICE_NAME")
    if [ -n "$CONTAINER_ID" ]; then
        STATUS=$(docker inspect -f '{{.State.Health.Status}}' "$CONTAINER_ID" 2>/dev/null || echo "starting")
        echo "Current status: $STATUS"
        if [ "$STATUS" == "healthy" ]; then
            echo "$SERVICE_NAME is healthy!"
            exit 0
        fi
    fi
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

echo "$SERVICE_NAME failed to become healthy in time."
$COMPOSE_ENGINE -f "$COMPOSE_FILE" logs "$SERVICE_NAME"
exit 1
