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
    echo " Error: Docker is required but not installed."
    exit 1
fi

echo " Launching Production Infrastructure Core..."

# 2. Deployment
$COMPOSE_CMD -f infrastructure/orchestration/docker-compose.yml up -d postgres pgbouncer redis rabbitmq minio

# 3. Micro-Service Verification
check_health() {
    local service=$1
    local retries=30
    echo "⏳ Verifying $service health..."
    until [ "$($COMPOSE_CMD -f infrastructure/orchestration/docker-compose.yml ps --format json | grep -E "\"Service\":\"$service\"" | grep -E "\"Health\":\"healthy\"")" ] || [ $retries -eq 0 ]; do
        sleep 2
        ((retries--))
    done
    if [ $retries -eq 0 ]; then
        echo " Fatal: $service failed to reach stable state."
        exit 1
    fi
    echo " $service is Stable."
}

check_health "postgres"
check_health "pgbouncer"
check_health "redis"
check_health "rabbitmq"
check_health "minio"

echo " Production Core is Online and Guarded."
