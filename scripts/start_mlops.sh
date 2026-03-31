#!/usr/bin/env bash
# scripts/start_mlops.sh - MLOps Stack Orchestrator (Hardened)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 1. Load Environment Utilities
if [ -f "scripts/utils_env.sh" ]; then
    source scripts/utils_env.sh
else
    echo "❌ Error: scripts/utils_env.sh not found."
    exit 1
fi

# Detect Engine
detect_container_engine
load_decrypted_secrets

# 2. Configuration
COMPOSE_FILE="infrastructure/orchestration/docker-compose.yml"
# We target the 'ml' profile services
ML_SERVICES=("mlflow" "ray-head" "mlops-worker" "ray-worker-1" "rl-training-worker")

echo "=============================================================================="
echo "🚀 Launching MLOps Infrastructure Cluster (Profile: ml)"
echo "=============================================================================="

# 3. Pull/Build Check
echo "🛠️ Ensuring MLOps images are ready..."
$COMPOSE_ENGINE -f "$COMPOSE_FILE" --profile ml build mlops-worker

# 4. Deployment
echo "🏗️ Starting services: ${ML_SERVICES[*]}..."
# Run with the ml profile enabled
$COMPOSE_ENGINE -f "$COMPOSE_FILE" --profile ml up -d "${ML_SERVICES[@]}"

# 5. Health Handshake
echo "⏳ Waiting for MLOps ecosystem to reach operational stability..."

check_health() {
    local service=$1
    local retries=40
    local interval=5
    
    echo -n "🔍 Verifying $service... "
    while [ $retries -gt 0 ]; do
        # Get container ID
        local cid=$($COMPOSE_ENGINE -f "$COMPOSE_FILE" ps -q "$service" 2>/dev/null | head -n 1)
        if [ -n "$cid" ]; then
            local status=$($CONTAINER_ENGINE inspect -f '{{.State.Health.Status}}' "$cid" 2>/dev/null || echo "starting")
            if [ "$status" == "healthy" ]; then
                echo "✅ Healthy"
                return 0
            fi
            echo -n "($status)..."
        else
            echo -n "(not found)..."
        fi
        sleep $interval
        ((retries--))
    done
    echo "❌ FAILED"
    return 1
}

for service in "${ML_SERVICES[@]}"; do
    if ! check_health "$service"; then
        echo "❌ Fatal: MLOps stack initialization failed at $service"
        $COMPOSE_ENGINE -f "$COMPOSE_FILE" logs "$service" | tail -n 20
        exit 1
    fi
done

echo "=============================================================================="
echo "🏁 MLOps Infrastructure is ONLINE and Healthy"
echo "Track experiments: http://localhost:5000"
echo "Ray Dashboard:     http://localhost:8265"
echo "=============================================================================="
