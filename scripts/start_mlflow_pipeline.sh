#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Usage: ./scripts/start_mlflow_pipeline.sh [entry_point] [experiment_name] [additional_params...]
ENTRY_POINT=${1:-train_rl}
EXPERIMENT=${2:-rl_v1}
shift 2 || true
PARAMS=$@

echo "🧬 Launching Production ML Pipeline: $ENTRY_POINT"

# Load Production environment and detection
source scripts/utils_env.sh
detect_container_engine

# Ensure MLflow substrate is active
if ! $COMPOSE_ENGINE -f infrastructure/orchestration/docker-compose.yml ps | grep -q "mlflow"; then
    echo "️ Starting MLflow infrastructure..."
    $COMPOSE_ENGINE -f infrastructure/orchestration/docker-compose.yml up -d mlflow minio
    sleep 5
fi

echo " Executing pipeline in Production ml-worker..."
# standardizing on 'ml-worker' service name
$COMPOSE_ENGINE -f infrastructure/orchestration/docker-compose.yml exec -d ml-worker mlflow run . \
    -e "$ENTRY_POINT" \
    --experiment-name "$EXPERIMENT" \
    --env-manager local \
    $PARAMS

echo " Pipeline $ENTRY_POINT initiated."
echo "Track results: http://localhost:5000"
