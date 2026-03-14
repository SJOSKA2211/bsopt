#!/usr/bin/env bash
# Initialize and start MLflow training pipeline via Docker
set -e

# Usage: ./scripts/start_mlflow_pipeline.sh [entry_point] [experiment_name] [additional_params...]
# Example: ./scripts/start_mlflow_pipeline.sh train_regressor tsla_v1 -P ticker=TSLA -P n_trials=50

PIPELINE_ENTRY_POINT=${1:-train_rl}
STUDY_NAME=${2:-rl_v1}
shift 2 || true
EXTRA_PARAMS=$@

# Detection logic for docker compose
if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
elif [ -x "./docker-compose" ]; then
    COMPOSE_CMD="./docker-compose"
else
    echo "❌ Error: docker compose not found. Please install it."
    exit 1
fi

# Ensure we are in the project root to find the binary
cd "$(dirname "$0")/.."

# Default project name for consistency if not set
export COMPOSE_PROJECT_NAME=${COMPOSE_PROJECT_NAME:-bsopt_revamp}

echo "Checking if mlops-worker is running..."
if ! $COMPOSE_CMD ps | grep -q "mlops-worker"; then
    echo "Starting mlops-worker (and dependencies)..."
    $COMPOSE_CMD --profile ml up -d mlops-worker
    echo "Waiting for MLflow server to be healthy..."
    sleep 5
fi

echo "Starting MLflow Pipeline: $PIPELINE_ENTRY_POINT with experiment: $STUDY_NAME"
echo "Extra parameters: $EXTRA_PARAMS"

# Check if entry point accepts study_name and pass it if not already in EXTRA_PARAMS
if [[ "$EXTRA_PARAMS" != *"study_name"* ]]; then
    EXTRA_PARAMS="-P study_name=$STUDY_NAME $EXTRA_PARAMS"
fi

# Use 'exec -d' to start the task inside the existing mlops-worker container
$COMPOSE_CMD exec -d mlops-worker mlflow run . \
    -e "$PIPELINE_ENTRY_POINT" \
    --experiment-name "$STUDY_NAME" \
    --env-manager local \
    $EXTRA_PARAMS

echo "=========================================================="
echo "Pipeline $PIPELINE_ENTRY_POINT initiated inside mlops-worker."
echo "Track results via the MLflow UI at http://localhost:5000"
echo "To view logs, run:"
echo "  $COMPOSE_CMD logs -f mlops-worker"
echo "=========================================================="
