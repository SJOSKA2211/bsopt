#!/usr/bin/env bash
# Initialize and start MLflow training pipeline via Docker
set -e

PIPELINE_ENTRY_POINT=${1:-train_rl}
STUDY_NAME=${2:-rl_trading_core_v1}
COMPOSE_CMD="./docker-compose"

# Ensure we are in the project root to find the binary
cd "$(dirname "$0")/.."

echo "Checking if mlops-worker is running..."
if ! $COMPOSE_CMD ps | grep -q "mlops-worker"; then
    echo "Starting mlops-worker (and dependencies)..."
    $COMPOSE_CMD --profile ml up -d mlops-worker
    echo "Waiting for MLflow server to be healthy..."
    sleep 5
fi

echo "Starting MLflow Pipeline: $PIPELINE_ENTRY_POINT with study name: $STUDY_NAME"

# Use 'run' to start a fresh container for the training job, then auto-remove it
$COMPOSE_CMD run --rm mlops-worker mlflow run . -e "$PIPELINE_ENTRY_POINT" --experiment-name "$STUDY_NAME" --env-manager local

echo "=========================================================="
echo "Pipeline $PIPELINE_ENTRY_POINT completed/terminated."
echo "Track results via the MLflow UI at http://localhost:5000"
echo "=========================================================="
