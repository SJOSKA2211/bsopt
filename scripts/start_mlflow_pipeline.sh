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

# Execute mlflow run in the background inside the mlops-worker container
$COMPOSE_CMD exec -d mlops-worker mlflow run . -e "$PIPELINE_ENTRY_POINT" --experiment-name "$STUDY_NAME" --no-conda

echo "=========================================================="
echo "Pipeline $PIPELINE_ENTRY_POINT initiated successfully."
echo "Track runs via the MLflow UI at http://localhost:5000"
echo "To view raw execution logs, run:"
echo "  docker compose logs -f mlops-worker"
echo "=========================================================="
