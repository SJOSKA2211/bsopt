#!/bin/bash
# scripts/start_stack_v2.sh - Orchestrate the full stack with health checks
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source scripts/utils_env.sh
detect_container_engine

echo " Launching BSOPT Production Ecosystem..."

# 1. Base PKI
bash scripts/setup_pki.sh

# 2. Core Infrastructure (Sequential health check)
echo " Starting Core Infrastructure..."
$COMPOSE_ENGINE -f infrastructure/orchestration/docker-compose.yml up -d postgres pgbouncer redis rabbitmq minio otel-collector

check_docker_health() {
    local service=$1
    local retries=30
    echo "⏳ Waiting for $service to be healthy..."
    until [ "$($COMPOSE_ENGINE -f infrastructure/orchestration/docker-compose.yml ps --format json | grep -E "\"Service\":\"$service\"" | grep -E "\"Health\":\"healthy\"")" ] || [ $retries -eq 0 ]; do
        sleep 2
        ((retries--))
    done
    if [ $retries -eq 0 ]; then
        echo " Fatal: $service failed to reach healthy state."
        $COMPOSE_ENGINE -f infrastructure/orchestration/docker-compose.yml logs "$service" | tail -n 20
        exit 1
    fi
    echo " $service is Stable."
}

# Core components with built-in healthchecks
for s in postgres pgbouncer redis rabbitmq minio; do
    check_docker_health "$s"
done

# 3. Ray Head node
echo "🧠 Starting Ray Cluster..."
bash scripts/run_ray_head.sh

# 4. Domain Services (Python based)
# Inject BSOPT_ALLOW_WEAK_SECRETS=True to bypass 32-char restriction in this environment
echo "️ Starting Domain Services..."
export BSOPT_ALLOW_WEAK_SECRETS=True
$COMPOSE_ENGINE -f infrastructure/orchestration/docker-compose.yml up -d auth-service api ml-inference neural-pricing mlflow

# 5. Final Readiness Audit
echo "🩺 Executing Final Readiness Audit..."
MAX_RETRIES=15
RETRY_COUNT=0
export PYTHONPATH=$(pwd):$(pwd)/src
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if python3 scripts/verify_readiness.py; then
        echo " SYSTEM GREEN - ALL SYSTEMS OPERATIONAL."
        exit 0
    fi
    echo "🟠 Waiting for service mesh stabilization... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 10
    ((RETRY_COUNT++))
done

echo " Fatal: Stack failed readiness audit."
exit 1
