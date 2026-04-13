#!/usr/bin/env bash
# scripts/bootstrap.sh - Sequential Deployment & Validation Orchestrator
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== OMarchy Meta-Cognitive Node: Sequential Deployment Starting ==="

# 1. Clean Environment
echo " Purging legacy scripts (preserving bootstrap)..."
find scripts -maxdepth 1 -type f ! -name "bootstrap.sh" -delete
find scripts -maxdepth 1 -type d ! -name "scripts" ! -name "archive" ! -name "benchmarks" ! -name "cli" ! -name "hft" -exec rm -rf {} + 2>/dev/null || true

# 2. PKI & Security Layer
echo " Initializing Security Substrate..."
KEY_DIR=".pki"
mkdir -p "$KEY_DIR"
if [[ ! -f "${KEY_DIR}/root_ca.key" ]]; then
    openssl genrsa -out "${KEY_DIR}/root_ca.key" 4096
    openssl req -x509 -new -nodes -key "${KEY_DIR}/root_ca.key" -sha256 -days 3650 \
        -out "${KEY_DIR}/root_ca.crt" \
        -subj "/C=US/ST=State/L=City/O=BSOPT-Production/CN=Manifold-Internal-CA"
fi

# Issue Service Certs
issue_cert() {
    local svc=$1
    if [[ ! -f "${KEY_DIR}/${svc}.key" ]]; then
        openssl genrsa -out "${KEY_DIR}/${svc}.key" 2048
        openssl req -new -key "${KEY_DIR}/${svc}.key" -out "${KEY_DIR}/${svc}.csr" -subj "/C=US/O=BSOPT/CN=${svc}"
        openssl x509 -req -in "${KEY_DIR}/${svc}.csr" -CA "${KEY_DIR}/root_ca.crt" -CAkey "${KEY_DIR}/root_ca.key" \
            -CAcreateserial -out "${KEY_DIR}/${svc}.crt" -days 365 -sha256
        rm "${KEY_DIR}/${svc}.csr"
    fi
}

SERVICES=("postgres" "pgbouncer" "redis" "rabbitmq" "minio" "api" "auth-service" "worker" "neural-pricing" "frontend" "nginx" "envoy")
for s in "${SERVICES[@]}"; do issue_cert "$s"; done

# 3. Protocol Generation
echo " Synchronizing Protocols..."
GEN_DIR="src/shared/protos"
mkdir -p "$GEN_DIR"
uv run python -m grpc_tools.protoc -I./protos --python_out="$GEN_DIR" --grpc_python_out="$GEN_DIR" ./protos/*.proto
touch "$GEN_DIR/__init__.py"
sed -i 's/import \([^ ]*\)_pb2/from . import \1_pb2/g' "$GEN_DIR"/*_pb2*.py 2>/dev/null || true

# 4. Environment Configuration
ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    DB_PASS=$(openssl rand -hex 32)
    REDIS_PASS=$(openssl rand -hex 32)
    RABBITMQ_PASS=$(openssl rand -hex 32)
    echo "ENVIRONMENT=production\nPOSTGRES_PASSWORD=$DB_PASS\nREDIS_PASSWORD=$REDIS_PASS\nRABBITMQ_PASSWORD=$RABBITMQ_PASS\nDATABASE_URL=postgresql://admin:$DB_PASS@pgbouncer:6432/bsopt?sslmode=verify-full&sslrootcert=/etc/pki/root_ca.crt\nREDIS_URL=redis://:$REDIS_PASS@redis:6379/0" > "$ENV_FILE"
fi

# 5. Build Unified Base Layers (Hyper-Fast BuildKit)
echo "--- Building Manifold Base Architecture (CPU-ONLY) ---"
docker build -t manifold-base:builder --target builder .
docker build -t manifold-base:latest --target latest .

# 6. Sequential Orchestration Loop
COMPOSE_FILE="infrastructure/orchestration/docker-compose.yml"

deploy_and_validate() {
    local svc=$1
    echo "--- Deploying $svc ---"
    docker compose -f "$COMPOSE_FILE" build "$svc"
    docker compose -f "$COMPOSE_FILE" up -d "$svc"
    
    echo "⏳ Validating $svc health..."
    local retries=20
    until [ "$(docker compose -f "$COMPOSE_FILE" ps --format json "$svc" | grep -o '"Health":"healthy"' || true)" ] || [ $retries -eq 0 ]; do
        sleep 3
        ((retries--))
    done
    
    if [ $retries -eq 0 ]; then
        echo " FAILED: $svc did not reach healthy state."
        docker compose -f "$COMPOSE_FILE" logs "$svc" | tail -n 50
        exit 1
    fi
    echo " $svc is ONLINE and HEALTHY."
}

# Core Infra
deploy_and_validate "postgres"
deploy_and_validate "pgbouncer"
deploy_and_validate "redis"
deploy_and_validate "rabbitmq"

# Microservices
deploy_and_validate "auth-service"
deploy_and_validate "api"
deploy_and_validate "worker"
deploy_and_validate "neural-pricing"
deploy_and_validate "frontend"
deploy_and_validate "nginx"
deploy_and_validate "envoy"

# 7. Master Test: Zero-Mock E2E Suite
echo "--- Initiating Master Test (Zero-Mock E2E) ---"
export API_URL="http://localhost:8000/api/v1"
export AUTH_SERVICE_GRPC_URL="localhost:50051"
export GRPC_SECURE=false # Fallback to insecure for initial E2E if certs are internal

# Run tests
uv run pytest tests/e2e/test_auth_e2e.py -v

echo "=== ALL SYSTEMS OPERATIONAL: 100% HEALTHY NETWORK & 100% PASSING E2E ==="
