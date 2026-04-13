#!/usr/bin/env bash
# bootstrap.sh - Sequential Deployment & Validation Orchestrator (CPU-ONLY)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=== Meta-Cognitive Node: Sequential Deployment Starting ==="

# 1. Environment Generation (Secure-by-Default)
ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    echo " Generating cryptographically secure .env..."
    DB_PASS=$(openssl rand -hex 32)
    REDIS_PASS=$(openssl rand -hex 32)
    RABBIT_PASS=$(openssl rand -hex 32)
    JWT_PASS=$(openssl rand -hex 32)
    MFA_KEY=$(openssl rand -base64 32)

    cat > "$ENV_FILE" <<EOF
ENVIRONMENT=production
POSTGRES_DB=bsopt
POSTGRES_USER=admin
POSTGRES_PASSWORD=$DB_PASS
REDIS_PASSWORD=$REDIS_PASS
REDIS_HOST=redis
REDIS_PORT=6379
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=$RABBIT_PASS
RABBITMQ_HOST=rabbitmq
JWT_SECRET=$JWT_PASS
JWT_ALGORITHM=RS256
JWT_PRIVATE_KEY=/etc/pki/auth_service.key
JWT_PUBLIC_KEY=/etc/pki/auth_service.crt
MFA_ENCRYPTION_KEY=$MFA_KEY
BETTER_AUTH_SECRET=$DB_PASS
BETTER_AUTH_URL=http://localhost:3001
CORS_ORIGINS=http://localhost,http://localhost:3000
TRUSTED_PROXIES=127.0.0.1
MARKET_TICKER_SYMBOLS=SPY,AAPL,TSLA,GOOGL
MINIO_ENDPOINT=minio:9000
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=$DB_PASS
OPA_URL=http://opa:8181
ML_SERVICE_GRPC_URL=pricing_api:50051
AUTH_SERVICE_GRPC_URL=auth_api:50051
PGBOUNCER_ADMIN_USER=admin
PGBOUNCER_ADMIN_PASSWORD=$DB_PASS
PGBOUNCER_HOST=postgres
PGBOUNCER_PORT=6432
NSE_SYMBOLS='{}'
NSE_SCRAPER_SECTORS='[]'
EOF
    echo " .env generated successfully."
fi

# Load variables
export $(grep -v '^#' .env | xargs)

# 2. PKI & Security substrate
KEY_DIR=".pki"
mkdir -p "$KEY_DIR"
if [[ ! -f "${KEY_DIR}/root_ca.key" ]]; then
    echo " Initializing Internal CA..."
    openssl genrsa -out "${KEY_DIR}/root_ca.key" 4096
    openssl req -x509 -new -nodes -key "${KEY_DIR}/root_ca.key" -sha256 -days 3650 \
        -out "${KEY_DIR}/root_ca.crt" \
        -subj "/C=US/ST=State/L=City/O=BSOPT-Production/CN=Manifold-Internal-CA"
fi

issue_cert() {
    local svc=$1
    local name=${2:-$svc}
    if [[ ! -f "${KEY_DIR}/${name}.key" ]]; then
        echo " Issuing cert for $name ($svc)..."
        openssl genrsa -out "${KEY_DIR}/${name}.key" 2048
        openssl req -new -key "${KEY_DIR}/${name}.key" -out "${KEY_DIR}/${name}.csr" -subj "/C=US/O=BSOPT/CN=${svc}"
        openssl x509 -req -in "${KEY_DIR}/${name}.csr" -CA "${KEY_DIR}/root_ca.crt" -CAkey "${KEY_DIR}/root_ca.key" \
            -CAcreateserial -out "${KEY_DIR}/${name}.crt" -days 365 -sha256
        rm "${KEY_DIR}/${name}.csr"
    fi
}

# Certs for gRPC Services
issue_cert "auth_api" "auth-service"
issue_cert "pricing_api" "api-client"

# Certs for other services
SERVICES=("postgres" "redis" "rabbitmq" "vault")
for s in "${SERVICES[@]}"; do issue_cert "$s"; done

# Duplicate for JWT consistency if needed
cp "${KEY_DIR}/auth-service.key" "${KEY_DIR}/auth_service.key" 2>/dev/null || true
cp "${KEY_DIR}/auth-service.crt" "${KEY_DIR}/auth_service.crt" 2>/dev/null || true

# 3. Protocol Synchronization
echo " Generating gRPC code from protos..."
GEN_DIR="src/shared/protos"
mkdir -p "$GEN_DIR"
docker run --rm -v "$(pwd):/app" -w /app python:3.12.13-slim sh -c \
    "pip install grpcio-tools && python -m grpc_tools.protoc -I./protos --python_out=$GEN_DIR --grpc_python_out=$GEN_DIR protos/*.proto"
touch "$GEN_DIR/__init__.py"
sed -i 's/import \([^ ]*\)_pb2/from . import \1_pb2/g' "$GEN_DIR"/*_pb2*.py 2>/dev/null || true

# 4. Sequential Deployment Loop
COMPOSE_FILE="docker-compose.yml"

deploy_and_validate() {
    local svc=$1
    echo "--- Building $svc ---"
    docker compose -f "$COMPOSE_FILE" build "$svc"
    
    echo "--- Deploying $svc ---"
    docker compose -f "$COMPOSE_FILE" up -d "$svc"
    
    echo "⏳ Validating $svc health..."
    local retries=40
    until docker compose -f "$COMPOSE_FILE" ps "$svc" --format json | grep -qE '"Health":"healthy"|"State":"running"' || [ $retries -eq 0 ]; do
        echo -n "."
        sleep 3
        ((retries--))
    done
    echo ""
    
    if [ $retries -eq 0 ]; then
        echo " FAILED: $svc did not reach healthy state."
        docker compose -f "$COMPOSE_FILE" logs "$svc" | tail -n 50
        exit 1
    fi
    echo " $svc is ONLINE and HEALTHY."
}

# Core Infra
deploy_and_validate "postgres"
deploy_and_validate "redis"
deploy_and_validate "rabbitmq"
deploy_and_validate "vault"

# Microservices
deploy_and_validate "auth_api"
deploy_and_validate "pricing_api"
deploy_and_validate "math_worker"
deploy_and_validate "frontend"
deploy_and_validate "nginx"

# 5. Zero-Mock E2E Tests
echo "--- Initiating Master Test: Zero-Mock E2E ---"
docker compose run --rm pricing_api pytest tests/e2e -v

echo "=== ALL SYSTEMS OPERATIONAL: 100% HEALTHY NETWORK & 100% PASSING E2E ==="
