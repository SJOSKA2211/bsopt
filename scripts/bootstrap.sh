#!/usr/bin/env bash
# scripts/bootstrap.sh - Institutional Zero-Touch Bootstrapper
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 1. Detect Container Engine
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
    COMPOSE_CMD="podman-compose"
elif command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
    if $CONTAINER_CMD compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
else
    echo "❌ Error: Neither docker nor podman is installed."
    exit 1
fi

echo "🚀 Using container orchestrator: $COMPOSE_CMD"

# 2. Institutional PKI & Secret Generation
echo "🔐 Initializing Security Substrate..."
bash scripts/setup_pki.sh

ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    echo "📝 Generating Institutional Secrets..."
    
    # High-Entropy Database Secrets
    DB_USER="equaflow_admin"
    DB_PASS=$(openssl rand -hex 32)
    DB_NAME="bsopt"
    
    # Redis & RabbitMQ Secrets
    REDIS_PASS=$(openssl rand -hex 32)
    RABBITMQ_PASS=$(openssl rand -hex 32)
    
    # MINIO Secrets
    MINIO_ROOT_PASS=$(openssl rand -hex 32)

    cat <<EOF > "$ENV_FILE"
# Institutional Environment Configuration
ENVIRONMENT=production
DEBUG=false

# Database Security
POSTGRES_USER=$DB_USER
POSTGRES_PASSWORD=$DB_PASS
POSTGRES_DB=$DB_NAME
DATABASE_URL=postgresql://$DB_USER:$DB_PASS@pgbouncer:5432/$DB_NAME

# Cache Security
REDIS_PASSWORD=$REDIS_PASS
REDIS_URL=redis://:$REDIS_PASS@redis:6379/0

# Messaging Security
RABBITMQ_USER=bsopt_admin
RABBITMQ_PASSWORD=$RABBITMQ_PASS

# Object Storage Security
MINIO_ROOT_USER=minio_admin
MINIO_ROOT_PASSWORD=$MINIO_ROOT_PASS

# "Zero-Mock" API Keys (USER MUST PROVIDE THESE)
ALPHA_VANTAGE_API_KEY=
POLYGON_API_KEY=
IBM_QUANTUM_TOKEN=

# Testing & CI
BSOPT_ALLOW_WEAK_SECRETS=false
EOF
    echo "✅ Secrets generated and secured in $ENV_FILE"
else
    echo "ℹ️ $ENV_FILE already exists. Preserving existing secure state."
fi

# 3. Microservices Lifecycle Management
echo "📦 Spinning up Core Institutional Stack..."
$COMPOSE_CMD -f infrastructure/orchestration/docker-compose.yml up -d postgres pgbouncer redis rabbitmq minio

# 4. Robust Healthcheck Polling
check_health() {
    local service=$1
    local retries=30
    echo "⏳ Waiting for $service to reach healthy state..."
    until [ "$($COMPOSE_CMD -f infrastructure/orchestration/docker-compose.yml ps --format json | grep -E "\"Service\":\"$service\"" | grep -E "\"Health\":\"healthy\"")" ] || [ $retries -eq 0 ]; do
        sleep 2
        ((retries--))
    done
    if [ $retries -eq 0 ]; then
        echo "❌ Fatal: $service failed to reach readiness."
        $COMPOSE_CMD -f infrastructure/orchestration/docker-compose.yml logs "$service"
        exit 1
    fi
    echo "✅ $service is Healthy."
}

check_health "postgres"
check_health "pgbouncer"
check_health "redis"
check_health "rabbitmq"
check_health "minio"

echo "🏁 Phase 0 Bootstrapping Complete. Institutional Stack is Operational."
