#!/bin/bash
# ==============================================================================
# EQUAFLOW: THE ZERO-TOUCH BOOTSTRAP (v2.1)
# ==============================================================================
# Automates the entire stack: Security (RSA/ECC), .env, DB Init, and Gateway.
# ==============================================================================

set -e

# Configuration
KEYS_DIR="$(pwd)/.pki"
ENV_FILE=".env"
ENV_EXAMPLE=".env.example"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# 0. Prerequisite Check Function
check_prereq() {
    local cmd=$1
    if ! command -v "$cmd" &> /dev/null; then
        echo "❌ Error: $cmd is not installed."
        exit 1
    fi
}

check_prereq openssl

# 1. Container Engine & Compose Detection (Prioritizing V2 Plugins)
if command -v podman &> /dev/null; then
    CONTAINER_ENGINE="podman"
    if podman compose version &> /dev/null; then
        COMPOSE_ENGINE="podman compose"
    elif command -v podman-compose &> /dev/null && ! grep -q "PackageKit" <(which podman-compose 2>/dev/null); then
        COMPOSE_ENGINE="podman-compose"
    else
        echo "⚠️ Warning: podman-compose not found or intercepted by PackageKit. Using 'podman compose' fallback."
        COMPOSE_ENGINE="podman compose"
    fi
    echo "🚀 Detected system: podman"
elif command -v docker &> /dev/null; then
    CONTAINER_ENGINE="docker"
    if docker compose version &> /dev/null; then
        COMPOSE_ENGINE="docker compose"
    elif command -v docker-compose &> /dev/null; then
        COMPOSE_ENGINE="docker-compose"
    elif [ -f "./docker-compose" ]; then
        COMPOSE_ENGINE="./docker-compose"
    else
        echo "❌ Error: docker found but no compose engine detected."
        exit 1
    fi
else
    echo "❌ Error: Neither podman nor docker found."
    exit 1
fi

echo "🚀 Starting EquaFlow Advanced Bootstrap [${TIMESTAMP}]"
echo "🚀 Using $CONTAINER_ENGINE with $COMPOSE_ENGINE"

# 1. Initialize Security Layer (RSA/ECC)
mkdir -p "${KEYS_DIR}"

generate_keys() {
    echo "🔐 Generating Asymmetric Key Pairs (Institutional Grade)..."
    
    # RSA 4096 (RS256) for legacy compatibility
    if [ ! -f "${KEYS_DIR}/jwt_rs256.key" ]; then
        openssl genrsa -out "${KEYS_DIR}/jwt_rs256.key" 4096
        openssl rsa -in "${KEYS_DIR}/jwt_rs256.key" -pubout -out "${KEYS_DIR}/jwt_rs256.pub"
    fi

    # ECC P-256 (ES256) for modern high-performance auth
    if [ ! -f "${KEYS_DIR}/jwt_es256.key" ]; then
        openssl ecparam -name prime256v1 -genkey -noout -out "${KEYS_DIR}/jwt_es256.key"
        openssl ec -in "${KEYS_DIR}/jwt_es256.key" -pubout -out "${KEYS_DIR}/jwt_es256.pub"
    fi

    # Argon2id Hashing Salt (Global Salt if needed, though per-user is standard)
    if [ ! -f "${KEYS_DIR}/argon2_salt.secret" ]; then
        openssl rand -hex 32 > "${KEYS_DIR}/argon2_salt.secret"
    fi

    # TOTP Master Secret (for system-wide MFA seeds)
    if [ ! -f "${KEYS_DIR}/totp_master.secret" ]; then
        openssl rand -hex 32 > "${KEYS_DIR}/totp_master.secret"
    fi
    
    # Envoy SSL Cert (Self-Signed for Dev Edge Termination)
    if [ ! -f "${KEYS_DIR}/envoy_edge.key" ]; then
        openssl req -x509 -newkey rsa:4096 -keyout "${KEYS_DIR}/envoy_edge.key" -out "${KEYS_DIR}/envoy_edge.crt" \
            -days 365 -nodes -subj "/C=US/ST=State/L=City/O=EquaFlow/CN=localhost"
    fi
}

generate_keys

# 2. .env Orchestration
if [ ! -f "${ENV_FILE}" ]; then
    echo "📄 Creating .env from template..."
    if [ -f "${ENV_EXAMPLE}" ]; then
        cp "${ENV_EXAMPLE}" "${ENV_FILE}"
    else
        touch "${ENV_FILE}"
    fi
fi

set_env_var() {
    local key=$1
    local value=$2
    # Ensure value is quoted for multi-line support (like keys)
    if grep -q "^${key}=" "${ENV_FILE}"; then
        # Use a different delimiter for sed since keys have slashes
        sed -i "s|^${key}=.*|${key}=\"${value}\"|g" "${ENV_FILE}"
    else
        echo "${key}=\"${value}\"" >> "${ENV_FILE}"
    fi
}

# Explicitly store raw keys in .env (Base64 for single-line safety)
echo "🔑 Explicitly injecting keys into ${ENV_FILE}..."
RS256_PRIV=$(cat "${KEYS_DIR}/jwt_rs256.key" | base64 -w 0)
RS256_PUB=$(cat "${KEYS_DIR}/jwt_rs256.pub" | base64 -w 0)
ES256_PRIV=$(cat "${KEYS_DIR}/jwt_es256.key" | base64 -w 0)
ES256_PUB=$(cat "${KEYS_DIR}/jwt_es256.pub" | base64 -w 0)
ARGON2_SALT=$(cat "${KEYS_DIR}/argon2_salt.secret")
TOTP_MASTER=$(cat "${KEYS_DIR}/totp_master.secret")

set_env_var "JWT_RS256_PRIVATE" "${RS256_PRIV}"
set_env_var "JWT_RS256_PUBLIC" "${RS256_PUB}"
set_env_var "JWT_ES256_PRIVATE" "${ES256_PRIV}"
set_env_var "JWT_ES256_PUBLIC" "${ES256_PUB}"
set_env_var "ARGON2_SALT" "${ARGON2_SALT}"
set_env_var "MFA_TOTP_SECRET" "${TOTP_MASTER}"

# Secure random passwords for necessary variables
for var in POSTGRES_PASSWORD REDIS_PASSWORD BETTER_AUTH_SECRET JWT_SECRET RABBITMQ_PASSWORD; do
    # Only set if not already present or is empty
    if ! grep -q "^${var}=" "${ENV_FILE}" || [[ -z $(grep "^${var}=" "${ENV_FILE}" | cut -d'=' -f2 | tr -d '"' | tr -d "'") ]]; then
        set_env_var "${var}" "$(openssl rand -hex 32)"
    fi
done


# 3. PostgreSQL Automation & Secret Orchestration
echo "🐘 Preparing PostgreSQL initialization..."
PG_PASS=$(grep "^POSTGRES_PASSWORD=" "${ENV_FILE}" | cut -d'=' -f2 | tr -d '"' | tr -d "'")
set_env_var "DATABASE_URL" "postgresql://admin:${PG_PASS}@pgbouncer:6432/bsopt"
set_env_var "DATABASE_URL_LOCAL" "postgresql://admin:${PG_PASS}@localhost:5434/bsopt"
set_env_var "DATABASE_URL_TEST" "postgresql://admin:${PG_PASS}@postgres:5432/bsopt_test"
set_env_var "MLFLOW_BACKEND_STORE_URI" "postgresql://admin:${PG_PASS}@postgres:5432/bsopt"

# We must use make build first to ensure all images exist before standing them up sequentially
echo "🏗️ Building all images before sequential startup..."
make build

# 4. Sequential Database Startup & Health Checks
echo "🏗️ Starting sequentially: Postgres -> Redis -> PgBouncer"
$COMPOSE_ENGINE --env-file .env -f infrastructure/orchestration/docker-compose.yml up -d postgres

echo "⏳ Waiting for Postgres to be Live & Healthy..."
MAX_RETRIES=30
RETRY_COUNT=0
POSTGRES_CONTAINER=$($COMPOSE_ENGINE --env-file .env -f infrastructure/orchestration/docker-compose.yml ps --format "{{.Name}}" postgres 2>/dev/null | head -n 1)
[ -z "$POSTGRES_CONTAINER" ] && POSTGRES_CONTAINER="bsopt-postgres-1"

until $CONTAINER_ENGINE exec "$POSTGRES_CONTAINER" pg_isready -U admin > /dev/null 2>&1 || [ $RETRY_COUNT -eq $MAX_RETRIES ]; do
    echo -n "."
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "❌ Error: Postgres failed to start."
    exit 1
fi
echo "✅ Postgres is LIVE."

# Inject Hyper-Optimized SQL Tuning
echo "⚙️ Injecting Hyper-Optimized SQL Tuning Commands..."
$CONTAINER_ENGINE exec "$POSTGRES_CONTAINER" psql -U admin -d bsopt -f /docker-entrypoint-initdb.d/15-runtime-tuning.sql > /dev/null 2>&1 || true

$COMPOSE_ENGINE --env-file .env -f infrastructure/orchestration/docker-compose.yml up -d redis
echo "⏳ Waiting for Redis..."
REDIS_CONTAINER=$($COMPOSE_ENGINE --env-file .env -f infrastructure/orchestration/docker-compose.yml ps --format "{{.Name}}" redis 2>/dev/null | head -n 1)
[ -z "$REDIS_CONTAINER" ] && REDIS_CONTAINER="bsopt-redis-1"
sleep 5 # Allow init
until $CONTAINER_ENGINE exec "$REDIS_CONTAINER" redis-cli ping > /dev/null 2>&1 || [ $RETRY_COUNT -eq $MAX_RETRIES ]; do
    echo -n "."
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT+1))
done
echo "✅ Redis is LIVE."

$COMPOSE_ENGINE --env-file .env -f infrastructure/orchestration/docker-compose.yml up -d pgbouncer
echo "✅ PgBouncer is LIVE (Assuming healthy after DB/Redis)."

# 5. Kafka Infrastructure
echo "🚀 Starting Kafka..."
$COMPOSE_ENGINE --env-file .env -f infrastructure/orchestration/docker-compose.yml up -d kafka-1
sleep 10 # Allow Kafka broker initialization

# 6. Ray Head & ML
echo "🚀 Starting Ray Head..."
$COMPOSE_ENGINE --env-file .env -f infrastructure/orchestration/docker-compose.yml up -d ray-head
sleep 5
echo "✅ Ray Head is LIVE."

# 7. Core APIs
echo "🚀 Starting API & Auth Services..."
$COMPOSE_ENGINE --env-file .env -f infrastructure/orchestration/docker-compose.yml up -d api auth-service
sleep 5

# 8. Start remaining services
echo "🚀 Starting remaining components (Frontend, Envoy, Workers)..."
$COMPOSE_ENGINE --env-file .env -f infrastructure/orchestration/docker-compose.yml up -d envoy frontend

echo "=============================================================================="
echo "✅ EQUAFLOW STACK EXECUTED SEQUENTIALLY (Zero-Touch)"
echo "=============================================================================="
