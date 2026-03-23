#!/bin/bash
# ==============================================================================
# EQUAFLOW: THE ZERO-TOUCH BOOTSTRAP (v3.1 - Hardened)
# ==============================================================================
# Automates the entire stack: PKI, encrypted secrets, DB Init, and Gateway.
# Features:
# - Root CA & Service-level mTLS
# - RSA-4096 Runtime Secret Vaulting
# - Container engine agnostic (podman/docker)
# - Strict sequential health gating
# ==============================================================================

set -e

# Configuration
KEYS_DIR="$(pwd)/.pki"
ENV_FILE=".env"
ENV_EXAMPLE=".env.example"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Central Configuration
# Load shared environment utilities
if [ -f "scripts/utils_env.sh" ]; then
    source scripts/utils_env.sh
else
    echo "[ERROR] scripts/utils_env.sh not found."
    exit 1
fi

COMPOSE_FILE="infrastructure/orchestration/docker-compose.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 0. Prerequisite Check Function
check_prereq() {
    local cmd=$1
    if ! command -v "$cmd" &> /dev/null; then
        log_error "$cmd is not installed. Please install it first."
        exit 1
    fi
}

check_prereq openssl

# ==============================================================================
# 1. Container Engine & Compose Detection
# ==============================================================================
# ==============================================================================
# 1. Container Engine & Compose Detection
# ==============================================================================
# Handled by scripts/utils_env.sh detect_container_engine

# Container exec wrapper
container_exec() {
    if [ -z "$CONTAINER_ENGINE" ]; then
        detect_container_engine
    fi
    local container_id=$1
    shift
    $CONTAINER_ENGINE exec "$container_id" "$@"
}

# ==============================================================================
# 2. Security Layer (PKI & Vault)
# ==============================================================================
initialize_pki() {
    log_info "Initializing Institutional PKI Layer..."
    chmod +x ./scripts/setup_pki.sh
    ./scripts/setup_pki.sh
}

# ==============================================================================
# 3. .env Orchestration & Encryption (Hardened)
# ==============================================================================
setup_env_file() {
    log_info "Setting up .env file..."
    
    if [ ! -f "${ENV_FILE}" ]; then
        if [ -f "${ENV_EXAMPLE}" ]; then
            cp "${ENV_EXAMPLE}" "${ENV_FILE}"
            log_success "Created .env from template"
        else
            touch "${ENV_FILE}"
            log_warn "Created empty .env file"
        fi
    fi
}

set_env_var() {
    local key=$1
    local value=$2
    # Ensure value is quoted correctly in sed
    if grep -q "^${key}=" "${ENV_FILE}"; then
        sed -i "s|^${key}=.*|${key}=\"${value}\"|g" "${ENV_FILE}"
    else
        echo "${key}=\"${value}\"" >> "${ENV_FILE}"
    fi
}

encrypt_secret() {
    local val=$1
    echo -n "$val" | openssl pkeyutl -encrypt -pubin -inkey "${KEYS_DIR}/vault/vault.pub" | base64 | tr -d '\n'
}

secure_env_file() {
    log_info "Securing sensitive environment variables..."
    
    # Ensure PKI is initialized first (needed for vault.pub and JWT keys)
    if [ ! -f "${KEYS_DIR}/vault/vault.pub" ]; then
        initialize_pki
    fi

    # 1. Generate core JWT/Encryption keys if missing from PKI
    if [ ! -f "${KEYS_DIR}/jwt_rs256.key" ]; then
        initialize_pki
    fi

    # 2. Map PKI keys to .env if not already present
    log_info "Mapping PKI keys to .env..."
    set_env_var "JWT_RS256_PRIVATE" "$(cat ${KEYS_DIR}/jwt_rs256.key | base64 | tr -d '\n')"
    set_env_var "JWT_RS256_PUBLIC" "$(cat ${KEYS_DIR}/jwt_rs256.pub | base64 | tr -d '\n')"
    set_env_var "JWT_ES256_PRIVATE" "$(cat ${KEYS_DIR}/jwt_es256.key | base64 | tr -d '\n')"
    set_env_var "JWT_ES256_PUBLIC" "$(cat ${KEYS_DIR}/jwt_es256.pub | base64 | tr -d '\n')"
    set_env_var "ARGON2_SALT" "$(cat ${KEYS_DIR}/argon2_salt.secret)"

    # 3. Generate and Encrypt passwords
    local SENSITIVE_VARS=("POSTGRES_PASSWORD" "REDIS_PASSWORD" "JWT_SECRET" "BETTER_AUTH_SECRET" "RABBITMQ_PASSWORD" "MINIO_ROOT_PASSWORD")
    
    for var in "${SENSITIVE_VARS[@]}"; do
        local CURRENT_VAL=$(grep "^${var}=" "${ENV_FILE}" | cut -d'=' -f2- | tr -d '"' | tr -d "'")
        
        if [ -z "$CURRENT_VAL" ] && ! grep -q "^ENC_${var}=" "${ENV_FILE}"; then
            log_info "Generating random $var..."
            local NEW_VAL=$(openssl rand -hex 32)
            set_env_var "${var}" "$NEW_VAL"
            CURRENT_VAL="$NEW_VAL"
        fi

        if [ -n "$CURRENT_VAL" ] && [[ ! "$CURRENT_VAL" =~ ^ENC_ ]]; then
            log_info "Encrypting $var..."
            local ENC_VAL=$(encrypt_secret "$CURRENT_VAL")
            set_env_var "ENC_${var}" "$ENC_VAL"
            sed -i "/^${var}=/d" "${ENV_FILE}"
        fi
    done
    
    log_success "Secrets vaulted securely in .env"
}

# ==============================================================================
# 4. Sequential Service Startup with Health Gating
# ==============================================================================
wait_for_service() {
    local service_name=$1
    local health_command=$2
    local max_retries=${3:-60}
    local base_interval=${4:-2}
    local retry_count=0
    
    log_info "Starting $service_name..."
    compose_cmd -f "$COMPOSE_FILE" up -d "$service_name"
    
    log_info "Waiting for $service_name to be healthy..."
    
    while [ $retry_count -lt $max_retries ]; do
        local container_id=$(compose_cmd -f "$COMPOSE_FILE" ps -q "$service_name" 2>/dev/null | head -n 1)
        if [ -n "$container_id" ]; then
            # Priority 1: Container Engine Health Status
            local state=$($CONTAINER_ENGINE inspect -f '{{.State.Health.Status}}' "$container_id" 2>/dev/null)
            if [ "$state" == "healthy" ]; then
                log_success "$service_name is healthy (reported by engine)"
                return 0
            fi
            
            # Priority 2: Manual health command callback
            if container_exec "$container_id" sh -c "$health_command" > /dev/null 2>&1; then
                log_success "$service_name is healthy (manual check)"
                return 0
            fi
        fi
        
        # Exponential backoff (capped at 10s)
        local sleep_time=$(( base_interval + (retry_count / 10) ))
        [ $sleep_time -gt 10 ] && sleep_time=10
        
        echo -n "."
        sleep $sleep_time
        retry_count=$((retry_count+1))
    done
    
    log_error "$service_name failed to become healthy. Logs:"
    compose_cmd -f "$COMPOSE_FILE" logs "$service_name" | tail -n 20
    return 1
}

# ==============================================================================
# 5. Main Bootstrap Sequence
# ==============================================================================
main() {
    echo "=============================================================================="
    echo -e "${BLUE}🚀 EquaFlow Institutional Bootstrap v4.5 (Hardened Edition)${NC} [${TIMESTAMP}]"
    echo "=============================================================================="
    
    detect_container_engine
    setup_env_file
    secure_env_file
    
    # Reload secrets into current shell session
    load_decrypted_secrets

    # Ensure base image is up to date
    if [ -f "infrastructure/orchestration/Dockerfile.base" ]; then
        log_info "Checking base image equaflow-base:latest..."
        $CONTAINER_ENGINE build -t equaflow-base:latest -f infrastructure/orchestration/Dockerfile.base .
    fi

    log_info "Building core operational cluster..."
    compose_cmd -f "$COMPOSE_FILE" build --parallel api auth-service worker neural-pricing
    
    echo ""
    echo "------------------------------------------------------------------------------"
    echo -e "${BLUE}Phase A: Zero-Trust Data Persistence (PostgreSQL/TimescaleDB)${NC}"
    echo "------------------------------------------------------------------------------"
    # Strict pg_isready loop as requested
    wait_for_service "postgres" "pg_isready -U ${POSTGRES_USER:-admin} -d ${POSTGRES_DB:-bsopt}"
    wait_for_service "pgbouncer" "pg_isready -h localhost -p 5432 -U ${POSTGRES_USER:-admin}"
    
    echo ""
    echo "------------------------------------------------------------------------------"
    echo -e "${BLUE}Phase B: Performance Backplane (Redis/RabbitMQ/Kafka)${NC}"
    echo "------------------------------------------------------------------------------"
    wait_for_service "redis" "redis-cli -a \"${REDIS_PASSWORD}\" ping"
    wait_for_service "rabbitmq" "rabbitmq-diagnostics -q check_running"
    
    if grep -q "kafka-1:" "$COMPOSE_FILE"; then
        wait_for_service "kafka-1" "echo > /dev/tcp/localhost/9092"
    fi
    
    echo ""
    echo "------------------------------------------------------------------------------"
    echo -e "${BLUE}Phase C: Application Mesh (Auth/API/ML)${NC}"
    echo "------------------------------------------------------------------------------"
    wait_for_service "auth-service" "wget -qO- http://localhost:3001/ || exit 1"
    wait_for_service "api" "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()\""
    wait_for_service "neural-pricing" "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()\""
    
    echo ""
    echo "------------------------------------------------------------------------------"
    echo -e "${BLUE}Phase D: Edge Ingress (Envoy Gateway)${NC}"
    echo "------------------------------------------------------------------------------"
    wait_for_service "envoy" "wget -qO- http://localhost:9901/ready || exit 1"
    
    echo ""
    echo "=============================================================================="
    log_success "EQUAFLOW STACK IS ONLINE AND HARDENED"
    echo "=============================================================================="
}

trap 'log_error "Bootstrap failed at step $BASH_COMMAND"; exit 1' ERR
trap 'log_error "Bootstrap interrupted"; exit 1' INT TERM
main "$@"

