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
detect_container_engine() {
    log_info "Detecting container engine..."
    
    # Try podman first (often preferred in institutional/rootless environments)
    if command -v podman &> /dev/null; then
        CONTAINER_ENGINE="podman"
        if podman compose version &> /dev/null; then
            COMPOSE_ENGINE="podman compose"
        elif command -v podman-compose &> /dev/null; then
            COMPOSE_ENGINE="podman-compose"
        else
            COMPOSE_ENGINE="podman compose"
        fi
        log_success "Detected: podman ($COMPOSE_ENGINE)"
    elif command -v docker &> /dev/null; then
        CONTAINER_ENGINE="docker"
        if docker compose version &> /dev/null; then
            COMPOSE_ENGINE="docker compose"
        elif command -v docker-compose &> /dev/null; then
            COMPOSE_ENGINE="docker-compose"
        else
            log_error "Docker found but no compose engine detected."
            exit 1
        fi
        log_success "Detected: docker ($COMPOSE_ENGINE)"
    else
        log_error "No container engine (podman/docker) found."
        exit 1
    fi
}

# Compose command wrapper
compose_cmd() {
    # Ensure secrets are loaded into the environment before we run compose commands
    load_decrypted_secrets
    
    if [ -f "$ENV_FILE" ]; then
        $COMPOSE_ENGINE --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
    else
        $COMPOSE_ENGINE -f "$COMPOSE_FILE" "$@"
    fi
}

# Container exec wrapper
container_exec() {
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
    
    # Generate passwords if missing or empty
    local SENSITIVE_VARS=("POSTGRES_PASSWORD" "REDIS_PASSWORD" "JWT_SECRET" "BETTER_AUTH_SECRET" "RABBITMQ_PASSWORD" "MINIO_ROOT_PASSWORD")
    
    for var in "${SENSITIVE_VARS[@]}"; do
        local CURRENT_VAL=$(grep "^${var}=" "${ENV_FILE}" | cut -d'=' -f2 | tr -d '"' | tr -d "'")
        if [ -z "$CURRENT_VAL" ] && ! grep -q "^ENC_${var}=" "${ENV_FILE}"; then
            log_info "Generating random $var..."
            local NEW_VAL=$(openssl rand -hex 32)
            set_env_var "${var}" "$NEW_VAL"
            CURRENT_VAL="$NEW_VAL"
        fi

        # Encrypt if not already encrypted
        if [ -n "$CURRENT_VAL" ] && [[ ! "$CURRENT_VAL" =~ ^ENC_ ]]; then
            log_info "Encrypting $var..."
            local ENC_VAL=$(encrypt_secret "$CURRENT_VAL")
            set_env_var "ENC_${var}" "$ENC_VAL"
            # Remove plaintext version for security at rest
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
    compose_cmd up -d "$service_name"
    
    log_info "Waiting for $service_name to be healthy..."
    
    while [ $retry_count -lt $max_retries ]; do
        local container_id=$(compose_cmd ps -q "$service_name" 2>/dev/null | head -n 1)
        if [ -n "$container_id" ]; then
            if container_exec "$container_id" sh -c "$health_command" > /dev/null 2>&1; then
                log_success "$service_name is healthy"
                return 0
            fi
        fi
        echo -n "."
        sleep $base_interval
        retry_count=$((retry_count+1))
    done
    
    log_error "$service_name failed to become healthy. Logs:"
    compose_cmd logs "$service_name" | tail -n 20
    return 1
}

# ==============================================================================
# 5. Main Bootstrap Sequence
# ==============================================================================
main() {
    echo "=============================================================================="
    echo -e "${BLUE}🚀 EquaFlow Institutional Bootstrap v4.0 (Enhanced)${NC} [${TIMESTAMP}]"
    echo "=============================================================================="
    
    detect_container_engine
    initialize_pki
    setup_env_file
    secure_env_file
    
    log_info "Building core images..."
    compose_cmd build api auth-service worker neural-pricing
    
    echo ""
    echo "------------------------------------------------------------------------------"
    echo -e "${BLUE}Phase A: Persistent Database Layer${NC}"
    echo "------------------------------------------------------------------------------"
    wait_for_service "postgres" "pg_isready -U admin -d bsopt"
    wait_for_service "pgbouncer" "pg_isready -h localhost -p 5432 -U admin"
    
    echo ""
    echo "------------------------------------------------------------------------------"
    echo -e "${BLUE}Phase B: Performance Caching & Message Bus${NC}"
    echo "------------------------------------------------------------------------------"
    # Note: load_decrypted_secrets ensures REDIS_PASSWORD is in shell env
    wait_for_service "redis" "redis-cli -a \${REDIS_PASSWORD} ping"
    wait_for_service "rabbitmq" "rabbitmq-diagnostics -q check_running"
    
    echo ""
    echo "------------------------------------------------------------------------------"
    echo -e "${BLUE}Phase C: Distributed Application Microservices${NC}"
    echo "------------------------------------------------------------------------------"
    wait_for_service "auth-service" "wget -qO- --spider http://localhost:3001/ || exit 1"
    wait_for_service "api" "wget -qO- --spider http://localhost:8000/health || exit 1"
    wait_for_service "neural-pricing" "wget -qO- --spider http://localhost:8000/health || exit 1"
    
    echo ""
    echo "------------------------------------------------------------------------------"
    echo -e "${BLUE}Phase D: Edge Orchestration (Envoy)${NC}"
    echo "------------------------------------------------------------------------------"
    wait_for_service "envoy" "wget -qO- --spider http://localhost:9901/ready || exit 1"
    
    echo ""
    echo "=============================================================================="
    echo -e "${GREEN}✅ EQUAFLOW INSTITUTIONAL STACK FULLY BOOTSTRAPPED${NC}"
    echo "=============================================================================="
}

trap 'log_error "Bootstrap interrupted"; exit 1' INT TERM
main "$@"
