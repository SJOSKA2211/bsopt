#!/bin/bash
# ==
# BSOPT: THE ZERO-TOUCH MODERNIZED BOOTSTRAP (v4.0 - CPU Optimized)
# ==
# Features:
# - Forced PKI Regeneration (mTLS & Asymmetric JWT)
# - Sequential Health-Aware Deployment
# - Strict Environment Validation (${VAR:?error})
# - BuildKit + Alpine/Distroless Optimization
# ==

set -e

# Configuration
KEYS_DIR="$(pwd)/.pki"
ENV_FILE=".env"
ENV_EXAMPLE=".env.example"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 1. Prerequisite Check
check_prereq() {
    local cmd=$1
    if ! command -v "$cmd" &> /dev/null; then
        log_error "$cmd is not installed."
        exit 1
    fi
}
check_prereq openssl
check_prereq docker

# 2. Container Engine Detection
detect_engine() {
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
}
detect_engine

# 3. PKI Regeneration (FORCED)
regenerate_pki() {
    log_info "Regenerating Infrastructure PKI (Forced)..."
    rm -rf "$KEYS_DIR"
    mkdir -p "$KEYS_DIR"
    chmod 700 "$KEYS_DIR"
    
    # Root CA
    openssl genrsa -out "$KEYS_DIR/root_ca.key" 4096
    openssl req -x509 -new -nodes -key "$KEYS_DIR/root_ca.key" -sha256 -days 3650 \
        -out "$KEYS_DIR/root_ca.crt" -subj "/CN=BSOPT-Root-CA/O=BSOPT/C=US"

    # Auth Identity (Server)
    openssl genrsa -out "$KEYS_DIR/auth-service.key" 2048
    openssl req -new -key "$KEYS_DIR/auth-service.key" -out "$KEYS_DIR/auth-service.csr" -subj "/CN=auth-api/O=BSOPT/C=US"
    openssl x509 -req -in "$KEYS_DIR/auth-service.csr" -CA "$KEYS_DIR/root_ca.crt" -CAkey "$KEYS_DIR/root_ca.key" \
        -CAcreateserial -out "$KEYS_DIR/auth-service.crt" -days 365 -sha256
    rm "$KEYS_DIR/auth-service.csr"

    # API Identity (Client mTLS)
    openssl genrsa -out "$KEYS_DIR/api-client.key" 2048
    openssl req -new -key "$KEYS_DIR/api-client.key" -out "$KEYS_DIR/api-client.csr" -subj "/CN=pricing-api/O=BSOPT/C=US"
    openssl x509 -req -in "$KEYS_DIR/api-client.csr" -CA "$KEYS_DIR/root_ca.crt" -CAkey "$KEYS_DIR/root_ca.key" \
        -CAcreateserial -out "$KEYS_DIR/api-client.crt" -days 365 -sha256
    rm "$KEYS_DIR/api-client.csr"

    # JWT Keys (RSA + EC)
    openssl genrsa -out "$KEYS_DIR/jwt_rs256.key" 4096
    openssl rsa -in "$KEYS_DIR/jwt_rs256.key" -pubout -out "$KEYS_DIR/jwt_rs256.pub"
    openssl ecparam -name prime256v1 -genkey -noout -out "$KEYS_DIR/jwt_es256.key"
    openssl ec -in "$KEYS_DIR/jwt_es256.key" -pubout -out "$KEYS_DIR/jwt_es256.pub"

    # Vault Keys
    openssl genrsa -out "$KEYS_DIR/vault.key" 4096
    openssl rsa -in "$KEYS_DIR/vault.key" -pubout -out "$KEYS_DIR/vault.pub"
    
    openssl rand -base64 32 > "$KEYS_DIR/argon2_salt.secret"
    chmod 600 "$KEYS_DIR"/*.key
    log_success "PKI Stack Regenerated."
}

# 4. Environment Stabilization
generate_env() {
    log_info "Generating stabilized .env file..."
    if [ ! -f "$ENV_EXAMPLE" ]; then
        log_error ".env.example missing. Cannot stabilize environment."
        exit 1
    fi

    # Read example, strip defaults, generate secure ones
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    
    # Helper to set/replace var
    set_var() {
        local key=$1
        local val=$2
        if grep -q "^${key}=" "$ENV_FILE"; then
            sed -i "s|^${key}=.*|${key}=${val}|g" "$ENV_FILE"
        else
            echo "${key}=${val}" >> "$ENV_FILE"
        fi
    }

    # Generate Secure Passwords and Defaults
    set_var "POSTGRES_USER" "admin"
    set_var "POSTGRES_DB" "bsopt"
    set_var "POSTGRES_PASSWORD" "$(openssl rand -hex 16)"
    set_var "REDIS_PASSWORD" "$(openssl rand -hex 16)"
    set_var "RABBITMQ_USER" "admin"
    set_var "RABBITMQ_PASSWORD" "$(openssl rand -hex 16)"
    set_var "JWT_SECRET" "$(openssl rand -hex 32)"
    set_var "BETTER_AUTH_SECRET" "$(openssl rand -hex 32)"
    
    # Map PKI to .env
    set_var "JWT_PRIVATE_KEY" "\"$(cat $KEYS_DIR/jwt_rs256.key | base64 -w0)\""
    set_var "JWT_PUBLIC_KEY" "\"$(cat $KEYS_DIR/jwt_rs256.pub | base64 -w0)\""
    set_var "GRPC_CA_CERT" "$KEYS_DIR/root_ca.crt"
    set_var "GRPC_SERVER_CERT" "$KEYS_DIR/auth-service.crt"
    set_var "GRPC_SERVER_KEY" "$KEYS_DIR/auth-service.key"
    set_var "GRPC_CLIENT_CERT" "$KEYS_DIR/api-client.crt"
    set_var "GRPC_CLIENT_KEY" "$KEYS_DIR/api-client.key"

    log_success ".env file stabilized and secured."
}

# 5. Sequential Deployment Loop
deploy_service() {
    local service=$1
    local check_cmd=$2
    
    log_info "Deploying $service..."
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    DOCKER_BUILDKIT=1 $COMPOSE_CMD build "$service"
    $COMPOSE_CMD up -d "$service"
    
    log_info "Waiting for $service health..."
    local retries=0
    while [ $retries -lt 30 ]; do
        if $COMPOSE_CMD ps "$service" | grep -q "healthy"; then
            log_success "$service is healthy."
            return 0
        fi
        # Fallback manual check
        if [ -n "$check_cmd" ] && $COMPOSE_CMD exec -T "$service" sh -c "$check_cmd" >/dev/null 2>&1; then
            log_success "$service is healthy (via manual check)."
            return 0
        fi
        echo -n "."
        sleep 2
        retries=$((retries + 1))
    done
    
    log_error "$service failed health check. Inspecting logs..."
    $COMPOSE_CMD logs "$service" | tail -n 20
    exit 1
}

# 6. Main Sequence
main() {
    clear
    echo -e "${GREEN}=== BSOPT MODERNIZED BOOTSTRAP ===${NC}"
    
    regenerate_pki
    generate_env
    
    log_info "Starting Sequential Deployment Cycle..."
    
    # Infrastructure
    deploy_service "postgres" "pg_isready -U admin"
    deploy_service "redis" "redis-cli -a \${REDIS_PASSWORD} ping"
    
    # Core
    deploy_service "auth_api" "curl -f http://localhost:3001/health"
    deploy_service "pricing_api" "curl -f http://localhost:8000/health"
    
    # Workers & Frontend
    deploy_service "math_worker" "celery -A src.workers.math_worker inspect ping"
    deploy_service "frontend" ""
    
    # Gateway
    deploy_service "nginx" ""
    
    log_success "DEPLOYMENT COMPLETE. ALL SERVICES HEALTHY."
    echo -e "${GREEN}API:${NC} http://localhost/api/v1"
    echo -e "${GREEN}UI:${NC}  http://localhost"
}

main "$@"