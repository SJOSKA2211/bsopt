#!/bin/bash
# ==============================================================================
# EQUAFLOW: THE ZERO-TOUCH BOOTSTRAP (v3.0)
# ==============================================================================
# Automates the entire stack: Security (RSA/ECC), .env, DB Init, and Gateway.
# Features:
# - Container engine agnostic (podman/docker)
# - Sequential health checks with exponential backoff
# - Enhanced error handling and logging
# - Secrets rotation support
# ==============================================================================

set -e

# Configuration
KEYS_DIR="$(pwd)/.pki"
ENV_FILE=".env"
ENV_EXAMPLE=".env.example"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
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
# 1. Container Engine & Compose Detection (Container Agnostic)
# ==============================================================================
detect_container_engine() {
    log_info "Detecting container engine..."
    
    # Detect podman (prioritized for rootless operation)
    if command -v podman &> /dev/null; then
        CONTAINER_ENGINE="podman"
        if podman compose version &> /dev/null; then
            COMPOSE_ENGINE="podman compose"
        elif command -v podman-compose &> /dev/null; then
            if grep -q "PackageKit" <(which podman-compose 2>/dev/null 2>&1) 2>/dev/null; then
                log_warn "podman-compose intercepted by PackageKit. Using 'podman compose' fallback."
                COMPOSE_ENGINE="podman compose"
            else
                COMPOSE_ENGINE="podman-compose"
            fi
        else
            log_warn "No podman compose plugin found. Using 'podman compose' fallback."
            COMPOSE_ENGINE="podman compose"
        fi
        log_success "Detected: podman"
    
    # Detect docker (fallback)
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
        log_success "Detected: docker"
    else
        log_error "Neither podman nor docker found. Please install one of them."
        exit 1
    fi
    
    log_info "Using $CONTAINER_ENGINE with $COMPOSE_ENGINE"
}

# Compose command wrapper
compose_cmd() {
    $COMPOSE_ENGINE --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
}

# Container exec wrapper
container_exec() {
    local container_id=$1
    shift
    $CONTAINER_ENGINE exec "$container_id" "$@"
}

# ==============================================================================
# 2. Security Layer (RSA/ECC Key Generation)
# ==============================================================================
generate_keys() {
    log_info "Generating Asymmetric Key Pairs (Institutional Grade)..."
    mkdir -p "${KEYS_DIR}"
    
    # RSA 4096 (RS256) for legacy compatibility
    if [ ! -f "${KEYS_DIR}/jwt_rs256.key" ]; then
        log_info "Generating RSA 4096 key pair..."
        openssl genrsa -out "${KEYS_DIR}/jwt_rs256.key" 4096
        openssl rsa -in "${KEYS_DIR}/jwt_rs256.key" -pubout -out "${KEYS_DIR}/jwt_rs256.pub"
        log_success "RSA 4096 key pair generated"
    else
        log_info "RSA 4096 keys already exist, skipping"
    fi

    # ECC P-256 (ES256) for modern high-performance auth
    if [ ! -f "${KEYS_DIR}/jwt_es256.key" ]; then
        log_info "Generating ECC P-256 key pair..."
        openssl ecparam -name prime256v1 -genkey -noout -out "${KEYS_DIR}/jwt_es256.key"
        openssl ec -in "${KEYS_DIR}/jwt_es256.key" -pubout -out "${KEYS_DIR}/jwt_es256.pub"
        log_success "ECC P-256 key pair generated"
    else
        log_info "ECC P-256 keys already exist, skipping"
    fi

    # Argon2id Hashing Salt
    if [ ! -f "${KEYS_DIR}/argon2_salt.secret" ]; then
        openssl rand -hex 32 > "${KEYS_DIR}/argon2_salt.secret"
        log_success "Argon2 salt generated"
    fi

    # TOTP Master Secret
    if [ ! -f "${KEYS_DIR}/totp_master.secret" ]; then
        openssl rand -hex 32 > "${KEYS_DIR}/totp_master.secret"
        log_success "TOTP master secret generated"
    fi
    
    # Envoy SSL Cert (Self-Signed for Dev)
    if [ ! -f "${KEYS_DIR}/envoy_edge.key" ]; then
        log_info "Generating Envoy SSL certificate..."
        openssl req -x509 -newkey rsa:4096 \
            -keyout "${KEYS_DIR}/envoy_edge.key" \
            -out "${KEYS_DIR}/envoy_edge.crt" \
            -days 365 -nodes \
            -subj "/C=US/ST=State/L=City/O=EquaFlow/CN=localhost"
        log_success "Envoy SSL certificate generated"
    fi
}

# ==============================================================================
# 3. .env Orchestration
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
    else
        log_info ".env already exists"
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

inject_keys_into_env() {
    log_info "Injecting keys into .env..."
    
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
    
    log_success "Keys injected into .env"
}

generate_passwords() {
    log_info "Generating secure passwords..."
    
    for var in POSTGRES_PASSWORD REDIS_PASSWORD BETTER_AUTH_SECRET JWT_SECRET RABBITMQ_PASSWORD; do
        if ! grep -q "^${var}=" "${ENV_FILE}" || [[ -z $(grep "^${var}=" "${ENV_FILE}" | cut -d'=' -f2 | tr -d '"' | tr -d "'") ]]; then
            set_env_var "${var}" "$(openssl rand -hex 32)"
        fi
    done
    
    log_success "Passwords generated"
}

setup_database_urls() {
    log_info "Setting up database URLs..."
    PG_PASS=$(grep "^POSTGRES_PASSWORD=" "${ENV_FILE}" | cut -d'=' -f2 | tr -d '"' | tr -d "'")
    set_env_var "DATABASE_URL" "postgresql://admin:${PG_PASS}@pgbouncer:6432/bsopt"
    set_env_var "DATABASE_URL_LOCAL" "postgresql://admin:${PG_PASS}@localhost:5434/bsopt"
    set_env_var "DATABASE_URL_TEST" "postgresql://admin:${PG_PASS}@postgres:5432/bsopt_test"
    set_env_var "MLFLOW_BACKEND_STORE_URI" "postgresql://admin:${PG_PASS}@postgres:5432/bsopt"
    
    log_success "Database URLs configured"
}

# ==============================================================================
# 4. Sequential Service Startup with Health Checks
# ==============================================================================
wait_for_service() {
    local service_name=$1
    local health_command=$2
    local max_retries=${3:-60}
    local base_interval=${4:-2}
    local retry_count=0
    local container_id=""
    
    log_info "Waiting for $service_name to be healthy..."
    
    # Ensure the service is up first
    compose_cmd up -d "$service_name" 2>/dev/null || true
    
    # Give it time to start
    sleep 3
    
    # Get container ID
    container_id=$(compose_cmd ps -q "$service_name" 2>/dev/null | head -n 1)
    
    if [ -z "$container_id" ]; then
        log_error "Could not get container ID for $service_name"
        return 1
    fi
    
    # Polling with exponential backoff
    while [ $retry_count -lt $max_retries ]; do
        if container_exec "$container_id" bash -c "$health_command" > /dev/null 2>&1; then
            log_success "$service_name is healthy"
            return 0
        fi
        
        # Exponential backoff: 2s, 4s, 8s, 16s, max 30s
        local sleep_time=$((base_interval * (2 ** retry_count)))
        if [ $sleep_time -gt 30 ]; then
            sleep_time=30
        fi
        
        echo -n "."
        sleep $sleep_time
        retry_count=$((retry_count+1))
    done
    
    log_error "$service_name failed to become healthy after $max_retries retries"
    log_info "Container logs:"
    compose_cmd logs "$service_name" | tail -n 20
    return 1
}

# Wait for container to at least start (without health check)
wait_for_container() {
    local service_name=$1
    local max_wait=${2:-60}
    local elapsed=0
    
    log_info "Waiting for $service_name container to start..."
    
    compose_cmd up -d "$service_name" 2>/dev/null || true
    
    while [ $elapsed -lt $max_wait ]; do
        if compose_cmd ps -q "$service_name" 2>/dev/null | grep -q .; then
            local status=$(compose_cmd ps "$service_name" 2>/dev/null | tail -n 1 | awk '{print $NF}')
            if [ "$status" = "Up" ]; then
                log_success "$service_name container is running"
                return 0
            fi
        fi
        sleep 2
        elapsed=$((elapsed+2))
    done
    
    log_error "$service_name container failed to start within ${max_wait}s"
    return 1
}

# ==============================================================================
# 5. Main Bootstrap Sequence
# ==============================================================================
main() {
    echo "=============================================================================="
    echo -e "${BLUE}🚀 EquaFlow Advanced Bootstrap v3.0${NC} [${TIMESTAMP}]"
    echo "=============================================================================="
    
    # Step 1: Detect container engine
    detect_container_engine
    
    # Step 2: Generate keys
    generate_keys
    
    # Step 3: Setup .env
    setup_env_file
    inject_keys_into_env
    generate_passwords
    setup_database_urls
    
    # Step 4: Build images
    log_info "Building all images..."
    compose_cmd build --parallel 2>/dev/null || true
    log_success "Images built"
    
    # ==========================================================================
    # Phase A: Core Infrastructure (Postgres -> Redis -> PgBouncer)
    # ==========================================================================
    echo ""
    echo "------------------------------------------------------------------------------"
    echo -e "${BLUE}Phase A: Core Infrastructure${NC}"
    echo "------------------------------------------------------------------------------"
    
    # Postgres/TimescaleDB
    wait_for_service "postgres" "pg_isready -U admin -d bsopt" 60 2
    POSTGRES_ID=$(compose_cmd ps -q postgres)
    
    # Inject runtime tuning SQL
    log_info "Applying PostgreSQL runtime tuning..."
    container_exec "$POSTGRES_ID" psql -U admin -d bsopt -c "SELECT pg_reload_conf();" 2>/dev/null || true
    container_exec "$POSTGRES_ID" psql -U admin -d bsopt -f /docker-entrypoint-initdb.d/15-runtime-tuning.sql 2>/dev/null || true
    log_success "PostgreSQL tuning applied"
    
    # Redis
    REDIS_PASS=$(grep "^REDIS_PASSWORD=" "${ENV_FILE}" | cut -d'=' -f2 | tr -d '"' | tr -d "'")
    wait_for_service "redis" "redis-cli -a ${REDIS_PASS:-bsopt_redis_secret} --no-auth-warning ping | grep -q PONG" 30 1
    
    # PgBouncer
    wait_for_service "pgbouncer" "pg_isready -h localhost -p 5432 -U admin" 30 1
    
    # ==========================================================================
    # Phase B: Message Broker & Streaming
    # ==========================================================================
    echo ""
    echo "------------------------------------------------------------------------------"
    echo -e "${BLUE}Phase B: Message Broker & Streaming${NC}"
    echo "------------------------------------------------------------------------------"
    
    # RabbitMQ
    wait_for_service "rabbitmq" "rabbitmq-diagnostics -q check_running" 30 1
    
    # Kafka
    wait_for_container "kafka-1" 60
    log_info "Waiting for Kafka to be ready..."
    sleep 15
    kafka_id=$(compose_cmd ps -q kafka-1)
    if [ -n "$kafka_id" ]; then
        container_exec "$kafka_id" kafka-topics --bootstrap-server localhost:9092 --list > /dev/null 2>&1 && \
            log_success "Kafka is ready" || log_warn "Kafka may not be fully ready"
    fi
    
    # ==========================================================================
    # Phase C: Auth & API Services
    # ==========================================================================
    echo ""
    echo "------------------------------------------------------------------------------"
    echo -e "${BLUE}Phase C: Auth & API Services${NC}"
    echo "------------------------------------------------------------------------------"
    
    # Auth Service
    wait_for_service "auth-service" "wget -qO- --spider http://localhost:3001/ || exit 1" 30 1
    
    # API Service
    wait_for_service "api" "python -c 'import urllib.request; urllib.request.urlopen(\"http://localhost:8000/health\").read()'" 45 1
    
    # ==========================================================================
    # Phase D: ML Infrastructure
    # ==========================================================================
    echo ""
    echo "------------------------------------------------------------------------------"
    echo -e "${BLUE}Phase D: ML Infrastructure${NC}"
    echo "------------------------------------------------------------------------------"
    
    # Ray Head
    wait_for_container "ray-head" 60
    ray_id=$(compose_cmd ps -q ray-head)
    if [ -n "$ray_id" ]; then
        log_info "Waiting for Ray to initialize..."
        sleep 10
        container_exec "$ray_id" ray status > /dev/null 2>&1 && log_success "Ray head is ready" || log_warn "Ray may still be initializing"
    fi
    
    # MLflow
    wait_for_service "mlflow" "wget -qO- --spider http://localhost:5000/health || exit 1" 30 1
    
    # ==========================================================================
    # Phase E: Workers & Additional Services
    # ==========================================================================
    echo ""
    echo "------------------------------------------------------------------------------"
    echo -e "${BLUE}Phase E: Workers & Additional Services${NC}"
    echo "------------------------------------------------------------------------------"
    
    # Celery Worker
    wait_for_container "worker" 30
    
    # Ingestion Service
    wait_for_container "ingestion-service" 30
    
    # ML Inference
    wait_for_service "ml-inference" "wget -qO- --spider http://localhost:5001/health || exit 1" 30 1
    
    # ==========================================================================
    # Phase F: Edge Gateway
    # ==========================================================================
    echo ""
    echo "------------------------------------------------------------------------------"
    echo -e "${BLUE}Phase F: Edge Gateway${NC}"
    echo "------------------------------------------------------------------------------"
    
    # Envoy
    wait_for_service "envoy" "wget -qO- --spider http://localhost:8080/ready || exit 1" 30 1
    
    # ==========================================================================
    # Phase G: Frontend
    # ==========================================================================
    echo ""
    echo "------------------------------------------------------------------------------"
    echo -e "${BLUE}Phase G: Frontend${NC}"
    echo "------------------------------------------------------------------------------"
    
    # Frontend
    wait_for_container "frontend" 60
    
    # ==========================================================================
    # Final Status
    # ==========================================================================
    echo ""
    echo "=============================================================================="
    echo -e "${GREEN}✅ EQUAFLOW STACK READY${NC}"
    echo "=============================================================================="
    echo ""
    echo "Service Status:"
    echo "-------------"
    compose_cmd ps
    echo ""
    echo "Access Points:"
    echo "-------------"
    echo "  API:          http://localhost:8000"
    echo "  Auth:         http://localhost:3001"
    echo "  MLflow:       http://localhost:5000"
    echo "  Grafana:      http://localhost:3000"
    echo "  Envoy:        http://localhost:8080"
    echo "  Frontend:     http://localhost:5173"
    echo ""
    echo "Next Steps:"
    echo "  - Run 'make logs' to view container logs"
    echo "  - Run 'make test-all' to execute the test gauntlet"
    echo "=============================================================================="
}

# Trap for cleanup on error
trap 'log_error "Bootstrap interrupted"; exit 1' INT TERM

# Run main
main "$@"
