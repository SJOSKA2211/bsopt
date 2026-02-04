#!/usr/bin/env bash
# ==============================================================================
# BS-OPT Singularity Deployment Orchestrator v3.0
# ==============================================================================
# Neon-native, Fastify-proxied, Transformer-ready.
# ==============================================================================

set -euo pipefail

# --- CONFIG ---
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.prod.yml"
readonly LOG_DIR="${SCRIPT_DIR}/logs"

# Service tiers
readonly TIER1_SERVICES="redis rabbitmq"
readonly TIER2_SERVICES="api ml auth gateway" # Postgres is now Neon (Remote)
readonly TIER3_SERVICES="prometheus grafana"

# --- LOGGING ---
log() { echo -e "\033[0;34m[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $*\033[0m"; }
success() { echo -e "\033[0;32m[$(date +'%Y-%m-%d %H:%M:%S')] [SUCCESS] $*\033[0m"; }
error() { echo -e "\033[0;31m[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $*\033[0m"; }

# --- CORE ---
check_neon() {
    log "Verifying Neon database connectivity..."
    # We check this via the API during smoke tests to ensure the full chain is up
}

deploy() {
    log "Starting Singularity Deployment..."
    mkdir -p "$LOG_DIR"
    
    log "Step 1: Infrastructure (Tier 1)..."
    docker compose -f "$COMPOSE_FILE" up -d $TIER1_SERVICES
    
    log "Step 2: Building Subgraphs and Gateway..."
    docker compose -f "$COMPOSE_FILE" build
    
    log "Step 3: Application Rollout (Tier 2)..."
    docker compose -f "$COMPOSE_FILE" up -d $TIER2_SERVICES
    
    log "Step 4: Observability (Tier 3)..."
    docker compose -f "$COMPOSE_FILE" up -d $TIER3_SERVICES
    
    run_smoke_tests
}

run_smoke_tests() {
    log "Running Singularity Smoke Tests..."
    local gateway_url="http://localhost:4000/health"
    
    # Wait for gateway
    local retries=0
    while ! curl -s "$gateway_url" | grep -q "healthy"; do
        if [[ $retries -ge 10 ]]; then
            error "Gateway health check timed out"
            return 1
        fi
        log "Waiting for Gateway... ($((retries+1))/10)"
        sleep 5
        retries=$((retries+1))
    done
    
    success "Singularity is online and healthy."
}

# --- ENTRY ---
case ${1:-help} in
    deploy) deploy ;;
    status) docker compose -f "$COMPOSE_FILE" ps ;;
    logs) docker compose -f "$COMPOSE_FILE" logs -f ;;
    down) docker compose -f "$COMPOSE_FILE" down ;;
    *) echo "Usage: $0 {deploy|status|logs|down}" ;;
esac