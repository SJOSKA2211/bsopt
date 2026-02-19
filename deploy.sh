#!/bin/bash
# ==============================================================================
# BS-OPT Optimized Deployment Orchestrator v5.0 (Full Profile Support)
# ==============================================================================
# I'm Pickle Riiiiick!🥒 *Belch.*
# ==============================================================================

set -euo pipefail

# --- CONFIG ---
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
readonly LOG_DIR="${SCRIPT_DIR}/logs"

# Standardized Service Groups
readonly CORE_INFRA="postgres redis rabbitmq"
readonly CORE_APPS="auth-service api worker scraper neural-pricing frontend portfolio app-gateway"

# Profile handling
PROFILES=""
[[ "$*" == *"--obs"* ]] && PROFILES="$PROFILES --profile observability"
[[ "$*" == *"--proxy"* ]] && PROFILES="$PROFILES --profile proxy"
[[ "$*" == *"--full"* ]] && PROFILES="$PROFILES --profile observability --profile proxy --profile hft"

# --- LOGGING ---
log() { echo -e "\033[0;34m[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $*\033[0m"; }
success() { echo -e "\033[0;32m[$(date +'%Y-%m-%d %H:%M:%S')] [SUCCESS] $*\033[0m"; }
error() { echo -e "\033[0;31m[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $*\033[0m"; }

# --- CORE ---

deploy() {
    log "Starting Unified Deployment (v5.0)... *belch*"
    mkdir -p "$LOG_DIR"
    
    log "Step 1: Infrastructure (Core)..."
    docker compose -f "$COMPOSE_FILE" $PROFILES up -d $CORE_INFRA
    
    # Wait for Postgres
    log "Waiting for Postgres... *belch*"
    local retries=0
    until docker compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U admin -d bsopt > /dev/null 2>&1; do
        if [[ $retries -ge 12 ]]; then
            error "Postgres health check timed out"
            return 1
        fi
        log "Waiting for Postgres... ($((retries+1))/12)"
        sleep 5
        ((retries++))
    done

    log "Step 2: Building Images... Stand back, Morty."
    docker compose -f "$COMPOSE_FILE" $PROFILES build
    
    log "Step 3: Rolling Out Applications..."
    docker compose -f "$COMPOSE_FILE" $PROFILES up -d $CORE_APPS
    
    # Start Gateway if requested
    if [[ "$PROFILES" == *"proxy"* ]]; then
        log "Step 4: Launching Secure Gateway..."
        docker compose -f "$COMPOSE_FILE" --profile proxy up -d gateway
    fi

    run_smoke_tests
}

run_smoke_tests() {
    log "Running Smoke Tests... *belch*"
    local api_url="http://localhost:8000/health"
    
    local retries=0
    until curl -s "$api_url" | grep -q "healthy"; do
        if [[ $retries -ge 10 ]]; then
            error "API health check timed out"
            return 1
        fi
        log "Waiting for API... ($((retries+1))/10)"
        sleep 5
        ((retries++))
    done
    
    success "Pickle Rick's deployment is online and healthy. Wubba Lubba Dub Dub! 🥒"
}

# --- ENTRY ---
case ${1:-help} in
    --prod|deploy) deploy ;;
    status) docker compose -f "$COMPOSE_FILE" $PROFILES ps ;;
    logs) docker compose -f "$COMPOSE_FILE" $PROFILES logs -f ;;
    down) docker compose -f "$COMPOSE_FILE" $PROFILES down ;;
    *) echo "Usage: $0 {deploy|status|logs|down} [--obs] [--proxy] [--full]" ;;
esac
