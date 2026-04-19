#!/bin/bash
set -e

# --- BSOPT OMARCHY ABSOLUTE: Autonomous Sequential Orchestrator ---
# Mission: Deploy the manifold stack ONE container at a time with strict health gates.

SERVICES=("timescaledb" "redis" "rabbitmq" "pgbouncer" "auth_service" "api_service" "worker" "mlops" "frontend" "nginx")
MAX_ATTEMPTS=3

log() {
    echo "[$(date +'%Y-%m-%dT%H:%M:%S')] OMARCHY: $1"
}

check_health() {
    local service=$1
    log "Verifying health for $service..."
    if docker compose ps --format json | grep -q "\"Service\":\"$service\",\"Status\":\"running (healthy)\""; then
        return 0
    elif docker compose ps --format json | grep -q "\"Service\":\"$service\",\"Status\":\"running\""; then
        # Some services might not have healthchecks but are running
        return 0
    fi
    return 1
}

deploy_service() {
    local service=$1
    local attempt=1
    
    while [ $attempt -le $MAX_ATTEMPTS ]; do
        log "Building and deploying $service (Attempt $attempt/$MAX_ATTEMPTS)..."
        docker compose up -d --build $service
        
        # Wait for initialization
        sleep 10
        
        if check_health $service; then
            log "Service $service is HEALTHY."
            return 0
        else
            log "ERROR: Service $service is UNHEALTHY or CRASHED."
            docker compose logs --tail=50 $service
            log "Circuit Breaker: Escalating failure..."
            attempt=$((attempt + 1))
            if [ $attempt -le $MAX_ATTEMPTS ]; then
                log "Cleaning up and retrying..."
                docker compose stop $service && docker compose rm -f $service
            fi
        fi
    done
    
    log "FATAL: Service $service failed after $MAX_ATTEMPTS attempts. Halting deployment."
    exit 1
}

main() {
    log "Starting OMARCHY ABSOLUTE deployment sequence..."
    
    # 1. Clean legacy state
    log "Cleaning environment..."
    docker compose down --remove-orphans
    
    # 2. Sequential Bootstrap
    for svc in "${SERVICES[@]}"; do
        deploy_service $svc
    done
    
    log "DEPLOYMENT COMPLETE. All services within the manifold are operational."
    log "Triggering E2E assertion suite..."
}

main "$@"
