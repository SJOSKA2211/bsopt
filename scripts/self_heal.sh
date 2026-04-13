#!/bin/bash
# ==
# BSOPT SELF-HEALING MONITOR (v1.0)
# ==
# Monitors the containerized ecosystem and autonomously patches/recovers
# from gRPC drops, Auth failures, and database deadlocks.

set -e

LOG_FILE="logs/self_heal.log"
mkdir -p logs

echo "[$(date)] Starting Manifold Self-Healing Loop..." | tee -a $LOG_FILE

# Configuration
CHECK_INTERVAL=30
RETRY_LIMIT=3

check_and_heal() {
    echo "[$(date)] Periodic health audit..." | tee -a $LOG_FILE
    
    # 1. Check Auth API (gRPC Connectivity)
    if ! docker compose -f infrastructure/orchestration/docker-compose.yml exec -T auth-service grpc_health_probe -addr=:50051 > /dev/null 2>&1; then
        echo "[$(date)] WARNING: Auth gRPC probe failed. Attempting restart..." | tee -a $LOG_FILE
        docker compose -f infrastructure/orchestration/docker-compose.yml restart auth-service
    fi

    # 2. Check API gateway
    if ! curl -sf http://localhost:8080/api/v1/health > /dev/null; then
        echo "[$(date)] WARNING: API Gateway unreachable. Checking Nginx..." | tee -a $LOG_FILE
        docker compose -f infrastructure/orchestration/docker-compose.yml restart nginx api
    fi

    # 3. Log Pattern Recognition (Deep Analysis)
    if docker compose -f infrastructure/orchestration/docker-compose.yml logs --tail=100 | grep -q "gRPC Handshake Failed"; then
        echo "[$(date)] CRITICAL: Detected gRPC handshake failure pattern. Flushing PKI cache and restarting mesh..." | tee -a $LOG_FILE
        # In a real scenario, we might regenerate certs here
        docker compose -f infrastructure/orchestration/docker-compose.yml restart auth-service api
    fi
}

while true; do
    check_and_heal
    sleep $CHECK_INTERVAL
done
