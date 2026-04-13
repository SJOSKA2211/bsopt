#!/bin/bash
# scripts/start_all_dev.sh - Unified Dev Stack Launcher
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Load environment and detection
source scripts/utils_env.sh
detect_container_engine
load_decrypted_secrets

echo " Unified Dev Stack Orchestrator v2026"
echo "=="

# 1. Start Core Infrastructure
echo " Phase 1: Launching Infrastructure Substrate..."
bash scripts/start_infra.sh

# 2. Start Application Services
echo "️ Phase 2: Launching Application Microservices..."
$COMPOSE_ENGINE -f infrastructure/orchestration/docker-compose.yml up -d --build auth-service api envoy frontend scraper neural-pricing worker

# 3. Synchronous Readiness Handshake
echo "🩺 Phase 3: Executing Readiness Audit..."
MAX_RETRIES=20
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if python3 scripts/verify_readiness.py; then
        echo " System Integrity Verified."
        break
    fi
    echo "🟠 Waiting for system stabilization... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 5
    ((RETRY_COUNT++))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo " Fatal: System failed to reach readiness."
    exit 1
fi

echo " STACK IS LIVE."
echo "Access Dashboard: http://localhost:5173"

# 4. Tail Logs (Optional)
if [[ "${1:-}" != "--no-tail" ]]; then
    echo " Tailing logs... (Ctrl-C to stop tailing)"
    $COMPOSE_ENGINE -f infrastructure/orchestration/docker-compose.yml logs -f --tail=20
fi
