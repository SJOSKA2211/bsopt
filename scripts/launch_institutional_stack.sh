#!/bin/bash
# scripts/launch_stack.sh - Zero-Touch Orchestrator
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo " Launch Orchestrator v2026"
echo "=="

# 1. Bootstrap State
echo " Step 1: Initializing Secure Substrate..."
bash scripts/bootstrap.sh

# 2. Start Full Stack
echo " Step 2: Launching Microservices Ecosystem..."
make up-d

# 3. Comprehensive Readiness Audit
echo "🩺 Step 3: Executing Readiness Audit..."
MAX_RETRIES=15
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if python3 scripts/verify_readiness.py; then
        echo " System Integrity Verified."
        break
    fi
    echo "🟠 Waiting for system stabilization... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 10
    ((RETRY_COUNT++))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo " Fatal: System failed to reach readiness."
    exit 1
fi

# 4. Final Smoke Test
python3 scripts/smoke_test.py

echo " STACK IS LIVE."
