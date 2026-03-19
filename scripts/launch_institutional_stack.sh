#!/bin/bash
set -e

# EquaFlow Institutional Launch Orchestrator
# A single "Push-Button" script to verify and start the entire stack.

echo "===================================================="
echo "EquaFlow Institutional Launch Orchestrator (v2026)"
echo "===================================================="

# 1. Start Infrastructure
echo -e "\n📦 Step 1: Starting Orchestration Stack..."
make up-d

# 2. Wait for Readiness
echo -e "\n🩺 Step 2: Waiting for System Readiness..."
MAX_RETRIES=12
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if python3 scripts/verify_readiness.py; then
        echo "✅ System is READY."
        break
    fi
    echo "Waiting for services to stabilize... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 10
    ((RETRY_COUNT++))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "❌ System Readiness check FAILED."
    exit 1
fi

# 3. Execute Day-0 Smoke Test
echo -e "\n🚀 Step 3: Executing Institutional Day-0 Smoke Test..."
python3 scripts/institutional_smoke_test.py

# 4. Success Summary
echo -e "\n===================================================="
echo "🎉 LAUNCH SUCCESSFUL: Institutional Stack is LIVE"
echo "===================================================="
echo "Access Dashboard: http://localhost:5173"
echo "Access GraphQL:   http://localhost:4000/graphql"
echo "Monitoring:       http://localhost:3000 (Grafana)"
echo "===================================================="
