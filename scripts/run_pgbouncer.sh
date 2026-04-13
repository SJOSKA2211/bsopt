#!/bin/bash
# scripts/run_pgbouncer.sh - Start PgBouncer and verify health via pg_isready
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 1. Detect Container Engine
if command -v docker &> /dev/null && docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo " Error: docker compose is not installed."
    exit 1
fi

# Load Hardened environment with secret decryption
source scripts/utils_env.sh
load_decrypted_secrets

# Function to check pgbouncer health
check_health() {
    if PGBOUNCER_ADMIN_PASSWORD="$POSTGRES_PASSWORD" .venv/bin/python -c "
import asyncio
import sys
import os
from scripts.system_sentinel import check_pgbouncer

async def run():
    os.environ['PGBOUNCER_HOST'] = '127.0.0.1'
    os.environ['PGBOUNCER_PORT'] = '6432'
    await check_pgbouncer()

if __name__ == '__main__':
    asyncio.run(run())
" 2>&1 | grep -q "\[HEALTHY\]"; then
        return 0
    else
        return 1
fi
}

echo " Checking if PgBouncer is already healthy..."
if check_health; then
    echo " PgBouncer is already ACTIVE and HEALTHY. Skipping start."
    echo " PgBouncer Pool Engine is Online and Verified."
    exit 0
fi

echo " Starting PgBouncer..."
# Try to start without --force-recreate first to avoid unnecessary restarts if it's just stopped
if ! $COMPOSE_CMD -f infrastructure/orchestration/docker-compose.yml up -d pgbouncer; then
    echo "️ Failed to start pgbouncer normally, trying with --force-recreate..."
    $COMPOSE_CMD -f infrastructure/orchestration/docker-compose.yml up -d --force-recreate pgbouncer
fi

echo "⏳ Verifying PgBouncer engine health..."
RETRIES=30
INTERVAL=5
SUCCESS=0

for ((i=1; i<=RETRIES; i++)); do
    echo "   [Attempt $i/$RETRIES] Querying Reporting Engine..."
    
    if check_health; then
        echo " PgBouncer is ACTIVE, OPTIMIZED, and reporting healthy metrics."
        SUCCESS=1
        break
    fi
    
    echo "   ️ Engine not fully pressurized yet. Retrying in ${INTERVAL}s..."
    sleep $INTERVAL
done

if [ $SUCCESS -eq 0 ]; then
    echo " Fatal: PgBouncer failed to reach optimized healthy state."
    exit 1
fi

echo " PgBouncer Pool Engine is Online and Verified."
