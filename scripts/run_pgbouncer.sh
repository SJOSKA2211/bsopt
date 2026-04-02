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
    echo "❌ Error: docker compose is not installed."
    exit 1
fi

echo "🚀 Starting PgBouncer..."
$COMPOSE_CMD -f infrastructure/orchestration/docker-compose.yml up -d pgbouncer

echo "⏳ Verifying PgBouncer health..."
RETRIES=30
INTERVAL=5
SUCCESS=0

for ((i=1; i<=RETRIES; i++)); do
    echo "   [Attempt $i/$RETRIES] Checking status..."
    
    # Use pg_isready inside the container which has the correct certs and networking
    if $COMPOSE_CMD -f infrastructure/orchestration/docker-compose.yml exec pgbouncer sh -c "PGSSLCERT=/etc/pgbouncer/pgbouncer.crt PGSSLKEY=/tmp/pgbouncer.key PGSSLROOTCERT=/etc/pgbouncer/root_ca.crt pg_isready -h 127.0.0.1 -p 6432 -U admin" | grep -q "accepting connections"; then
        echo "✅ PgBouncer is ACTIVE and accepting connections."
        SUCCESS=1
        break
    fi
    
    echo "   ⚠️ PgBouncer not ready yet. Retrying in ${INTERVAL}s..."
    sleep $INTERVAL
done

if [ $SUCCESS -eq 0 ]; then
    echo "❌ Fatal: PgBouncer failed to reach healthy state after $RETRIES attempts."
    exit 1
fi

echo "🏁 PgBouncer is Online and Verified."
