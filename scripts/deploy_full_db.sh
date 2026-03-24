#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🏟️ Launching Production Persistent Substrate Deployment..."

# Load Production environment and detection
source scripts/utils_env.sh
detect_container_engine
load_decrypted_secrets

# 1. Start Persistent Tier
echo "📦 Phase 1: Initializing PostgreSQL & TimescaleDB Substrate..."
$COMPOSE_ENGINE -f infrastructure/orchestration/docker-compose.yml up -d postgres pgbouncer

# 2. Readiness Handshake
echo "⏳ Waiting for database stabilization..."
until $COMPOSE_ENGINE -f infrastructure/orchestration/docker-compose.yml exec -T postgres pg_isready -U admin -d bsopt > /dev/null 2>&1; do
    sleep 2
done

# 3. Schema & Migration Factory
echo "📜 Phase 2: Executing Full Schema & Migration Factory..."
# We utilize the containerized deployment script for consistency
$COMPOSE_ENGINE -f infrastructure/orchestration/docker-compose.yml exec -T postgres /bin/bash -c "
  for script in \$(find /docker-entrypoint-initdb.d/ -name '*.sql' | sort); do
    echo \"Applying \$script...\"
    psql -U admin -d bsopt -f \$script
  done
"

# 4. Optimization Pass
echo "🛠️ Phase 3: Executing Production Performance Optimization..."
bash scripts/deploy_db_updates.sh

echo "🎉 DATABASE SUBSTRATE IS FULLY DEPLOYED AND OPTIMIZED."
