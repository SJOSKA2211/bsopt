#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔐 Launching Production Manifold Auth Service..."

# Load Production environment
source scripts/utils_env.sh
load_decrypted_secrets

export PORT=${AUTH_PORT:-3001}
export BETTER_AUTH_SECRET=${AUTH_SECRET:-$BETTER_AUTH_SECRET}
export AUDIT_VAULT_KEY=${AUDIT_VAULT_KEY:-12345678901234567890123456789012}
export BSOPT_ALLOW_WEAK_SECRETS=true

# Execute with Python substrate
cd "$PROJECT_ROOT"
export PYTHONPATH="."

echo "DEBUG: BETTER_AUTH_SECRET=$BETTER_AUTH_SECRET"
echo "DEBUG: REDIS_PASSWORD=$REDIS_PASSWORD"
echo "DEBUG: RABBITMQ_PASSWORD=$RABBITMQ_PASSWORD"

if [ "${ENVIRONMENT:-development}" == "production" ]; then
    echo "🏗️ Running in PRODUCTION mode..."
    exec .venv/bin/python3 -u -m src.auth.auth_server
else
    echo "🛠️ Running in DEVELOPMENT mode with hot-reload..."
    exec .venv/bin/python3 -u -m src.auth.auth_server
fi
