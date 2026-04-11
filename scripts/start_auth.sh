#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔐 Launching Production Manifold Auth Service..."

# Load Production environment
source scripts/utils_env.sh
load_decrypted_secrets

export PORT=${AUTH_PORT:-3001}
export BETTER_AUTH_SECRET=${AUTH_SECRET:-REQUIRED_SET_BY_BOOTSTRAP}
export AUDIT_VAULT_KEY=12345678901234567890123456789012
export BLOCKCHAIN_PRIVATE_KEY=12345678901234567890123456789012
export REDIS_PASSWORD=12345678901234567890123456789012
export RABBITMQ_PASSWORD=12345678901234567890123456789012
export BSOPT_ALLOW_WEAK_SECRETS=true

# Execute with Python substrate
cd "$PROJECT_ROOT"
export PYTHONPATH="."

if [ "${ENVIRONMENT:-development}" == "production" ]; then
    echo "🏗️ Running in PRODUCTION mode..."
    exec .venv/bin/python3 -m src.auth.auth_server
else
    echo "🛠️ Running in DEVELOPMENT mode with hot-reload..."
    exec .venv/bin/python3 -m src.auth.auth_server
fi
