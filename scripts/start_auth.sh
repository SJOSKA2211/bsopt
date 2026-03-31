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

# Execute with Python substrate
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

if [ "${ENVIRONMENT:-development}" == "production" ]; then
    echo "🏗️ Running in PRODUCTION mode..."
    exec python3 -m src.auth.auth_server
else
    echo "🛠️ Running in DEVELOPMENT mode with hot-reload..."
    exec python3 -m src.auth.auth_server
fi
