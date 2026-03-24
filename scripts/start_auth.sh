#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔐 Launching Institutional EquaFlow Auth Service..."

# Load institutional environment
source scripts/utils_env.sh
load_decrypted_secrets

export PORT=${AUTH_PORT:-3001}
export BETTER_AUTH_SECRET=${AUTH_SECRET:-REQUIRED_SET_BY_BOOTSTRAP}

# Execute with institutional Node.js substrate
cd src/auth
if [ ! -d "node_modules" ]; then
    echo "📦 Initializing Node.js dependencies..."
    npm install --quiet
fi

if [ "${ENVIRONMENT:-development}" == "production" ]; then
    echo "🏗️ Running in PRODUCTION mode..."
    exec npm run start
else
    echo "🛠️ Running in DEVELOPMENT mode with hot-reload..."
    exec npm run dev
fi
