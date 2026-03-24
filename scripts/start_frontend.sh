#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🎨 Launching Institutional EquaFlow Dashboard..."

cd src/frontend

if [ ! -d "node_modules" ]; then
    echo "📦 Initializing Frontend dependencies..."
    npm install --quiet
fi

if [ "${ENVIRONMENT:-development}" == "production" ]; then
    echo "🏗️ Running in PRODUCTION mode..."
    exec npm run start
else
    echo "🛠️ Running in DEVELOPMENT mode with hot-reload..."
    exec npm run dev
fi
