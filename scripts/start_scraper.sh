#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🕷️ Launching Production Manifold Scraper Substrate..."

# Load Production environment
source scripts/utils_env.sh
load_decrypted_secrets

export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src

# Execute with Production Python substrate
if command -v uv > /dev/null; then
    exec uv run -m src.ingestion.engine
else
    exec python3 -m src.ingestion.engine
fi
