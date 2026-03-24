#!/bin/bash
# scripts/start_scraper.sh - Institutional Market Scraper Orchestrator (Zero-Mock)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🕷️ Launching Institutional EquaFlow Scraper Substrate..."

# Load institutional environment
source scripts/utils_env.sh
load_decrypted_secrets

# Institutional Runtime Environment
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src

# Execute with institutional Python substrate
if command -v uv > /dev/null; then
    exec uv run -m src.ingestion.engine
else
    exec python3 -m src.ingestion.engine
fi
