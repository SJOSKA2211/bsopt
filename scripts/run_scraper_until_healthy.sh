#!/bin/bash
# scripts/run_scraper_until_healthy.sh
set -euo pipefail

# 1. Setup Environment
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SOURCE_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f "scripts/utils_env.sh" ]; then
    source scripts/utils_env.sh
else
    echo "[ERROR] scripts/utils_env.sh not found."
    exit 1
fi

detect_container_engine
load_decrypted_secrets

COMPOSE_FILE="infrastructure/orchestration/docker-compose.yml"

# 2. Start Scraper Services
echo "🚀 Starting Scraper services using $CONTAINER_ENGINE..."
# We start ingestion-service, nse-scraper, yfinance-scraper, and transformer
$COMPOSE_ENGINE -f "$COMPOSE_FILE" up -d ingestion-service nse-scraper yfinance-scraper transformer

# 3. Wait for Healthy (Reported by Revamped Engine)
echo "⏳ Handing over to Revamped Health Engine..."
uv run python scripts/engine_health.py --wait
