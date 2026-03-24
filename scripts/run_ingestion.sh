#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 Launching Institutional Financial Data Ingestion (Zero-Mock)..."

# Ensure environment is clean
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src

# Standardize on uv for execution if available
if command -v uv > /dev/null; then
    RUN_CMD="uv run"
else
    RUN_CMD="python3"
fi

# Multi-Provider Concurrent Ingestion
echo "📊 Fetching Market Data from Polygon and Alpha Vantage Substrates..."
$RUN_CMD -m src.ingestion.pipeline --providers polygon yfinance nse

echo "✅ Ingestion Pipeline Sync Complete."
