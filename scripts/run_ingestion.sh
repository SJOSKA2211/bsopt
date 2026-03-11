#!/bin/bash
# High-Volume Asynchronous Data Ingestion Launcher
# Triggers NSE scraper and US Market (yfinance) bulk downloads concurrently.

set -e

echo "🚀 Starting High-Volume Financial Data Ingestion..."

# Set PYTHONPATH to include src
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Run the concurrent ingestion orchestrator
python3 src/scrapers/concurrent_ingestion.py

echo "✅ Ingestion Pipeline Complete."
