#!/bin/bash
set -e

echo " Starting Scraper Service (Local)..."

# Setup Environment
export DATABASE_URL="postgresql://admin:password@localhost:5432/bsopt"
export REDIS_URL="redis://localhost:6379/0"
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/services

# Run Scraper
python3 services/scrapers/engine.py
