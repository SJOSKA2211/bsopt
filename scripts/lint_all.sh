#!/bin/bash
set -e

echo "🥒 Pickle Rick's Linting Engine 🥒"

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate Virtual Environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run Ruff
echo "🕵️  Running Ruff..."
ruff check .

# Run Mypy (optional, can be slow)
if [[ "$*" == *"--strict"* ]]; then
    echo "🧠 Running Mypy..."
    mypy src
fi

echo "✅ Linting complete. No slop detected."
