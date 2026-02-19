#!/bin/bash
# I'm Pickle Riiiiick!🥒 *Belch.*
# Standardizing on the containerized linting environment.

echo "🥒 Running Containerized Linting Engine... Stand back, Morty."

<<<<<<< Updated upstream
# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate Virtual Environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
=======
# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Fix it, Jerry!"
    exit 1
>>>>>>> Stashed changes
fi

make lint
make format
make security-scan
