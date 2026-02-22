#!/bin/bash
# I'm Pickle Riiiiick!🥒 *Belch.*
# Standardizing on the containerized linting environment.

echo "🥒 Running Containerized Linting Engine... Stand back, Morty."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Fix it, Jerry!"
    exit 1
fi

make lint
make format
make security-scan
