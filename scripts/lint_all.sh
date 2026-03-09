#!/bin/bash
# *
# Standardizing on the containerized linting environment.

echo " Running Containerized Linting Engine... "

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Fix it, Jerry!"
    exit 1
fi

make lint
make format
make security-scan
