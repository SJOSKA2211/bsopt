#!/bin/bash
# *
# Standardizing on the containerized test environment.

echo " Running Containerized Tests... "

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Using local fallback (Jerry mode)..."
    export ENVIRONMENT=test
    export DATABASE_URL="sqlite:///:memory:"
    pytest --cov=src "$@"
    exit $?
fi

# Use the Makefile target I built for this
make test-all
