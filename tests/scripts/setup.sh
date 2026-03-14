#!/bin/bash

# Black-Scholes Option Pricing Platform - Enhanced Setup Script
# This script prepares a robust development environment.

set -euo pipefail # Exit on error, treat unset variables as errors, and propagate exit codes
trap 'echo "❌ An error occurred. Exiting..." >&2' ERR

echo "========================================="
echo "  BSOPT Platform - Enhanced Setup"
echo "========================================="
echo ""

# --- Helper Functions ---
command_exists() {
    command -v "$1" &> /dev/null
}

# --- Prerequisite Checks ---
echo "1. Checking prerequisites..."

if ! command_exists docker; then
    echo "❌ Docker is not installed. Please install Docker and try again." >&2
    exit 1
fi
echo "   ✓ Docker found"

if command_exists docker-compose; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "❌ Docker Compose is not installed. Please install it and try again." >&2
    exit 1
fi
echo "   ✓ Docker Compose found"
echo ""

# --- Environment Setup (.env) ---
echo "2. Setting up environment file..."
if [ -f .env ]; then
    echo "   ✓ '.env' file already exists. Skipping creation."
else
    echo "   - No '.env' file found. Creating from '.env.example'..."
    cp .env.example .env
    
    # Generate a new secret key automatically
    echo "   - Generating new JWT secret..."
    if command_exists openssl; then
        NEW_SECRET=$(openssl rand -hex 32)
        sed -i "s|your-super-secret-key-change-this-in-production-min-32-chars|$NEW_SECRET|" .env
        echo "   ✓ New JWT secret written to '.env'."
    else
        echo "   ⚠️ openssl not found. Please generate a strong JWT_SECRET in '.env' manually."
    fi
fi
echo ""

# --- Build and Start Services ---
echo "3. Building and starting Docker services..."
echo "   - Building images..."
$COMPOSE_CMD build
echo "   - Starting services in detached mode..."
$COMPOSE_CMD up -d
echo "   ✓ All services started."
echo ""

# --- Database Initialization ---
echo "4. Initializing database..."
echo "   - Waiting for PostgreSQL to be ready (up to 30s)..."

# Wait for the database to become healthy
WAIT_COUNT=0
until $COMPOSE_CMD exec -T db pg_isready -U "user" -d "options_db" &> /dev/null; do
    ((WAIT_COUNT++))
    if [ "$WAIT_COUNT" -gt 15 ]; then
        echo "   ❌ Database did not become ready in time. Please check 'docker-compose logs db'." >&2
        exit 1
    fi
    sleep 2
done

echo "   ✓ PostgreSQL is ready."
echo "   - Applying database schema..."
$COMPOSE_CMD exec api alembic upgrade head
echo "   ✓ Database schema initialized."
echo ""

# --- Final Instructions ---
echo "========================================="
echo "✅  Setup Complete!"
echo "========================================="
echo ""
echo "Your development environment is up and running."
echo ""
echo "Services:"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Frontend: http://localhost:3000"
echo "  - MLflow:   http://localhost:5000"
echo "  - Jupyter:  http://localhost:8888"
echo "  - RabbitMQ: http://localhost:15672"
echo ""
echo "Next steps:"
echo "  - To view logs: $COMPOSE_CMD logs -f"
echo "  - To stop services: $COMPOSE_CMD down"
echo ""