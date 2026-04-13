#!/bin/bash
# ==
# Manifold: AUTOMATED REBUILD & DEPLOY
# ==
set -e

# 1. Load Secrets
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo " Starting Manifold Optimized Build Process..."

# 2. Build Base Images
echo " Building hardened foundations: manifold-base:builder and manifold-base:latest..."
docker build -t manifold-base:builder \
    --target builder \
    -f infrastructure/orchestration/Dockerfile.base .

docker build -t manifold-base:latest \
    --target production \
    -f infrastructure/orchestration/Dockerfile.base .

# 3. Build Stack
echo "️  Building service stack using docker-compose..."
cd infrastructure && docker-compose build

# 4. Deploy
echo " Deploying isolated networks and services..."
docker-compose up -d

echo " Manifold deployed successfully."
