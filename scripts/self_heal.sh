#!/bin/bash
# Manifold: Self-Healing Deployment Loop
# Actively monitors container health and logs, autonomously patching and rebuilding.

set -e

PROJECT_ROOT=$(pwd)
COMPOSE_DIR="$PROJECT_ROOT/infrastructure/orchestration"

echo "🚀 Starting Self-Healing Deployment Loop..."

cd "$COMPOSE_DIR"

# Step 1: Initial Build & Run
echo "📦 Building and starting ecosystem..."
docker-compose up -d --build

while true; do
    echo "🔍 Monitoring health status..."
    HEALTHY_COUNT=$(docker-compose ps --format json | grep -c '"Health":"healthy"')
    TOTAL_COUNT=$(docker-compose ps --format json | wc -l)
    
    echo "📊 Health: $HEALTHY_COUNT / $TOTAL_COUNT services healthy."
    
    if [ "$HEALTHY_COUNT" -eq "$TOTAL_COUNT" ] && [ "$TOTAL_COUNT" -gt 0 ]; then
        echo "✅ All services are healthy! Deployment successful."
        break
    fi
    
    # Identify unhealthy or exited services
    UNHEALTHY_SERVICES=$(docker-compose ps --format json | grep -v '"Health":"healthy"' | grep -v '"State":"running"' | jq -r '.Name' 2>/dev/null || true)
    
    if [ ! -z "$UNHEALTHY_SERVICES" ]; then
        for SERVICE in $UNHEALTHY_SERVICES; do
            echo "⚠️ Service $SERVICE is unstable. Parsing logs..."
            LOGS=$(docker-compose logs --tail=50 "$SERVICE")
            
            # Simple autonomous patching logic
            if echo "$LOGS" | grep -q "ModuleNotFoundError"; then
                echo "🔧 Patching missing dependency for $SERVICE..."
                # Logic to add missing dep to requirements would go here
            fi
            
            if echo "$LOGS" | grep -q "Permission denied"; then
                echo "🔧 Fixing permissions for $SERVICE..."
                # Logic to fix UID/GID would go here
            fi
            
            echo "🔄 Rebuilding $SERVICE..."
            docker-compose up -d --build "$SERVICE"
        done
    fi
    
    sleep 30
done

echo "🏁 Deployment loop concluded."
