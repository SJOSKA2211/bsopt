#!/bin/bash
set -e

# EquaFlow Blue-Green Deployment Script
# Orchestrates seamless transition between environment stacks.

COLOR_BLUE="blue"
COLOR_GREEN="green"

# 1. Determine current active color
CURRENT_COLOR=$(curl -s http://localhost:8080/health | jq -r .env_color || echo "blue")
NEW_COLOR=$([[ "$CURRENT_COLOR" == "blue" ]] && echo "green" || echo "blue")

echo "🚀 Starting Blue-Green Deployment: [Current: $CURRENT_COLOR] -> [Target: $NEW_COLOR]"

# 2. Spin up the NEW stack
# We use a suffix for src in the compose file or manage multiple projects
echo "🏗️ Building and starting $NEW_COLOR stack..."
export ENV_COLOR=$NEW_COLOR
podman-compose -p "bsopt-$NEW_COLOR" up -d --build

# 3. Health Check the NEW stack
echo "🩺 Verifying $NEW_COLOR stack health..."
MAX_RETRIES=10
RETRY_COUNT=0
HEALTHY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "http://localhost:$( [[ $NEW_COLOR == "green" ]] && echo "5002" || echo "5001" )/health" | grep -q "healthy"; then
        HEALTHY=true
        break
    fi
    echo "Wait for $NEW_COLOR stack to stabilize... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 5
    ((RETRY_COUNT++))
done

if [ "$HEALTHY" = false ]; then
    echo "❌ $NEW_COLOR stack health check FAILED. Rolling back..."
    podman-compose -p "bsopt-$NEW_COLOR" down
    exit 1
fi

# 4. Traffic Switch (Envoy Update)
echo "🔄 Switching traffic to $NEW_COLOR stack..."
# Repoint the 'api' upstream in Envoy config
sed -i "s/api-$CURRENT_COLOR/api-$NEW_COLOR/g" docker/envoy.yaml
# Hot-reload Envoy (using SIGHUP if supported or restart)
podman kill -s SIGHUP envoy || podman restart envoy

# 5. Cleanup (Optional: keep Blue for 5 mins then scale down)
echo "🧹 Deployment successful. $NEW_COLOR is now LIVE."
echo "Blue stack (bsopt-$CURRENT_COLOR) will be retained for 300s for emergency rollback."
sleep 300
podman-compose -p "bsopt-$CURRENT_COLOR" down
echo "✅ Finished. $CURRENT_COLOR stack decommissioned."
