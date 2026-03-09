#!/bin/bash
# High-Performance Engine's Unified Dev Stack Launcher 
# "Optimizing manifold execution. We're using Docker."

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Detect Docker Compose (High-Performance Detection)
if [ -x "./docker-compose" ]; then
    COMPOSE_BIN="./docker-compose"
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_BIN="docker-compose"
elif docker compose version >/dev/null 2>&1; then
    COMPOSE_BIN="docker compose"
else
    echo "❌ Docker Compose not found. Fix it, jerry!"
    exit 1
fi

DOCKER_COMPOSE="$COMPOSE_BIN -f docker-compose.dev.yml"

# Trap Ctrl-C to shut down containers if requested (optional, usually we keep them up)
cleanup() {
    echo -e "\n🛑 Signal received. Containers remain running. Use './scripts/start_infra.sh down' if you want a full stop."
    exit 0
}
trap cleanup SIGINT SIGTERM

# 0. Argument Parsing
RUN_LINT=false
NO_TAIL=false
for arg in "$@"; do
    if [ "$arg" == "--lint" ]; then RUN_LINT=true; fi
    if [ "$arg" == "--no-tail" ]; then NO_TAIL=true; fi
done

# 1. Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Fix it, Jerry!"
    exit 1
fi

# 2. Check Infrastructure status
echo " Checking Infrastructure status..."
# Check for key infra containers (postgres is the best signal)
INFRA_RUNNING=$($DOCKER_COMPOSE ps --services --filter "status=running" | grep -q "^postgres$" && echo true || echo false)

if [ "$INFRA_RUNNING" = "false" ]; then
    echo " Infrastructure missing. Launching via start_infra.sh..."
    ./scripts/start_infra.sh
else
    echo " Infrastructure already active. Skipping redundant initialization."
fi

# 3. Run Lint if requested
if [ "$RUN_LINT" = true ]; then
    echo " Running Containerized Lint..."
    $DOCKER_COMPOSE --profile test run --rm test-runner ruff check . || exit 1
fi

# 4. Start Ray Cluster in Docker
echo "🐝 Starting Ray Cluster (Containerized)..."
$DOCKER_COMPOSE up -d --build ray-head rl-training-worker

# 5. Start App Services in Docker
echo " Launching App Services (Containerized)..."
$DOCKER_COMPOSE up -d --build auth-service api app-gateway frontend scraper neural-pricing worker-ml

# 6. Unified Health Check
echo "⏳ Waiting for App Services to be healthy..."
MAX_RETRIES=30
RETRY_COUNT=0

check_app_health() {
    # Check API
    curl -s http://localhost:8000/health | grep -q "ok" || return 1
    # Check Auth
    curl -s http://localhost:3001/health | grep -q "operational" || return 1
    # Check Gateway
    curl -s http://localhost:4000/health | grep -q "operational" || return 1
    # Check Neural Pricing
    curl -s http://localhost:8001/health | grep -q "ok" || return 1
    return 0
}

until check_app_health || [ $RETRY_COUNT -eq $MAX_RETRIES ]; do
    printf "."
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "\n❌ Some services failed to stabilize. Check 'docker compose logs'."
else
    echo -e "\n All services are UP and HEALTHY. High-Performance Active."
fi

if [ "$NO_TAIL" = true ]; then
    exit 0
fi

# 7. Tail Logs
echo "📡 Tailing logs... (Ctrl-C to stop tailing)"
$DOCKER_COMPOSE logs -f --tail=10
