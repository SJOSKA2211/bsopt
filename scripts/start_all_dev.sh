#!/bin/bash
echo "🥒 Pickle Rick's Dev Stack Launcher 🥒"

# Project root (in case script is run from scripts/ dir)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Trap Ctrl-C and other signals to kill all background processes
trap "echo -e '\n🛑 Shutting down...'; trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# 1. Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Fix it, Jerry!"
    exit 1
fi

# 2. Start Infrastructure
./scripts/start_infra.sh &
PID_INFRA=$!

# 3. Wait for Infrastructure (Postgres, Redis)
echo "⏳ Waiting for Infrastructure to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0
DOCKER_COMPOSE="docker compose -f docker-compose.dev.yml"

until (
    $DOCKER_COMPOSE exec -T postgres pg_isready -U admin > /dev/null 2>&1 && \
    $DOCKER_COMPOSE exec -T redis redis-cli ping | grep -q PONG > /dev/null 2>&1
) || [ $RETRY_COUNT -eq $MAX_RETRIES ]; do
    printf "."
    sleep 1
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "\n❌ Infrastructure failed to stabilize. Check Docker logs:"
    $DOCKER_COMPOSE logs --tail=20
    exit 1
fi
echo -e "\n✅ Infrastructure is READY."

# 4. Start App Services
echo "🚀 Starting App Services..."
./scripts/start_auth.sh &
PID_AUTH=$!

./scripts/start_api.sh &
PID_API=$!

./scripts/start_frontend.sh &
PID_FRONT=$!

# 5. Tail Logs in Background
echo "📡 Tailing logs in background (Ctrl-C to stop all)..."
$DOCKER_COMPOSE logs -f --tail=0 &

# Wait for all background processes
wait
