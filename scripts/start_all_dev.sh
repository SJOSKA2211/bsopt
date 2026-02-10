#!/bin/bash
echo " Joseph Kamau Maina's Dev Stack Launcher "

# Project root (in case script is run from scripts/ dir)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Trap Ctrl-C and other signals to kill all background processes
cleanup() {
    echo -e "\n🛑 Shutting down..."
    # Kill the entire process group except the current process
    trap - SIGTERM
    ray stop --force > /dev/null 2>&1
    kill -- -$$ 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# 0. Environment Setup
export RAY_DEDUP_LOGS=0
export RAY_IGNORE_UNSTABLE_API_WARNING=1
export COMPOSE_DOCKER_CLI_BUILD=1
export DOCKER_BUILDKIT=1
export PYTHONWARNINGS="ignore::FutureWarning:ray,ignore::UserWarning:ray"
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-2}
export RAY_ADDRESS="127.0.0.1:6380"
export RAY_USAGE_STATS_ENABLED=0
export RAY_metrics_export_port=0
export RAY_metrics_export_binaries_path=""
export RAY_event_log_binaries_path=""
export RAY_memory_monitor_refresh_ms=0

# 0. Argument Parsing
RUN_LINT=false
NO_TAIL=false
for arg in "$@"; do
    if [ "$arg" == "--lint" ]; then
        RUN_LINT=true
    fi
    if [ "$arg" == "--no-tail" ]; then
        NO_TAIL=true
    fi
done

if [ "$RUN_LINT" = true ]; then
    ./scripts/lint_all.sh || exit 1
fi

# 1. Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Fix it, Jerry!"
    exit 1
fi

# 2. Start Infrastructure if not already running
DOCKER_COMPOSE="docker compose -f docker-compose.dev.yml"
RUNNING_SERVICES=$($DOCKER_COMPOSE ps --services --filter "status=running")
if [[ ! $RUNNING_SERVICES =~ "postgres" ]] || [[ ! $RUNNING_SERVICES =~ "redis" ]] || [[ ! $RUNNING_SERVICES =~ "rabbitmq" ]]; then
    echo " Starting Infrastructure (Postgres, Redis, RabbitMQ)..."
    $DOCKER_COMPOSE up -d --no-recreate postgres redis rabbitmq
fi


# 3. Wait for Infrastructure (Postgres, Redis, RabbitMQ)
echo "⏳ Waiting for Infrastructure to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0

check_infra_health() {
    local status=0
    local message=""

    $DOCKER_COMPOSE exec -T postgres pg_isready -U admin -d bsopt > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        message+="Postgres not ready. "
        status=1
    fi

    $DOCKER_COMPOSE exec -T redis redis-cli ping | grep -q PONG > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        message+="Redis not ready. "
        status=1
    fi

    # Check RabbitMQ health
    $DOCKER_COMPOSE exec -T rabbitmq rabbitmq-diagnostics -q check_running > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        message+="RabbitMQ not ready. "
        status=1
    fi

    if [ $status -ne 0 ]; then
        echo -n "$message"
        return 1
    fi
    return 0
}

until check_infra_health || [ $RETRY_COUNT -eq $MAX_RETRIES ]; do
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

# 3.5 Start Ray Cluster
echo "🐝 Starting Ray Cluster..."
if [ -d ".venv" ]; then

    source .venv/bin/activate
fi

ray stop --force > /dev/null 2>&1
# Try starting head node. On some systems --head requires --port
if ray start --head --num-cpus=$RAY_NUM_CPUS --include-dashboard=false --disable-usage-stats --port=6380 > /dev/null 2>&1; then
    export RAY_ADDRESS="127.0.0.1:6380"
    echo "✅ Ray Cluster is UP (127.0.0.1:6380)."
else
    # Fallback to local init if ray start fails
    echo "⚠️  Ray Cluster failed to start via 'ray start'. Services will attempt local ray.init()."
    unset RAY_ADDRESS
fi

# 4. Start App Services
echo " Starting App Services..."
./scripts/start_auth.sh &
PID_AUTH=$!

./scripts/start_api.sh &
PID_API=$!

./scripts/start_frontend.sh &
PID_FRONT=$!

./scripts/start_workers.sh &
PID_WORKERS=$!

./scripts/start_scraper.sh &
PID_SCRAPER=$!

./scripts/start_neural_pricing.sh &
PID_NEURAL=$!

# 5. Wait for App Services
echo "⏳ Waiting for App Services to be healthy..."
RETRY_COUNT=0
MAX_RETRIES=60

until (
    curl -s http://localhost:3001 > /dev/null && \
    curl -s http://localhost:8000/health > /dev/null && \
    curl -s http://localhost:5173 > /dev/null && \
    curl -s http://localhost:8001/health > /dev/null
) || [ $RETRY_COUNT -eq $MAX_RETRIES ]; do
    printf "."
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "\n❌ Some services failed to start. Check logs."
else
    echo -e "\n✅ All services are UP and HEALTHY."
fi

if [ "$NO_TAIL" = true ]; then
    echo " Services are running in background. Use 'docker compose logs -f' to tail."
    exit 0
fi

# 6. Tail Logs in Background
echo "📡 Tailing logs in background (Ctrl-C to stop all)..."
$DOCKER_COMPOSE logs -f --tail=0 &

