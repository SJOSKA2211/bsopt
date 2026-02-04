#!/bin/bash
# Headless Load Test for C100k
# Usage: ./run_load_test.sh <target_url>

TARGET_URL=${1:-"http://localhost"}
USERS=100000
SPAWN_RATE=1000
RUN_TIME="10m"

echo "🚀 Starting C100k WebSocket Stress Test..."
echo "Target: $TARGET_URL"
echo "Users: $USERS"

.venv/bin/locust -f performance/locust_ws_stress.py \
    --headless \
    --users $USERS \
    --spawn-rate $SPAWN_RATE \
    --run-time $RUN_TIME \
    --host $TARGET_URL \
    --html report_c100k.html \
    --csv performance_results
