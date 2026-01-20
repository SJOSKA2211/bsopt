#!/usr/bin/env bash
# tests/test_deploy_task1.sh

set -euo pipefail

# Path to the script under test
DEPLOY_SCRIPT="./deploy.sh"
LOG_FILE="logs/deployment.log"

# Clean up before tests
rm -f .deployment_lock
rm -f "$LOG_FILE"
mkdir -p logs

echo "Testing logging..."
# Run a command that should log
"$DEPLOY_SCRIPT" validate-system > /dev/null 2>&1 || true

if [[ ! -f "$LOG_FILE" ]]; then
    echo "FAILED: Log file $LOG_FILE not created"
    exit 1
fi

if ! grep -q "INFO" "$LOG_FILE"; then
    echo "FAILED: INFO level logs not found in $LOG_FILE"
    exit 1
fi
echo "SUCCESS: Logging works"

echo "Testing locking..."
# Start a background process to hold the lock
sleep 10 &
SLEEP_PID=$!
echo "$SLEEP_PID" > .deployment_lock

# Run a command that should check the lock
OUTPUT=$("$DEPLOY_SCRIPT" validate-system 2>&1 || true)

if echo "$OUTPUT" | grep -q "Another deployment is already running"; then
    echo "SUCCESS: Lock file detected correctly"
else
    echo "FAILED: Lock file not detected for validate-system"
    kill "$SLEEP_PID" || true
    rm -f .deployment_lock
    exit 1
fi

kill "$SLEEP_PID" || true
rm -f .deployment_lock
echo "Task 1 tests passed!"
