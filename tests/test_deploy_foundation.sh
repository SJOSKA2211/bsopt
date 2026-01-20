#!/usr/bin/env bash
# tests/test_deploy_foundation.sh

set -euo pipefail

# Path to the script under test
DEPLOY_SCRIPT="./deploy.sh"

# 1. Test if log function works
echo "Testing logging functions..."
if ! grep -q "log()" "$DEPLOY_SCRIPT"; then
    echo "FAILED: log function not found in $DEPLOY_SCRIPT"
    exit 1
fi

# 2. Test locking mechanism
echo "Testing locking mechanism..."
# Start a background process to hold the lock
sleep 10 &
SLEEP_PID=$!
echo "Mock PID: $SLEEP_PID"
LOCK_FILE=".deployment_lock"
echo "$SLEEP_PID" > "$LOCK_FILE"
echo "Lock file created at $(pwd)/$LOCK_FILE with content $(cat $LOCK_FILE)"

OUTPUT=$("$DEPLOY_SCRIPT" deploy 2>&1 || true)
echo "Output from deploy.sh: $OUTPUT"

if echo "$OUTPUT" | grep -q "Another deployment is already running"; then
    echo "SUCCESS: Lock file detected correctly"
else
    echo "FAILED: Lock file not detected"
    kill "$SLEEP_PID" 2>/dev/null || true
    rm -f "$LOCK_FILE"
    exit 1
fi

kill "$SLEEP_PID" 2>/dev/null || true
rm -f "$LOCK_FILE"
echo "Foundation tests passed!"
