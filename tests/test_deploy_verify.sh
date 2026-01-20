#!/usr/bin/env bash
# tests/test_deploy_verify.sh

set -euo pipefail

DEPLOY_SCRIPT="./deploy.sh"

echo "Testing verify command recognition..."
OUTPUT=$("./deploy.sh" verify 2>&1 || true)

if echo "$OUTPUT" | grep -qi "smoke tests"; then
    echo "SUCCESS: verify command recognized"
else
    echo "FAILED: verify command not recognized"
    echo "Output: $OUTPUT"
    exit 1
fi

