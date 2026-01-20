#!/usr/bin/env bash
# tests/test_deploy_optimize.sh

set -euo pipefail

DEPLOY_SCRIPT="./deploy.sh"

echo "Testing optimize command..."
OUTPUT=$("./deploy.sh" optimize 2>&1 || true)

if echo "$OUTPUT" | grep -q "Applying kernel optimizations" || echo "$OUTPUT" | grep -q "Skipping kernel tuning"; then
    echo "SUCCESS: optimize_kernel executed"
else
    echo "FAILED: optimize_kernel output not recognized"
    echo "Output: $OUTPUT"
    exit 1
fi

