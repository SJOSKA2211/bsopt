#!/usr/bin/env bash
# tests/test_deploy_rollback.sh

set -euo pipefail

DEPLOY_SCRIPT="./deploy.sh"

echo "Testing backup/rollback command recognition..."
# We can run backup safely as it just tars files
OUTPUT_BACKUP=$("./deploy.sh" backup 2>&1 || true)
OUTPUT_ROLLBACK=$("./deploy.sh" rollback 2>&1 || true)

if echo "$OUTPUT_BACKUP" | grep -q "Checkpointing"; then
    echo "SUCCESS: backup command recognized"
else
    echo "FAILED: backup command not recognized"
    echo "Output: $OUTPUT_BACKUP"
    exit 1
fi

if echo "$OUTPUT_ROLLBACK" | grep -q "Rollback"; then
    echo "SUCCESS: rollback command recognized"
else
    echo "FAILED: rollback command not recognized"
    echo "Output: $OUTPUT_ROLLBACK"
    exit 1
fi

