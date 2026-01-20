#!/usr/bin/env bash
# tests/test_deploy_tier1.sh

set -euo pipefail

DEPLOY_SCRIPT="./deploy.sh"

echo "Testing deploy-tier1 command presence..."
# We can't easily run it because Docker is missing, so we check for recognition
OUTPUT=$("./deploy.sh" deploy-tier1 2>&1 || true)

if echo "$OUTPUT" | grep -q "Tier 1"; then
    echo "SUCCESS: deploy-tier1 command recognized"
else
    echo "FAILED: deploy-tier1 command not recognized or failed unexpectedly"
    echo "Output: $OUTPUT"
    exit 1
fi