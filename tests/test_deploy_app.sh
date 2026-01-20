#!/usr/bin/env bash
# tests/test_deploy_app.sh

set -euo pipefail

DEPLOY_SCRIPT="./deploy.sh"

echo "Testing deploy-tier2/3 command recognition..."
OUTPUT2=$("./deploy.sh" deploy-tier2 2>&1 || true)
OUTPUT3=$("./deploy.sh" deploy-tier3 2>&1 || true)

if echo "$OUTPUT2" | grep -q "Tier 2" && echo "$OUTPUT3" | grep -q "Tier 3"; then
    echo "SUCCESS: Tier 2 and Tier 3 commands recognized"
else
    echo "FAILED: Commands not recognized"
    echo "Tier 2 Output: $OUTPUT2"
    echo "Tier 3 Output: $OUTPUT3"
    exit 1
fi

