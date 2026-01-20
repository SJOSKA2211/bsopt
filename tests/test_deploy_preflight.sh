#!/usr/bin/env bash
# tests/test_deploy_preflight.sh

set -euo pipefail

DEPLOY_SCRIPT="./deploy.sh"

echo "Testing dependency check..."
# Ensure core dependencies are present in the list
if ! grep -q "\"docker\"" "$DEPLOY_SCRIPT" || ! grep -q "\"jq\"" "$DEPLOY_SCRIPT"; then
    echo "FAILED: Core dependencies not found in $DEPLOY_SCRIPT"
    exit 1
fi

echo "Testing resource validation..."
# Check if MIN_DISK_GB is 50
if ! grep -q "readonly MIN_DISK_GB=50" "$DEPLOY_SCRIPT"; then
    echo "FAILED: MIN_DISK_GB is not 50"
    exit 1
fi

echo "Testing port availability check..."
# Ensure it checks port 80 and 443
if ! grep -q "80" "$DEPLOY_SCRIPT" || ! grep -q "443" "$DEPLOY_SCRIPT"; then
    echo "FAILED: Port checks for 80/443 not found"
    exit 1
fi

echo "Pre-flight tests passed!"