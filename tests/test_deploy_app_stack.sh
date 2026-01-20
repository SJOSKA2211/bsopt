#!/usr/bin/env bash
# tests/test_deploy_app_stack.sh

set -euo pipefail

DEPLOY_SCRIPT="./deploy.sh"

export TESTING=true
source "$DEPLOY_SCRIPT"

test_deploy_tier2() {
    echo "Testing deploy_tier2 (presence)..."
    if grep -q "deploy_tier2()" "$DEPLOY_SCRIPT"; then
        echo "SUCCESS: deploy_tier2 function exists"
    else
        echo "FAILED: deploy_tier2 function missing"
        exit 1
    fi
}

test_deploy_tier3() {
    echo "Testing deploy_tier3 (presence)..."
    if grep -q "deploy_tier3()" "$DEPLOY_SCRIPT"; then
        echo "SUCCESS: deploy_tier3 function exists"
    else
        echo "FAILED: deploy_tier3 function missing"
        exit 1
    fi
}

test_deploy_tier2
test_deploy_tier3

echo "App stack tests passed!"
