#!/usr/bin/env bash
# tests/test_deploy_verification.sh

set -euo pipefail

DEPLOY_SCRIPT="./deploy.sh"

export TESTING=true
source "$DEPLOY_SCRIPT"

test_run_smoke_tests() {
    echo "Testing run_smoke_tests (presence)..."
    if grep -q "run_smoke_tests()" "$DEPLOY_SCRIPT"; then
        echo "SUCCESS: run_smoke_tests function exists"
    else
        echo "FAILED: run_smoke_tests function missing"
        exit 1
    fi
}

test_rollback() {
    echo "Testing rollback (presence)..."
    if grep -q "rollback()" "$DEPLOY_SCRIPT"; then
        echo "SUCCESS: rollback function exists"
    else
        echo "FAILED: rollback function missing"
        exit 1
    fi
}

test_run_smoke_tests
test_rollback

echo "Verification tests passed!"
