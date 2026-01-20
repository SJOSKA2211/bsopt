#!/usr/bin/env bash
# tests/test_deploy_env.sh

set -euo pipefail

DEPLOY_SCRIPT="./deploy.sh"
TEST_ENV=".env.test"

# Clean up
rm -f "$TEST_ENV"

echo "Testing secret generation..."
# Create a dummy .env with "changeme" values
echo "POSTGRES_PASSWORD=changeme" > .env.test
# Temporarily point ENV_FILE to .env.test
# This is tricky without modifying deploy.sh to accept an env file path
# Let's just run it and check our real .env (backing it up first)

if [[ -f ".env" ]]; then
    mv .env .env.bak
fi

"./deploy.sh" setup-env > /dev/null 2>&1

if [[ ! -f ".env" ]]; then
    echo "FAILED: .env not created"
    [[ -f ".env.bak" ]] && mv .env.bak .env
    exit 1
fi

if grep -q "POSTGRES_PASSWORD=changeme" ".env"; then
    echo "FAILED: POSTGRES_PASSWORD still 'changeme'"
    [[ -f ".env.bak" ]] && mv .env.bak .env
    exit 1
fi

echo "SUCCESS: Secrets generated correctly"

# Restore backup
[[ -f ".env.bak" ]] && mv .env.bak .env

