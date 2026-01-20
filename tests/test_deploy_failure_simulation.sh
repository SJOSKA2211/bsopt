#!/usr/bin/env bash
# tests/test_deploy_failure_simulation.sh

set -euo pipefail

DEPLOY_SCRIPT="./deploy.sh"

echo "Testing automated rollback on failure..."

# 1. Create a successful checkpoint first
./deploy.sh backup > /dev/null 2>&1

# 2. Inject a failure by making docker command fail
# We can do this by prepending a PATH that has a fake docker
mkdir -p .tmp_bin
cat > .tmp_bin/docker <<EOF
#!/usr/bin/env bash
echo "Mocking docker failure"
exit 1
EOF
chmod +x .tmp_bin/docker

# Run deploy with the fake docker in PATH
OUTPUT=$(PATH="$(pwd)/.tmp_bin:$PATH" "$DEPLOY_SCRIPT" deploy 2>&1 || true)

if echo "$OUTPUT" | grep -q "Initiating rollback"; then
    echo "SUCCESS: Automated rollback triggered"
else
    echo "FAILED: Automated rollback not triggered"
    echo "Output: $OUTPUT"
    rm -rf .tmp_bin
    exit 1
fi

rm -rf .tmp_bin
echo "Failure simulation passed!"
