#!/usr/bin/env bash
# tests/test_deploy_environment.sh

set -euo pipefail

DEPLOY_SCRIPT="./deploy.sh"
ENV_FILE=".env"
CONFIG_DIR="monitoring"

export TESTING=true
source "$DEPLOY_SCRIPT"

test_generate_secrets() {
    echo "Testing generate_secrets..."
    # Ensure env file doesn't exist
    rm -f "$ENV_FILE"
    
    if generate_secrets; then
        if [[ -f "$ENV_FILE" ]]; then
            echo "SUCCESS: .env file generated"
            # Check if critical secrets are present
            if grep -q "JWT_SECRET=" "$ENV_FILE"; then
                echo "SUCCESS: JWT_SECRET found"
            else
                echo "FAILED: JWT_SECRET missing"
                exit 1
            fi
        else
            echo "FAILED: .env file not created"
            exit 1
        fi
    else
        echo "FAILED: generate_secrets command failed"
        exit 1
    fi
}

test_scaffold_configs() {
    echo "Testing scaffold_configs..."
    # Ensure directories don't exist
    rm -rf "$CONFIG_DIR"
    
    if scaffold_configs; then
        if [[ -d "$CONFIG_DIR/prometheus" ]] && [[ -f "$CONFIG_DIR/prometheus/prometheus.yml" ]]; then
            echo "SUCCESS: Prometheus config scaffolded"
        else
            echo "FAILED: Prometheus config missing"
            exit 1
        fi
    else
        echo "FAILED: scaffold_configs command failed"
        exit 1
    fi
}

test_generate_secrets
test_scaffold_configs

echo "Environment tests passed!"
