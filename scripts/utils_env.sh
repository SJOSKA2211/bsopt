#!/bin/bash
# scripts/utils_env.sh - Institutional Environment & Secret Substrate
set -euo pipefail

# Function to load decrypted secrets from the hardened PKI vault
load_decrypted_secrets() {
    local ENV_FILE=".env"
    local vault_key=".pki/vault/vault.key"
    
    if [ ! -f "$ENV_FILE" ]; then
        echo "🟠 No .env file found. Proceeding with system environment."
        return 0
    fi
    
    # Standard .env export (handling basic key=value)
    export $(grep -v '^#' "$ENV_FILE" | xargs)

    if [ ! -f "$vault_key" ]; then
        return 0
    fi

    echo "🔐 Decrypting Institutional Secrets..."
    # Logic to decrypt ENC_ variables if they exist
    while IFS= read -r line; do
        [[ "$line" =~ ^#.*$ ]] && continue
        [[ -z "$line" ]] && continue
        if [[ $line =~ ^ENC_([A-Z0-9_]+)=\"(.+)\"$ ]]; then
            local var_name="${BASH_REMATCH[1]}"
            local encrypted_val="${BASH_REMATCH[2]}"
            local decrypted_val=$(echo -n "$encrypted_val" | base64 -d | openssl pkeyutl -decrypt -inkey "$vault_key" 2>/dev/null)
            if [ -n "$decrypted_val" ]; then
                export "$var_name"="$decrypted_val"
            fi
        fi
    done < "$ENV_FILE"
}

# Detect container engine with institutional precision
detect_container_engine() {
    if command -v podman >/dev/null 2>&1; then
        export CONTAINER_ENGINE="podman"
        export COMPOSE_ENGINE="podman compose"
    elif command -v docker >/dev/null 2>&1; then
        export CONTAINER_ENGINE="docker"
        if docker compose version >/dev/null 2>&1; then
            export COMPOSE_ENGINE="docker compose"
        else
            export COMPOSE_ENGINE="docker-compose"
        fi
    else
        echo "❌ Error: Institutional container engine not detected."
        exit 1
    fi
}
