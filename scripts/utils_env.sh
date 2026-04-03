#!/bin/bash
# scripts/utils_env.sh - Environment & Secret Substrate
set -euo pipefail

# Function to load decrypted secrets from the hardened PKI vault
load_decrypted_secrets() {
    local ENV_FILE=".env"
    local vault_key=".pki/vault.key"
    
    if [ ! -f "$ENV_FILE" ]; then
        echo "🟠 No .env file found. Proceeding with system environment."
        return 0
    fi
    
    # Standard .env export (handling basic key=value)
    export $(grep -v '^#' "$ENV_FILE" | xargs)

    if [ ! -f "$vault_key" ]; then
        return 0
    fi

    echo "🔐 Decrypting Secrets..."
    # Logic to decrypt ENC_ variables if they exist
    while IFS= read -r line; do
        [[ "$line" =~ ^#.*$ ]] && continue
        [[ -z "$line" ]] && continue
        if [[ $line =~ ^ENC_([A-Z0-9_]+)=\"(.+)\"$ ]]; then
            local var_name="${BASH_REMATCH[1]}"
            local encrypted_val="${BASH_REMATCH[2]}"
            local decrypted_val=$(echo -n "$encrypted_val" | base64 -d | openssl pkeyutl -decrypt -inkey "$vault_key" 2>/dev/null || echo "__DECRYPT_FAILED__")
            if [ "$decrypted_val" != "__DECRYPT_FAILED__" ]; then
                # Clean any trailing nulls or non-printable chars that Bash dislikes
                local clean_val=$(echo -n "$decrypted_val" | tr -d '\000-\010\013\014\016-\037')
                export "$var_name"="$clean_val"
            else
                echo "🔴 Failed to decrypt $var_name"
            fi
        fi
    done < "$ENV_FILE"
}

# Detect container engine
detect_container_engine() {
    if command -v docker >/dev/null 2>&1; then
        export CONTAINER_ENGINE="docker"
        if docker compose version >/dev/null 2>&1; then
            export COMPOSE_ENGINE="docker compose"
        else
            export COMPOSE_ENGINE="docker-compose"
        fi
    else
        echo "❌ Error: Docker engine not detected. This system requires docker."
        exit 1
    fi
}
