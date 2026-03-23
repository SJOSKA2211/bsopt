#!/bin/bash
# EquaFlow Environment Utilities (Hardened v3.1)

# Function to load decrypted secrets into the current session
load_decrypted_secrets() {
    local ENV_FILE=".env"
    local vault_key=".pki/vault/vault.key"
    
    if [ ! -f "$ENV_FILE" ]; then
        return 0
    fi
    
    if [ ! -f "$vault_key" ]; then
        echo "[WARN] Vault key not found at $vault_key. Encrypted secrets will not be loaded."
        return 0
    fi

    # Detect openssl command
    local OPENSSL_CMD="openssl"
    if ! command -v openssl >/dev/null 2>&1; then
        echo "[ERROR] OpenSSL not found. Cannot decrypt secrets."
        return 1
    fi

    # Decrypt all ENC_ variables and export them
    while IFS= read -r line; do
        if [[ $line =~ ^ENC_([A-Z0-9_]+)=\"(.+)\"$ ]]; then
            local var_name="${BASH_REMATCH[1]}"
            local encrypted_val="${BASH_REMATCH[2]}"
            
            # Decrypt using the vault key
            local decrypted_val=$(echo -n "$encrypted_val" | base64 -d | $OPENSSL_CMD pkeyutl -decrypt -inkey "$vault_key" 2>/dev/null)
            
            if [ -n "$decrypted_val" ]; then
                export "$var_name"="$decrypted_val"
            else
                echo "[WARN] Failed to decrypt $var_name"
            fi
        fi
    done < "$ENV_FILE"
}

# Detect container engine (podman/docker)
detect_container_engine() {
    if command -v podman >/dev/null 2>&1; then
        CONTAINER_ENGINE="podman"
        if podman compose version >/dev/null 2>&1; then
            COMPOSE_ENGINE="podman compose"
        else
            COMPOSE_ENGINE="podman-compose"
        fi
    elif command -v docker >/dev/null 2>&1; then
        CONTAINER_ENGINE="docker"
        if docker compose version >/dev/null 2>&1; then
            COMPOSE_ENGINE="docker compose"
        else
            COMPOSE_ENGINE="docker-compose"
        fi
    else
        echo "[ERROR] No container engine (podman/docker) found."
        exit 1
    fi
    export CONTAINER_ENGINE
    export COMPOSE_ENGINE
}

# Wrapper for container compose that ensures secrets are loaded
compose_cmd() {
    if [ -z "$COMPOSE_ENGINE" ]; then
        detect_container_engine
    fi
    load_decrypted_secrets
    
    local ENV_FILE=".env"
    if [ -f "$ENV_FILE" ]; then
        $COMPOSE_ENGINE --env-file "$ENV_FILE" "$@"
    else
        $COMPOSE_ENGINE "$@"
    fi
}
