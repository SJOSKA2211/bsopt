#!/bin/bash
# Manifold: Environment & Secret Management Utilities (Hardened v1.1)

detect_container_engine() {
    if command -v podman &> /dev/null; then
        export CONTAINER_ENGINE="podman"
        export COMPOSE_ENGINE="podman-compose"
    elif command -v docker &> /dev/null; then
        export CONTAINER_ENGINE="docker"
        if docker compose version &> /dev/null; then
            export COMPOSE_ENGINE="docker compose"
        else
            export COMPOSE_ENGINE="docker-compose"
        fi
    else
        echo "[ERROR] Neither docker nor podman found."
        exit 1
    fi
    # log_info "Detected container engine: $CONTAINER_ENGINE using $COMPOSE_ENGINE"
}

load_decrypted_secrets() {
    local ENV_FILE=".env"
    local KEYS_DIR=".pki"
    
    if [ ! -f "$ENV_FILE" ]; then
        return
    fi
    
    # Extract ENC_* variables
    local ENC_VARS=$(grep "^ENC_" "$ENV_FILE" | cut -d'=' -f1)
    
    for enc_key in $ENC_VARS; do
        local key=${enc_key#ENC_}
        local enc_val=$(grep "^${enc_key}=" "$ENV_FILE" | cut -d'=' -f2- | tr -d '"' | tr -d "'")
        
        if [ -n "$enc_val" ] && [ -f "${KEYS_DIR}/vault.key" ]; then
            # Decrypt value
            local dec_val=$(echo -n "$enc_val" | base64 -d | openssl pkeyutl -decrypt -inkey "${KEYS_DIR}/vault.key")
            export "$key"="$dec_val"
            # echo "[DEBUG] Decrypted $key"
        fi
    done
}
