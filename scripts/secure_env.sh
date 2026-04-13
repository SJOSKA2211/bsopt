#!/bin/bash
# Manifold: Secret Vaulting Orchestrator
# Extracts logic from bootstrap.sh to secure .env without starting containers yet.

set -e
source scripts/utils_env.sh

KEYS_DIR=".pki"
ENV_FILE=".env"
ENV_EXAMPLE=".env.example"

log_info() { echo -e "\033[0;34m[INFO]\033[0m $1"; }
log_success() { echo -e "\033[0;32m[SUCCESS]\033[0m $1"; }

setup_env_file() {
    if [ ! -f "${ENV_FILE}" ]; then
        if [ -f "${ENV_EXAMPLE}" ]; then
            cp "${ENV_EXAMPLE}" "${ENV_FILE}"
            log_success "Created .env from template"
        else
            touch "${ENV_FILE}"
        fi
    fi
}

set_env_var() {
    local key=$1
    local value=$2
    if grep -q "^${key}=" "${ENV_FILE}"; then
        sed -i "s|^${key}=.*|${key}=\"${value}\"|g" "${ENV_FILE}"
    else
        echo "${key}=\"${value}\"" >> "${ENV_FILE}"
    fi
}

encrypt_secret() {
    local val=$1
    echo -n "$val" | openssl pkeyutl -encrypt -pubin -inkey "${KEYS_DIR}/vault.pub" | base64 | tr -d '\n'
}

secure_env_file() {
    log_info "Mapping PKI keys to .env..."
    set_env_var "JWT_RS256_PRIVATE" "$(cat ${KEYS_DIR}/jwt_rs256.key | base64 | tr -d '\n')"
    set_env_var "JWT_RS256_PUBLIC" "$(cat ${KEYS_DIR}/jwt_rs256.pub | base64 | tr -d '\n')"
    set_env_var "JWT_ES256_PRIVATE" "$(cat ${KEYS_DIR}/jwt_es256.key | base64 | tr -d '\n')"
    set_env_var "JWT_ES256_PUBLIC" "$(cat ${KEYS_DIR}/jwt_es256.pub | base64 | tr -d '\n')"
    set_env_var "ARGON2_SALT" "$(cat ${KEYS_DIR}/argon2_salt.secret)"

    local SENSITIVE_VARS=("POSTGRES_PASSWORD" "REDIS_PASSWORD" "JWT_SECRET" "BETTER_AUTH_SECRET" "RABBITMQ_PASSWORD" "MINIO_ROOT_PASSWORD")
    
    for var in "${SENSITIVE_VARS[@]}"; do
        local CURRENT_VAL=$(grep "^${var}=" "${ENV_FILE}" | cut -d'=' -f2- | tr -d '"' | tr -d "'")
        
        if [ -z "$CURRENT_VAL" ] && ! grep -q "^ENC_${var}=" "${ENV_FILE}"; then
            log_info "Generating random $var..."
            local NEW_VAL=$(openssl rand -hex 32)
            set_env_var "${var}" "$NEW_VAL"
            CURRENT_VAL="$NEW_VAL"
        fi

        if [ -n "$CURRENT_VAL" ] && [[ ! "$CURRENT_VAL" =~ ^ENC_ ]]; then
            log_info "Encrypting $var..."
            local ENC_VAL=$(encrypt_secret "$CURRENT_VAL")
            set_env_var "ENC_${var}" "$ENC_VAL"
            sed -i "/^${var}=/d" "${ENV_FILE}"
        fi
    done
    log_success "Secrets vaulted securely in .env"
}

setup_env_file
secure_env_file
