#!/bin/bash
source scripts/utils_env.sh
KEYS_DIR="$(pwd)/.pki"
ENV_FILE=".env"

log_info() { echo "[INFO] $1"; }
log_success() { echo "[SUCCESS] $1"; }

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

# Ensure placeholders are gone if they were somehow reverted or missed
sed -i 's/REQUIRED_SET_BY_BOOTSTRAP/test_password_32_chars_long_1234567/g' .env

if [ ! -f "${KEYS_DIR}/vault.pub" ]; then
    ./scripts/setup_pki.sh
fi

SENSITIVE_VARS=("POSTGRES_PASSWORD" "REDIS_PASSWORD" "JWT_SECRET" "BETTER_AUTH_SECRET" "RABBITMQ_PASSWORD" "MINIO_ROOT_PASSWORD")

for var in "${SENSITIVE_VARS[@]}"; do
    if grep -q "^${var}=" "${ENV_FILE}"; then
        CURRENT_VAL=$(grep "^${var}=" "${ENV_FILE}" | cut -d'=' -f2- | tr -d '"' | tr -d "'")
        if [ -n "$CURRENT_VAL" ] && [[ ! "$CURRENT_VAL" =~ ^ENC_ ]]; then
            log_info "Encrypting $var..."
            ENC_VAL=$(encrypt_secret "$CURRENT_VAL")
            set_env_var "ENC_${var}" "$ENC_VAL"
            sed -i "/^${var}=/d" "${ENV_FILE}"
        fi
    fi
done
log_success "Secrets secured manually."
