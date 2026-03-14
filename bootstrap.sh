#!/bin/bash
# ==============================================================================
# EQUAFLOW: THE ZERO-TOUCH BOOTSTRAP (v2.0)
# ==============================================================================
# Automates the entire stack: Security (RSA/ECC), .env, DB Init, and Gateway.
# ==============================================================================

set -e

# Configuration
KEYS_DIR="/home/h8t3dj4y/.bsopt/pki"
ENV_FILE=".env"
ENV_EXAMPLE=".env.example"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "🚀 Starting EquaFlow Advanced Bootstrap [${TIMESTAMP}]"

# 1. Initialize Security Layer (RSA/ECC)
mkdir -p "${KEYS_DIR}"

generate_keys() {
    echo "🔐 Generating Asymmetric Key Pairs (Institutional Grade)..."
    
    # RSA 4096 (RS256)
    if [ ! -f "${KEYS_DIR}/jwt_rs256.key" ]; then
        openssl genrsa -out "${KEYS_DIR}/jwt_rs256.key" 4096
        openssl rsa -in "${KEYS_DIR}/jwt_rs256.key" -pubout -out "${KEYS_DIR}/jwt_rs256.pub"
    fi

    # ECC P-256 (ES256)
    if [ ! -f "${KEYS_DIR}/jwt_es256.key" ]; then
        openssl ecparam -name prime256v1 -genkey -noout -out "${KEYS_DIR}/jwt_es256.key"
        openssl ec -in "${KEYS_DIR}/jwt_es256.key" -pubout -out "${KEYS_DIR}/jwt_es256.pub"
    fi

    # TOTP Master Secret
    if [ ! -f "${KEYS_DIR}/totp_master.secret" ]; then
        openssl rand -hex 32 > "${KEYS_DIR}/totp_master.secret"
    fi
    
    # Envoy SSL Cert (Self-Signed for Dev Edge)
    if [ ! -f "${KEYS_DIR}/envoy_edge.key" ]; then
        openssl req -x509 -newkey rsa:4096 -keyout "${KEYS_DIR}/envoy_edge.key" -out "${KEYS_DIR}/envoy_edge.crt" \
            -days 365 -nodes -subj "/C=US/ST=State/L=City/O=BSOPT/CN=localhost"
    fi
}

generate_keys

# 2. .env Orchestration
if [ ! -f "${ENV_FILE}" ]; then
    echo "📄 Creating .env from template..."
    cp "${ENV_EXAMPLE}" "${ENV_FILE}"
fi

set_env_var() {
    local key=$1
    local value=$2
    # Escape for sed and handle multiline if needed
    if grep -q "^${key}=" "${ENV_FILE}"; then
        sed -i "s|^${key}=.*|${key}=\"${value}\"|g" "${ENV_FILE}"
    else
        echo "${key}=\"${value}\"" >> "${ENV_FILE}"
    fi
}

# Explicitly store raw keys in .env (Base64 for single-line safety)
echo "🔑 Explicitly injecting keys into ${ENV_FILE}..."
RS256_PRIV=$(cat "${KEYS_DIR}/jwt_rs256.key" | base64 -w 0)
RS256_PUB=$(cat "${KEYS_DIR}/jwt_rs256.pub" | base64 -w 0)
ES256_PRIV=$(cat "${KEYS_DIR}/jwt_es256.key" | base64 -w 0)
ES256_PUB=$(cat "${KEYS_DIR}/jwt_es256.pub" | base64 -w 0)
TOTP_MASTER=$(cat "${KEYS_DIR}/totp_master.secret")

set_env_var "JWT_RS256_PRIVATE" "${RS256_PRIV}"
set_env_var "JWT_RS256_PUBLIC" "${RS256_PUB}"
set_env_var "JWT_ES256_PRIVATE" "${ES256_PRIV}"
set_env_var "JWT_ES256_PUBLIC" "${ES256_PUB}"
set_env_var "MFA_TOTP_SECRET" "${TOTP_MASTER}"

# Secure random passwords
for var in POSTGRES_PASSWORD REDIS_PASSWORD BETTER_AUTH_SECRET JWT_SECRET; do
    if ! grep -q "^${var}=" "${ENV_FILE}" || [[ -z $(grep "^${var}=" "${ENV_FILE}" | cut -d'=' -f2) ]]; then
        set_env_var "${var}" "$(openssl rand -hex 16)"
    fi
done

# 3. PostgreSQL Automation
echo "🐘 Preparing PostgreSQL initialization..."
# Parse .env to get credentials for container health checks or initial DDL
PG_PASS=$(grep "^POSTGRES_PASSWORD=" "${ENV_FILE}" | cut -d'=' -f2 | tr -d '"')
set_env_var "DATABASE_URL" "postgresql://admin:${PG_PASS}@pgbouncer:6432/bsopt"

# 4. Success Marker
echo "✅ EquaFlow Stack Bootstrapped Successfully."
echo "🐳 Run 'make up' to start the manifold."
