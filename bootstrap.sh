#!/bin/bash
# ==============================================================================
# EQUAFLOW: THE ZERO-TOUCH BOOTSTRAP (v2.1)
# ==============================================================================
# Automates the entire stack: Security (RSA/ECC), .env, DB Init, and Gateway.
# ==============================================================================

set -e

# Configuration
KEYS_DIR="$(pwd)/.pki"
ENV_FILE=".env"
ENV_EXAMPLE=".env.example"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "🚀 Starting EquaFlow Advanced Bootstrap [${TIMESTAMP}]"

# 0. Prerequisite Check
check_prereq() {
    local cmd=$1
    if ! command -v "$cmd" &> /dev/null; then
        echo "❌ Error: $cmd is not installed."
        exit 1
    fi
}

check_prereq openssl
check_prereq docker
if ! docker compose version &> /dev/null && ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: docker compose is not installed."
    exit 1
fi

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
    if [ -f "${ENV_EXAMPLE}" ]; then
        cp "${ENV_EXAMPLE}" "${ENV_FILE}"
    else
        touch "${ENV_FILE}"
    fi
fi

set_env_var() {
    local key=$1
    local value=$2
    # Ensure value is properly escaped for sed if it contains special characters
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

# Secure random passwords for necessary variables
for var in POSTGRES_PASSWORD REDIS_PASSWORD BETTER_AUTH_SECRET JWT_SECRET; do
    if ! grep -q "^${var}=" "${ENV_FILE}" || [[ -z $(grep "^${var}=" "${ENV_FILE}" | cut -d'=' -f2 | tr -d '"') ]]; then
        set_env_var "${var}" "$(openssl rand -hex 16)"
    fi
done

# 3. PostgreSQL Automation & Secret Orchestration
echo "🐘 Preparing PostgreSQL initialization..."
PG_PASS=$(grep "^POSTGRES_PASSWORD=" "${ENV_FILE}" | cut -d'=' -f2 | tr -d '"')
set_env_var "DATABASE_URL" "postgresql://admin:${PG_PASS}@pgbouncer:6432/bsopt"
set_env_var "DATABASE_URL_LOCAL" "postgresql://admin:${PG_PASS}@localhost:5434/bsopt"
set_env_var "DATABASE_URL_TEST" "postgresql://admin:${PG_PASS}@postgres:5432/bsopt_test"
set_env_var "MLFLOW_BACKEND_STORE_URI" "postgresql://admin:${PG_PASS}@postgres:5432/bsopt"
set_env_var "POSTGRES_PASSWORD" "${PG_PASS}"

# 4. Success Marker
echo "✅ EquaFlow Stack Bootstrapped Successfully."
echo "🛠️  Next Steps:"
echo "   1. run 'make build && make up' to launch the manifold"
echo "   2. run 'make test-all' to verify the gauntlet"
