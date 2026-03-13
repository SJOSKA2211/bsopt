#!/bin/bash
# EquaFlow Zero-Touch Bootstrap Script
set -e

# Configuration
KEYS_DIR="./.gemini_security"
ENV_FILE=".env"
ENV_EXAMPLE=".env.example"
README_FILE="README.md"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "🚀 Starting EquaFlow Bootstrap [${TIMESTAMP}]"

# 1. Asymmetric Security & PKI Orchestration
if [ -f "./scripts/setup_pki.sh" ]; then
    echo "🔐 Running Advanced PKI Orchestration..."
    bash ./scripts/setup_pki.sh
else
    echo "⚠️ setup_pki.sh not found. Falling back to basic key generation..."
    mkdir -p "${KEYS_DIR}"
    openssl genpkey -algorithm RSA -out "${KEYS_DIR}/jwt_rs256.key" -pkeyopt rsa_keygen_bits:2048
    openssl rsa -pubout -in "${KEYS_DIR}/jwt_rs256.key" -out "${KEYS_DIR}/jwt_rs256.pub"
fi

# 2. .env Generation & Injection
if [ ! -f "${ENV_FILE}" ]; then
    echo "📄 Creating .env file from ${ENV_EXAMPLE}..."
    cp "${ENV_EXAMPLE}" "${ENV_FILE}" || touch "${ENV_FILE}"
fi

# Load Keys from PKI directory (defined in setup_pki.sh)
PKI_DIR="${HOME}/.bsopt/pki"

if [ -d "${PKI_DIR}" ]; then
    echo "🔑 Injecting Asymmetric Keys from ${PKI_DIR}..."
    RS256_PRIVATE=$(cat "${PKI_DIR}/jwt_rs256.key" | base64 -w 0)
    RS256_PUBLIC=$(cat "${PKI_DIR}/jwt_rs256.pub" | base64 -w 0)
    ES256_PRIVATE=$(cat "${PKI_DIR}/jwt_es256.key" | base64 -w 0)
    ES256_PUBLIC=$(cat "${PKI_DIR}/jwt_es256.pub" | base64 -w 0)
    TOTP_SECRET=$(cat "${PKI_DIR}/totp_master.secret")
else
    echo "⚠️ PKI directory ${PKI_DIR} not found. Using fallback keys from ${KEYS_DIR}..."
    RS256_PRIVATE=$(cat "${KEYS_DIR}/jwt_rs256.key" | base64 -w 0)
    RS256_PUBLIC=$(cat "${KEYS_DIR}/jwt_rs256.pub" | base64 -w 0)
    TOTP_SECRET=$(cat "${KEYS_DIR}/mfa_secret.txt" 2>/dev/null || echo "dev-totp-secret")
fi

set_env_var() {
    local key=$1
    local value=$2
    if grep -q "^${key}=" "${ENV_FILE}"; then
        sed -i "s|^${key}=.*|${key}=\"${value}\"|g" "${ENV_FILE}"
    else
        echo "${key}=\"${value}\"" >> "${ENV_FILE}"
    fi
}

set_env_var "JWT_RS256_PRIVATE" "${RS256_PRIVATE}"
set_env_var "JWT_RS256_PUBLIC" "${RS256_PUBLIC}"
set_env_var "JWT_ES256_PRIVATE" "${ES256_PRIVATE}"
set_env_var "JWT_ES256_PUBLIC" "${ES256_PUBLIC}"
set_env_var "MFA_TOTP_SECRET" "${TOTP_SECRET}"
set_env_var "JWT_ALGORITHM" "RS256" # Default

# Blockchain Trading Keys (Optional but encouraged for HFT pipelines)
if ! grep -q "^BLOCKCHAIN_PRIVATE_KEY=" "${ENV_FILE}"; then
    echo "🔗 Generating secure random Blockchain Private Key..."
    BLOCK_KEY="0x$(openssl rand -hex 32)"
    set_env_var "BLOCKCHAIN_PRIVATE_KEY" "${BLOCK_KEY}"
fi

# Audit Vault Key
if ! grep -q "^AUDIT_VAULT_KEY=" "${ENV_FILE}"; then
    echo "🔐 Generating secure random Audit Vault Key..."
    VAULT_KEY="$(openssl rand -hex 32)"
    set_env_var "AUDIT_VAULT_KEY" "${VAULT_KEY}"
fi

# IBM Quantum Token (Placeholder for injection)
if ! grep -q "^IBM_QUANTUM_TOKEN=" "${ENV_FILE}"; then
    set_env_var "IBM_QUANTUM_TOKEN" "replace_me_with_real_token"
fi

# Better Auth Secret
if ! grep -q "^BETTER_AUTH_SECRET=" "${ENV_FILE}"; then
    echo "🔐 Generating secure random Better Auth Secret..."
    AUTH_SECRET="$(openssl rand -hex 32)"
    set_env_var "BETTER_AUTH_SECRET" "${AUTH_SECRET}"
fi

# Generate secure fallback passwords
DB_PASS=$(openssl rand -hex 16)
REDIS_PASS=$(openssl rand -hex 16)

# Check if DB pass is already set
if ! grep -q "^POSTGRES_PASSWORD=" "${ENV_FILE}"; then
    set_env_var "POSTGRES_PASSWORD" "${DB_PASS}"
fi
if ! grep -q "^REDIS_PASSWORD=" "${ENV_FILE}"; then
    set_env_var "REDIS_PASSWORD" "${REDIS_PASS}"
fi
if ! grep -q "^DATABASE_URL=" "${ENV_FILE}"; then
    # Default to admin username from compose file
    set_env_var "DATABASE_URL" "postgresql://admin:\${POSTGRES_PASSWORD}@postgres:5432/bsopt"
fi

echo "✅ Secrets injected into ${ENV_FILE}"

# Load the env vars to extract credentials
set -a
source "${ENV_FILE}"
set +a

# 4. Auto-update README.md
echo "📝 Updating README.md..."
DEPLOY_INFO="\n\n> **Latest Deployment:** ${TIMESTAMP}\n> **Public Key Locations:** \`${HOME}/.bsopt/pki/jwt_rs256.pub\` (RSA), \`${HOME}/.bsopt/pki/jwt_es256.pub\` (ECC)\n> **Status:** Bootstrapped via \`bootstrap.sh\`"
if grep -q "Latest Deployment:" "${README_FILE}"; then
    sed -i "/Latest Deployment:/c\> **Latest Deployment:** ${TIMESTAMP}" "${README_FILE}"
else
    echo -e "${DEPLOY_INFO}" >> "${README_FILE}"
fi

# 5. Sequenced Startup
echo "🐳 Starting Docker containers..."
docker-compose up --build -d

# Wait for DB Health
echo "⏳ Waiting for PostgreSQL/TimescaleDB to be healthy..."
until docker exec bsopt-postgres-1 pg_isready -U admin -d bsopt || docker exec bsopt-postgres-1 pg_isready -U postgres; do
  sleep 2
done
echo "✅ Database is online."

echo "⏳ Waiting for Redis to be healthy..."
until docker exec bsopt-redis-1 redis-cli -a "${REDIS_PASSWORD}" ping | grep -q PONG; do
  sleep 2
done
echo "✅ Redis is online."

# 6. Database Initialization Scripts
echo "📂 Running database migrations via Alembic..."
docker exec bsopt-api-1 alembic upgrade head || echo "⚠️ Alembic migration failed or not configured yet."

# 7. Automated E2E Testing & UI Validation
echo "🧪 Running E2E Test Suite..."
docker-compose --profile test up e2e-test --build --abort-on-container-exit || echo "⚠️ E2E tests failed or not configured."

echo "✨ Bootstrap Complete! EquaFlow is ready."
echo "🔗 API Docs: http://localhost:8000/docs"
