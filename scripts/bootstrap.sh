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

# Inject Asymmetric Keys (Base64 encoded for env safety)
RS256_PRIVATE=$(cat "${HOME}/.bsopt/pki/jwt_rs256.key" | base64 -w 0)
RS256_PUBLIC=$(cat "${HOME}/.bsopt/pki/jwt_rs256.pub" | base64 -w 0)
ES256_PRIVATE=$(cat "${HOME}/.bsopt/pki/jwt_es256.key" | base64 -w 0)
ES256_PUBLIC=$(cat "${HOME}/.bsopt/pki/jwt_es256.pub" | base64 -w 0)
TOTP_SECRET=$(cat "${HOME}/.bsopt/pki/totp_master.secret")

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

echo "✨ Bootstrap Complete! EquaFlow is ready."
echo "🔗 API Docs: http://localhost:8000/docs"
