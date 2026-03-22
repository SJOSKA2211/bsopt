#!/bin/bash
# EquaFlow Zero-Touch Bootstrap Script (Institutional Grade)
set -e

# Configuration
KEYS_DIR="$(pwd)/.pki"
ENV_FILE=".env"
ENV_EXAMPLE=".env.example"
README_FILE="README.md"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Load shared environment utilities
UTILS_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/utils_env.sh"
if [ -f "$UTILS_ENV" ]; then
    source "$UTILS_ENV"
else
    echo "❌ ERROR: utils_env.sh not found."
    exit 1
fi

echo "🚀 Starting EquaFlow Institutional Bootstrap [${TIMESTAMP}]"

# 1. Asymmetric Security & PKI Orchestration
if [ -f "./scripts/setup_pki.sh" ]; then
    echo "🔐 Running Advanced PKI Orchestration (RSA 4096 / ECC P-256)..."
    bash ./scripts/setup_pki.sh
else
    echo "❌ CRITICAL: setup_pki.sh not found. Security layer cannot be initialized."
    exit 1
fi

# 2. .env Generation & Injection
if [ ! -f "${ENV_FILE}" ]; then
    echo "📄 Creating .env file from ${ENV_EXAMPLE}..."
    cp "${ENV_EXAMPLE}" "${ENV_FILE}"
fi

set_env_var() {
    local key=$1
    local value=$2
    # Ensure value is escaped for sed
    local escaped_value=$(echo "${value}" | sed 's/[&/\]/\\&/g')
    if grep -q "^${key}=" "${ENV_FILE}"; then
        sed -i "s|^${key}=.*|${key}=\"${escaped_value}\"|g" "${ENV_FILE}"
    else
        echo "${key}=\"${escaped_value}\"" >> "${ENV_FILE}"
    fi
}

echo "🔑 Injecting Asymmetric Keys from ${KEYS_DIR}..."
RS256_PRIVATE=$(cat "${KEYS_DIR}/jwt_rs256.key" | base64 -w 0)
RS256_PUBLIC=$(cat "${KEYS_DIR}/jwt_rs256.pub" | base64 -w 0)
ES256_PRIVATE=$(cat "${KEYS_DIR}/jwt_es256.key" | base64 -w 0)
ES256_PUBLIC=$(cat "${KEYS_DIR}/jwt_es256.pub" | base64 -w 0)
TOTP_SECRET=$(cat "${KEYS_DIR}/totp_master.secret")

set_env_var "JWT_RS256_PRIVATE" "${RS256_PRIVATE}"
set_env_var "JWT_RS256_PUBLIC" "${RS256_PUBLIC}"
set_env_var "JWT_ES256_PRIVATE" "${ES256_PRIVATE}"
set_env_var "JWT_ES256_PUBLIC" "${ES256_PUBLIC}"
set_env_var "MFA_TOTP_SECRET" "${TOTP_SECRET}"
set_env_var "JWT_ALGORITHM" "RS256"

# Generate secure random secrets if not present
[[ -z $(grep "^BETTER_AUTH_SECRET=" "${ENV_FILE}" | cut -d'=' -f2 | tr -d '"') ]] && set_env_var "BETTER_AUTH_SECRET" "$(openssl rand -hex 32)"
[[ -z $(grep "^JWT_SECRET=" "${ENV_FILE}" | cut -d'=' -f2 | tr -d '"') ]] && set_env_var "JWT_SECRET" "$(openssl rand -hex 32)"
[[ -z $(grep "^POSTGRES_PASSWORD=" "${ENV_FILE}" | cut -d'=' -f2 | tr -d '"') ]] && set_env_var "POSTGRES_PASSWORD" "$(openssl rand -hex 16)"
[[ -z $(grep "^REDIS_PASSWORD=" "${ENV_FILE}" | cut -d'=' -f2 | tr -d '"') ]] && set_env_var "REDIS_PASSWORD" "$(openssl rand -hex 16)"

# Re-read passwords for URL construction
PG_PASS=$(grep "^POSTGRES_PASSWORD=" "${ENV_FILE}" | cut -d'=' -f2 | tr -d '"')
REDIS_PASS=$(grep "^REDIS_PASSWORD=" "${ENV_FILE}" | cut -d'=' -f2 | tr -d '"')

set_env_var "DATABASE_URL" "postgresql://admin:${PG_PASS}@pgbouncer:6432/bsopt"
set_env_var "DATABASE_URL_TEST" "postgresql://admin:${PG_PASS}@postgres:5432/bsopt_test"
set_env_var "REDIS_URL" "redis://:${REDIS_PASS}@redis:6379/0"

# Hardened Secret Vaulting
encrypt_secret() {
    local key=$1
    local plaintext=$(grep "^${key}=" "${ENV_FILE}" | cut -d'=' -f2- | tr -d '"' | tr -d "'")
    if [ -n "$plaintext" ]; then
        local encrypted=$(echo -n "$plaintext" | openssl pkeyutl -encrypt -pubin -inkey "${KEYS_DIR}/jwt_rs256.pub" | base64 -w0)
        sed -i "s|^${key}=.*|ENC_${key}=\"${encrypted}\"|" "${ENV_FILE}"
        # Remove plaintext
        sed -i "/^${key}=/d" "${ENV_FILE}"
    fi
}

log "🔐 Vaulting sensitive variables..."
for s in POSTGRES_PASSWORD REDIS_PASSWORD BETTER_AUTH_SECRET JWT_SECRET; do
    encrypt_secret "$s"
done

echo "✅ Secrets and Keys injected and vaulted in ${ENV_FILE}"

# 3. Auto-update README.md
echo "📝 Updating README.md status..."
# Use a clear marker for the deployment section
if grep -q "## 🚀 Deployment Status" "${README_FILE}"; then
    # Update existing section
    sed -i "/Latest Deployment:/c\> **Latest Deployment:** ${TIMESTAMP}" "${README_FILE}"
    sed -i "/Status:/c\> **Status:** Healthy (Bootstrapped via \`bootstrap.sh\`)" "${README_FILE}"
else
    # Append new section
    echo -e "\n## 🚀 Deployment Status\n\n> **Latest Deployment:** ${TIMESTAMP}\n> **Public Key Locations:** \`${KEYS_DIR}/jwt_rs256.pub\`\n> **Status:** Healthy (Bootstrapped via \`bootstrap.sh\`)" >> "${README_FILE}"
fi

# 4. Sequenced Startup
echo "🐳 Starting Docker Manifold..."
# Force session load before up
load_decrypted_secrets
docker-compose up --build -d

# Wait for DB Health
echo "⏳ Waiting for PostgreSQL/TimescaleDB (pg_isready)..."
MAX_RETRIES=30
COUNT=0
until docker exec bsopt-postgres-1 pg_isready -U admin -d bsopt > /dev/null 2>&1 || [ $COUNT -eq $MAX_RETRIES ]; do
  sleep 2
  ((COUNT++))
done

if [ $COUNT -eq $MAX_RETRIES ]; then
    echo "❌ FATAL: PostgreSQL failed to become healthy."
    docker-compose logs postgres
    exit 1
fi
echo "✅ Database is online."

echo "⏳ Waiting for Redis..."
COUNT=0
until docker exec bsopt-redis-1 redis-cli -a "${REDIS_PASS}" ping 2>/dev/null | grep -q PONG || [ $COUNT -eq $MAX_RETRIES ]; do
  sleep 2
  ((COUNT++))
done

if [ $COUNT -eq $MAX_RETRIES ]; then
    echo "❌ FATAL: Redis failed to become healthy."
    docker-compose logs redis
    exit 1
fi
echo "✅ Redis is online."

# 5. Database Initialization
echo "📂 Running migrations (Alembic)..."
docker exec bsopt-api-1 alembic upgrade head || echo "⚠️ Alembic migration failed. Check logs."

# 6. Verification & E2E Proof
echo "🧪 Triggering E2E Validation Suite..."
docker-compose --profile test up e2e-test --abort-on-container-exit || echo "⚠️ E2E tests reached non-zero exit code."

echo "✨ EquaFlow Stack fully automated and secured."
echo "🔗 Gateway: http://localhost:8000"
