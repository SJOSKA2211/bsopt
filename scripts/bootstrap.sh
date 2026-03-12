#!/usr/bin/env bash

# ==============================================================================
# BS-OPT: COMPLETE STACK AUTOMATION & SECURITY BOOTSTRAPPING
# ==============================================================================
# This script automates the setup of the entire stack and initializes secure
# cryptographically generated secrets.
# ==============================================================================

set -e

echo "========================================================="
echo "   Initiating High-Performance Engine Bootstrapping      "
echo "========================================================="

ENV_FILE=".env"
ENV_TEST_FILE=".env.test"

# Function to generate secure hex keys
generate_secret() {
    openssl rand -hex 32
}

echo "[*] Ensuring environment files exist..."

if [ ! -f "$ENV_FILE" ]; then
    echo "[+] Generating production $ENV_FILE..."
    JWT_SECRET=$(generate_secret)
    MFA_SECRET=$(generate_secret)
    DB_PASSWORD=$(openssl rand -hex 16)
    
    cat <<EOF > $ENV_FILE
# Security
JWT_SECRET_KEY=$JWT_SECRET
MFA_ENCRYPTION_KEY=$MFA_SECRET
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
POSTGRES_USER=admin
POSTGRES_PASSWORD=$DB_PASSWORD
POSTGRES_DB=bsopt
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
DATABASE_URL=postgresql+asyncpg://admin:$DB_PASSWORD@postgres:5432/bsopt

# Environment
ENVIRONMENT=dev

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100

# Monitoring
PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc_dir
EOF
    echo "[+] Production secrets generated and saved."
else
    echo "[-] $ENV_FILE already exists, skipping secret generation."
fi

if [ ! -f "$ENV_TEST_FILE" ]; then
    echo "[+] Generating test $ENV_TEST_FILE..."
    JWT_SECRET=$(generate_secret)
    MFA_SECRET=$(generate_secret)
    
    cat <<EOF > $ENV_TEST_FILE
# Security
JWT_SECRET_KEY=$JWT_SECRET
MFA_ENCRYPTION_KEY=$MFA_SECRET
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
POSTGRES_USER=admin
POSTGRES_PASSWORD=password
POSTGRES_DB=bsopt_test
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
DATABASE_URL_TEST=postgresql+asyncpg://admin:password@postgres:5432/bsopt_test

# Environment
ENVIRONMENT=test
EOF
    echo "[+] Test secrets generated and saved."
else
    echo "[-] $ENV_TEST_FILE already exists, skipping secret generation."
fi

# Ensure multiproc directory for Prometheus exists
mkdir -p /tmp/prometheus_multiproc_dir

echo "========================================================="
echo "   Bootstrapping Complete. Ready for Ignition.           "
echo "========================================================="
echo "To start development stack: docker compose -f docker-compose.dev.yml up -d"
echo "To start production stack: docker compose -f docker-compose.yml up -d"
