#!/usr/bin/env bash
# scripts/bootstrap.sh
set -euo pipefail

# 1. Detect Container Engine
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
    COMPOSE_CMD="podman-compose"
elif command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
    if $CONTAINER_CMD compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
else
    echo "[!] Error: Neither docker nor podman is installed."
    exit 1
fi

echo "[*] Using container orchestrator: $COMPOSE_CMD"

# 2. Generate ECC Asymmetric Keys and Database Secrets
ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    echo "[*] Generating ECC Asymmetric Key Pairs and Secrets..."
    
    # Generate SECP256R1 (prime256v1) key pair
    openssl ecparam -genkey -name prime256v1 -noout -out jwt_private.pem 2>/dev/null
    openssl ec -in jwt_private.pem -pubout -out jwt_public.pem 2>/dev/null

    # Format keys for environment variable injection (replace newlines with \n)
    JWT_PRIV=$(awk 'NF {sub(/\r/, ""); printf "%s\\n",$0;}' jwt_private.pem)
    JWT_PUB=$(awk 'NF {sub(/\r/, ""); printf "%s\\n",$0;}' jwt_public.pem)

    # Generate secure 32-byte hex for Postgres
    DB_USER="equaflow_admin"
    DB_PASS=$(openssl rand -hex 32)
    DB_NAME="equaflow_db"

    cat <<EOF > "$ENV_FILE"
POSTGRES_USER=$DB_USER
POSTGRES_PASSWORD=$DB_PASS
POSTGRES_DB=$DB_NAME
DATABASE_URL=postgresql://$DB_USER:$DB_PASS@timescaledb:5432/$DB_NAME
JWT_PRIVATE_KEY="$JWT_PRIV"
JWT_PUBLIC_KEY="$JWT_PUB"
EOF
    echo "[*] Secrets generated and locked in $ENV_FILE"
    rm -f jwt_private.pem jwt_public.pem
else
    echo "[*] $ENV_FILE already exists. Utilizing existing secure state."
fi

# Load environment variables
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

# 3. Spin up TimescaleDB
echo "[*] Initializing TimescaleDB Live Environment..."
$COMPOSE_CMD -f infrastructure/docker-compose.yml up -d timescaledb

# 4. Strict Polling Loop for DB Health
echo "[*] Executing pg_isready polling loop..."
RETRIES=30
# We use the container name 'timescaledb' directly if we are on the same network or engine
until $CONTAINER_CMD exec timescaledb pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" > /dev/null 2>&1 || [ $RETRIES -eq 0 ]; do
    echo "    -> Waiting for DB connection... $((RETRIES--)) attempts remaining..."
    sleep 2
done

if [ $RETRIES -eq 0 ]; then
    echo "[!] Fatal: TimescaleDB failed to reach readiness. Tailing logs:"
    $CONTAINER_CMD logs timescaledb
    exit 1
fi
echo "[+] TimescaleDB is actively accepting connections."

# 5. Spin up Envoy Gateway
echo "[*] Initializing Envoy API Gateway..."
$COMPOSE_CMD -f infrastructure/docker-compose.yml up -d envoy

echo "[+] Phase 0 Bootstrapping Complete. Stack is healthy."
