#!/bin/bash
#  OPTIMIZED: Automated Internal PKI for Zero-Trust mTLS & Asymmetric Security
# Generates Root CA, Server/Client Certs, JWT keys, and Vault keys.

set -e

# Standardize on .pki/ in the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY_DIR="${PROJECT_ROOT}/.pki"
mkdir -p "$KEY_DIR"
mkdir -p "$KEY_DIR/vault"

echo "🔐 Initializing Enterprise Security Layer in $KEY_DIR..."

# 1. Generate Root CA (RSA 4096)
if [[ ! -f "${KEY_DIR}/root_ca.key" ]]; then
    echo "📜 Generating Root CA (RSA 4096)..."
    openssl genrsa -out "${KEY_DIR}/root_ca.key" 4096
    openssl req -x509 -new -nodes -key "${KEY_DIR}/root_ca.key" -sha256 -days 3650 \
        -out "${KEY_DIR}/root_ca.crt" \
        -subj "/C=US/ST=State/L=City/O=BSOPT-INSTITUTIONAL/CN=EquaFlow-Internal-CA"
    chmod 600 "${KEY_DIR}/root_ca.key"
fi

# 2. Generate Asymmetric JWT Key Pairs
# RS256 (RSA 4096)
if [[ ! -f "${KEY_DIR}/jwt_rs256.key" ]]; then
    echo "🔑 Generating RS256 JWT Key Pair (RSA 4096)..."
    openssl genrsa -out "${KEY_DIR}/jwt_rs256.key" 4096
    openssl rsa -in "${KEY_DIR}/jwt_rs256.key" -pubout -out "${KEY_DIR}/jwt_rs256.pub"
    chmod 600 "${KEY_DIR}/jwt_rs256.key"
fi

# ES256 (ECC P-256)
if [[ ! -f "${KEY_DIR}/jwt_es256.key" ]]; then
    echo "🔑 Generating ES256 JWT Key Pair (ECC P-256)..."
    openssl ecparam -name prime256v1 -genkey -noout -out "${KEY_DIR}/jwt_es256.key"
    openssl ec -in "${KEY_DIR}/jwt_es256.key" -pubout -out "${KEY_DIR}/jwt_es256.pub"
    chmod 600 "${KEY_DIR}/jwt_es256.key"
fi

# 3. Generate Vault RSA 4096 Key Pair
if [[ ! -f "${KEY_DIR}/vault/vault.key" ]]; then
    echo "🔑 Generating Vault RSA 4096 Key Pair..."
    openssl genrsa -out "${KEY_DIR}/vault/vault.key" 4096
    openssl rsa -in "${KEY_DIR}/vault/vault.key" -pubout -out "${KEY_DIR}/vault/vault.pub"
    chmod 600 "${KEY_DIR}/vault/vault.key"
fi

# 4. Generate TOTP Master Secret
if [[ ! -f "${KEY_DIR}/totp_master.secret" ]]; then
    echo "🛡️ Generating TOTP Master Secret..."
    openssl rand -hex 32 > "${KEY_DIR}/totp_master.secret"
    chmod 600 "${KEY_DIR}/totp_master.secret"
fi

# 5. Generate Argon2 Salt
if [[ ! -f "${KEY_DIR}/argon2_salt.secret" ]]; then
    echo "🛡️ Generating Argon2 Salt..."
    openssl rand -hex 32 > "${KEY_DIR}/argon2_salt.secret"
    chmod 600 "${KEY_DIR}/argon2_salt.secret"
fi

# 6. Function to Issue Service Certificates (Zero-Trust mTLS)
issue_cert() {
    local service_name=$1
    local type=$2 # server or client
    
    if [[ -f "${KEY_DIR}/${service_name}.crt" ]]; then
        echo "⏭️ Certificate for $service_name already exists, skipping."
        return
    fi

    echo "📜 Issuing $type certificate for $service_name..."
    
    # Private Key
    openssl genrsa -out "${KEY_DIR}/${service_name}.key" 2048
    
    # CSR
    openssl req -new -key "${KEY_DIR}/${service_name}.key" \
        -out "${KEY_DIR}/${service_name}.csr" \
        -subj "/C=US/ST=State/L=City/O=BSOPT/CN=${service_name}"
        
    # Sign with Root CA
    openssl x509 -req -in "${KEY_DIR}/${service_name}.csr" \
        -CA "${KEY_DIR}/root_ca.crt" -CAkey "${KEY_DIR}/root_ca.key" \
        -CAcreateserial -out "${KEY_DIR}/${service_name}.crt" \
        -days 365 -sha256
        
    chmod 600 "${KEY_DIR}/${service_name}.key"
    rm "${KEY_DIR}/${service_name}.csr"
}

# Server Certificates
issue_cert "postgres" "server"
issue_cert "envoy_edge" "server"

# Client Certificates
CLIENT_SERVICES=(
    "api"
    "auth-service"
    "portfolio"
    "ml-inference"
    "worker"
    "ingestion-service"
    "transformer"
    "nse-scraper"
    "yfinance-scraper"
    "neural-pricing"
    "mlops-worker"
    "persistence-worker"
    "mlflow"
    "ray-head"
    "ray-worker-1"
    "rl-training-worker"
    "test-runner"
    "pgbouncer"
)

for service in "${CLIENT_SERVICES[@]}"; do
    issue_cert "$service" "client"
done

echo "✅ Security Layer Finalized in $KEY_DIR"
