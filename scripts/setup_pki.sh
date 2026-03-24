#!/bin/bash
#  scripts/setup_pki.sh - Institutional Automated Internal PKI
# Generates Root CA, Server/Client Certs, JWT keys, and Vault keys with Zero-Mock integrity.

set -euo pipefail

# Standardize on .pki/ in the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY_DIR="${PROJECT_ROOT}/.pki"
mkdir -p "$KEY_DIR"
mkdir -p "$KEY_DIR/vault"

echo "🔐 Initializing Institutional Security Layer in $KEY_DIR..."

# 1. Generate Root CA (RSA 4096)
if [[ ! -f "${KEY_DIR}/root_ca.key" ]]; then
    echo "📜 Generating Root CA (RSA 4096)..."
    openssl genrsa -out "${KEY_DIR}/root_ca.key" 4096
    openssl req -x509 -new -nodes -key "${KEY_DIR}/root_ca.key" -sha256 -days 3650 \
        -out "${KEY_DIR}/root_ca.crt" \
        -subj "/C=US/ST=State/L=City/O=BSOPT-INSTITUTIONAL/CN=EquaFlow-Internal-CA"
    chmod 600 "${KEY_DIR}/root_ca.key"
fi

# 2. Generate Asymmetric JWT Key Pairs (ECC P-256 for performance/security balance)
if [[ ! -f "${KEY_DIR}/jwt_es256.key" ]]; then
    echo "🔑 Generating ES256 JWT Key Pair (ECC P-256)..."
    openssl ecparam -name prime256v1 -genkey -noout -out "${KEY_DIR}/jwt_es256.key"
    openssl ec -in "${KEY_DIR}/jwt_es256.key" -pubout -out "${KEY_DIR}/jwt_es256.pub"
    chmod 600 "${KEY_DIR}/jwt_es256.key"
fi

# 3. Generate Vault RSA 4096 Key Pair for persistence encryption
if [[ ! -f "${KEY_DIR}/vault/vault.key" ]]; then
    echo "🔑 Generating Vault RSA 4096 Key Pair..."
    openssl genrsa -out "${KEY_DIR}/vault/vault.key" 4096
    openssl rsa -in "${KEY_DIR}/vault/vault.key" -pubout -out "${KEY_DIR}/vault/vault.pub"
    chmod 600 "${KEY_DIR}/vault/vault.key"
fi

# 4. Function to Issue Service Certificates (Zero-Trust mTLS)
issue_cert() {
    local service_name=$1
    local type=$2 # server or client
    
    if [[ -f "${KEY_DIR}/${service_name}.crt" ]] && [[ -f "${KEY_DIR}/${service_name}.key" ]]; then
        return
    fi

    echo "📜 Issuing $type certificate for $service_name..."
    openssl genrsa -out "${KEY_DIR}/${service_name}.key" 2048
    openssl req -new -key "${KEY_DIR}/${service_name}.key" \
        -out "${KEY_DIR}/${service_name}.csr" \
        -subj "/C=US/ST=State/L=City/O=BSOPT/CN=${service_name}"
    openssl x509 -req -in "${KEY_DIR}/${service_name}.csr" \
        -CA "${KEY_DIR}/root_ca.crt" -CAkey "${KEY_DIR}/root_ca.key" \
        -CAcreateserial -out "${KEY_DIR}/${service_name}.crt" \
        -days 365 -sha256
    chmod 600 "${KEY_DIR}/${service_name}.key"
    rm "${KEY_DIR}/${service_name}.csr"
}

# Core Infrastructure
issue_cert "postgres" "server"
issue_cert "pgbouncer" "server"
issue_cert "envoy" "server"
issue_cert "redis" "server"
issue_cert "rabbitmq" "server"
issue_cert "minio" "server"

# Client Services
CLIENT_SERVICES=("api" "auth-service" "worker" "scraper" "neural-pricing")
for service in "${CLIENT_SERVICES[@]}"; do
    issue_cert "$service" "client"
done

echo "✅ Institutional Security Layer Finalized in $KEY_DIR"
