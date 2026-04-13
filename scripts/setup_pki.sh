#!/bin/bash

set -euo pipefail

# Standardize on .pki/ in the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY_DIR="${PROJECT_ROOT}/.pki"
mkdir -p "$KEY_DIR"
mkdir -p "$KEY_DIR/vault"

echo " Initializing Production Security Layer in $KEY_DIR..."

# 1. Generate Root CA (RSA 4096)
if [[ ! -f "${KEY_DIR}/root_ca.key" ]]; then
    echo " Generating Root CA (RSA 4096)..."
    openssl genrsa -out "${KEY_DIR}/root_ca.key" 4096
    openssl req -x509 -new -nodes -key "${KEY_DIR}/root_ca.key" -sha256 -days 3650 \
        -out "${KEY_DIR}/root_ca.crt" \
        -subj "/C=US/ST=State/L=City/O=BSOPT-Production/CN=Manifold-Internal-CA"
    chmod 600 "${KEY_DIR}/root_ca.key"
fi

# 2. Generate Asymmetric JWT Key Pairs (ECC P-256 for performance/security balance)
if [[ ! -f "${KEY_DIR}/jwt_es256.key" ]]; then
    echo " Generating ES256 JWT Key Pair (ECC P-256)..."
    openssl ecparam -name prime256v1 -genkey -noout -out "${KEY_DIR}/jwt_es256.key"
    openssl ec -in "${KEY_DIR}/jwt_es256.key" -pubout -out "${KEY_DIR}/jwt_es256.pub"
    chmod 600 "${KEY_DIR}/jwt_es256.key"
fi

if [[ ! -f "${KEY_DIR}/jwt_rs256.key" ]]; then
    echo " Generating RS256 JWT Key Pair (RSA 4096)..."
    openssl genrsa -out "${KEY_DIR}/jwt_rs256.key" 4096
    openssl rsa -in "${KEY_DIR}/jwt_rs256.key" -pubout -out "${KEY_DIR}/jwt_rs256.pub"
    chmod 600 "${KEY_DIR}/jwt_rs256.key"
fi

# 3. Generate Vault RSA 4096 Key Pair for persistence encryption
if [[ ! -f "${KEY_DIR}/vault.key" ]]; then
    echo " Generating Vault RSA 4096 Key Pair..."
    openssl genrsa -out "${KEY_DIR}/vault.key" 4096
    openssl rsa -in "${KEY_DIR}/vault.key" -pubout -out "${KEY_DIR}/vault.pub"
    chmod 600 "${KEY_DIR}/vault.key"
fi

if [[ ! -f "${KEY_DIR}/argon2_salt.secret" ]]; then
    echo " Generating Argon2 Salt..."
    openssl rand -hex 16 > "${KEY_DIR}/argon2_salt.secret"
    chmod 600 "${KEY_DIR}/argon2_salt.secret"
fi

# 4. Function to Issue Service Certificates (Zero-Trust mTLS)
issue_cert() {
    local service_name=$1
    local type=$2 # server or client
    
    if [[ -f "${KEY_DIR}/${service_name}.crt" ]] && [[ -f "${KEY_DIR}/${service_name}.key" ]]; then
        return
    fi

    echo " Issuing $type certificate for $service_name..."
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
issue_cert "nginx" "server"

# Client Services
CLIENT_SERVICES=("api" "auth-service" "worker" "scraper" "nse-scraper" "yfinance-scraper" "ingestion-service" "neural-pricing" "ray-head" "mlflow" "mlops-worker" "ray-worker-1" "rl-training-worker")
for service in "${CLIENT_SERVICES[@]}"; do
    issue_cert "$service" "client"
done

echo " Production Security Layer Finalized in $KEY_DIR"
