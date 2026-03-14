#!/bin/bash
#  OPTIMIZED: Automated Internal PKI for Zero-Trust mTLS & Asymmetric Security
# Generates Root CA, ECC/RSA JWT keys, and TOTP secrets.

set -e

KEY_DIR="${HOME}/.bsopt/pki"
mkdir -p "$KEY_DIR"

echo "🔐 Initializing Enterprise Security Layer..."

# 1. Generate Root CA (RSA 4096)
if [[ ! -f "${KEY_DIR}/root_ca.key" ]]; then
    echo "📜 Generating Root CA (RSA 4096)..."
    openssl genrsa -out "${KEY_DIR}/root_ca.key" 4096
    openssl req -x509 -new -nodes -key "${KEY_DIR}/root_ca.key" -sha256 -days 3650 \
        -out "${KEY_DIR}/root_ca.crt" \
        -subj "/C=US/ST=State/L=City/O=BSOPT-INSTITUTIONAL/CN=EquaFlow-Internal-CA"
fi

# 2. Generate Asymmetric JWT Key Pairs
# RS256 (RSA 4096) - Upgraded from 2048 for high-security environments
if [[ ! -f "${KEY_DIR}/jwt_rs256.key" ]]; then
    echo "🔑 Generating RS256 JWT Key Pair (RSA 4096)..."
    openssl genrsa -out "${KEY_DIR}/jwt_rs256.key" 4096
    openssl rsa -in "${KEY_DIR}/jwt_rs256.key" -pubout -out "${KEY_DIR}/jwt_rs256.pub"
fi

# ES256 (ECC P-256)
if [[ ! -f "${KEY_DIR}/jwt_es256.key" ]]; then
    echo "🔑 Generating ES256 JWT Key Pair (ECC P-256)..."
    openssl ecparam -name prime256v1 -genkey -noout -out "${KEY_DIR}/jwt_es256.key"
    openssl ec -in "${KEY_DIR}/jwt_es256.key" -pubout -out "${KEY_DIR}/jwt_es256.pub"
fi

# 3. Generate TOTP Master Secret
if [[ ! -f "${KEY_DIR}/totp_master.secret" ]]; then
    echo "🛡️ Generating TOTP Master Secret..."
    # 32 bytes of secure entropy for TOTP secrets
    openssl rand -hex 32 > "${KEY_DIR}/totp_master.secret"
fi

# 4. Function to Issue Service Certificates (Zero-Trust mTLS)
issue_cert() {
    local service_name=$1
    echo "📜 Issuing certificate for $service_name..."
    
    # Private Key (RSA 2048 is sufficient for service-to-service certs if rotated frequently)
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
}

# Issue certs for the core microservices triad
issue_cert "api-gateway"
issue_cert "pricing-subgraph"
issue_cert "ml-subgraph"

echo "✅ Security Layer Finalized in $KEY_DIR"
