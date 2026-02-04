#!/bin/bash
# 🚀 SINGULARITY: Automated Internal PKI for Zero-Trust mTLS
# Generates a Root CA and issues ephemeral certificates for platform services.

set -e

KEY_DIR="${HOME}/.bsopt/pki"
mkdir -p "$KEY_DIR"

# 1. Generate Root CA
if [[ ! -f "${KEY_DIR}/root_ca.key" ]]; then
    echo "🛡️ Initializing Internal Root CA..."
    openssl genrsa -out "${KEY_DIR}/root_ca.key" 4096
    openssl req -x509 -new -nodes -key "${KEY_DIR}/root_ca.key" -sha256 -days 3650 \
        -out "${KEY_DIR}/root_ca.crt" \
        -subj "/C=US/ST=State/L=City/O=BSOPT-GOD-MODE/CN=BSOPT-Internal-CA"
fi

# 2. Function to Issue Service Certificates
issue_cert() {
    local service_name=$1
    echo "📜 Issuing certificate for $service_name..."
    
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
}

# Issue certs for the core triad
issue_cert "api-gateway"
issue_cert "pricing-subgraph"
issue_cert "ml-subgraph"

echo "✅ Internal PKI initialized in $KEY_DIR"
