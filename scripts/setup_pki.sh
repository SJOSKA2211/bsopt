#!/bin/bash
# Manifold: PKI & mTLS Configuration Script (Hardened v2.0)
# Generates Root CA and service-specific identities for zero-trust comms.

set -e

KEYS_DIR=".pki"
mkdir -p "$KEYS_DIR"
chmod 700 "$KEYS_DIR"

log_info() { echo -e "\033[0;34m[INFO]\033[0m $1"; }

log_info "Generating Root Certificate Authority..."
if [ ! -f "$KEYS_DIR/root_ca.key" ]; then
    openssl genrsa -out "$KEYS_DIR/root_ca.key" 4096
    openssl req -x509 -new -nodes -key "$KEYS_DIR/root_ca.key" -sha256 -days 3650 \
        -out "$KEYS_DIR/root_ca.crt" -subj "/CN=Manifold-Root-CA/O=Manifold/C=US"
fi

log_info "Generating Auth Service Identity (mTLS)..."
if [ ! -f "$KEYS_DIR/auth-service.key" ]; then
    openssl genrsa -out "$KEYS_DIR/auth-service.key" 2048
    openssl req -new -key "$KEYS_DIR/auth-service.key" -out "$KEYS_DIR/auth-service.csr" \
        -subj "/CN=auth-service/O=Manifold/C=US"
    
    # Sign with Root CA
    openssl x509 -req -in "$KEYS_DIR/auth-service.csr" -CA "$KEYS_DIR/root_ca.crt" \
        -CAkey "$KEYS_DIR/root_ca.key" -CAcreateserial -out "$KEYS_DIR/auth-service.crt" \
        -days 365 -sha256
    rm "$KEYS_DIR/auth-service.csr"
fi

log_info "Generating JWT Cryptographic Keys..."
# RSA 256
if [ ! -f "$KEYS_DIR/jwt_rs256.key" ]; then
    openssl genrsa -out "$KEYS_DIR/jwt_rs256.key" 4096
    openssl rsa -in "$KEYS_DIR/jwt_rs256.key" -pubout -out "$KEYS_DIR/jwt_rs256.pub"
fi

# ES 256 (Elliptic Curve)
if [ ! -f "$KEYS_DIR/jwt_es256.key" ]; then
    openssl ecparam -name prime256v1 -genkey -noout -out "$KEYS_DIR/jwt_es256.key"
    openssl ec -in "$KEYS_DIR/jwt_es256.key" -pubout -out "$KEYS_DIR/jwt_es256.pub"
fi

log_info "Generating Runtime Secret Vault Keys (RSA-4096)..."
if [ ! -f "$KEYS_DIR/vault.key" ]; then
    openssl genrsa -out "$KEYS_DIR/vault.key" 4096
    openssl rsa -in "$KEYS_DIR/vault.key" -pubout -out "$KEYS_DIR/vault.pub"
fi

log_info "Generating Global Argon2 Salt..."
if [ ! -f "$KEYS_DIR/argon2_salt.secret" ]; then
    openssl rand -base64 32 > "$KEYS_DIR/argon2_salt.secret"
fi

chmod 600 "$KEYS_DIR"/*.key
chmod 600 "$KEYS_DIR"/*.secret
log_info "PKI Initialization Complete."
