#!/usr/bin/env bash

# ==============================================================================
# BS-OPT: COMPLETE STACK AUTOMATION & SECURITY BOOTSTRAPPING
# ==============================================================================
# Automates environment setup, secret generation, and stack deployment.
# ==============================================================================

set -e

# --- Configuration ---
ENV_FILE=".env"
ENV_EXAMPLE=".env.example"
DOCKER_COMPOSE_PROD="docker-compose.yml"
DOCKER_COMPOSE_DEV="docker-compose.dev.yml"

echo "🚀 Initiating BS-OPT High-Performance Manifold..."

# --- 1. Security Automation: Secret Generation ---
if [ ! -f "$ENV_FILE" ]; then
    echo "🔐 Generating cryptographically secure secrets..."
    
    # Generate secrets if openssl is available
    JWT_SECRET=$(openssl rand -hex 32)
    # MFA Key must be 32 URL-safe base64-encoded bytes for Fernet
    MFA_SECRET=$(openssl rand -base64 32)
    AUTH_SECRET=$(openssl rand -hex 32)
    DB_PASSWORD=$(openssl rand -hex 16)
    REDIS_PASSWORD=$(openssl rand -hex 16)
    RABBITMQ_PASSWORD=$(openssl rand -hex 16)
    
    if [ -f "$ENV_EXAMPLE" ]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
    else
        touch "$ENV_FILE"
    fi
    
    # Inject secrets (using a temporary file for safety)
    # Use a more reliable way to replace or append
    update_env() {
        local key=$1
        local value=$2
        if grep -q "^$key=" "$ENV_FILE"; then
            sed -i "s|^$key=.*|$key=$value|" "$ENV_FILE"
        else
            echo "$key=$value" >> "$ENV_FILE"
        fi
    }

    update_env "JWT_SECRET" "$JWT_SECRET"
    update_env "MFA_ENCRYPTION_KEY" "$MFA_SECRET"
    update_env "BETTER_AUTH_SECRET" "$AUTH_SECRET"
    update_env "POSTGRES_PASSWORD" "$DB_PASSWORD"
    update_env "REDIS_PASSWORD" "$REDIS_PASSWORD"
    update_env "RABBITMQ_PASSWORD" "$RABBITMQ_PASSWORD"
    
    # Update URLs that contain passwords
    sed -i "s|:password@|:$DB_PASSWORD@|g" "$ENV_FILE"
    sed -i "s|:bsopt_redis_secret@|:$REDIS_PASSWORD@|g" "$ENV_FILE"
    
    echo "✅ Secrets injected into $ENV_FILE"
else
    echo "ℹ️  $ENV_FILE already exists. Skipping secret generation."
fi

# --- 2. Environment Selection ---
ENVIRONMENT=${1:-dev}
echo "🌍 Setting environment to: $ENVIRONMENT"

# --- 3. Automated Startup ---
if [ "$ENVIRONMENT" == "prod" ]; then
    echo "🏗️  Starting Production Manifold..."
    docker-compose -f "$DOCKER_COMPOSE_PROD" up --build -d
else
    echo "🛠️  Starting Development Manifold..."
    # Ensure dev compose exists, fallback to main if not
    if [ -f "$DOCKER_COMPOSE_DEV" ]; then
        docker-compose -f "$DOCKER_COMPOSE_DEV" up --build -d
    else
        docker-compose -f "$DOCKER_COMPOSE_PROD" up --build -d
    fi
fi

echo "=========================================================="
echo "⚡ BS-OPT Manifold Ignition Complete!"
echo "=========================================================="
