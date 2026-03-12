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
    ENCRYPTION_KEY=$(openssl rand -hex 32)
    DB_PASSWORD=$(openssl rand -hex 16)
    REDIS_PASSWORD=$(openssl rand -hex 16)
    RABBITMQ_PASSWORD=$(openssl rand -hex 16)
    
    if [ -f "$ENV_EXAMPLE" ]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
    else
        touch "$ENV_FILE"
    fi
    
    # Inject secrets (using a temporary file for safety)
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
    update_env "ENCRYPTION_KEY" "$ENCRYPTION_KEY"
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

# --- 3. Automated Startup & Health Gates ---
if [ "$ENVIRONMENT" == "prod" ]; then
    COMPOSE_CMD="docker-compose -f $DOCKER_COMPOSE_PROD"
else
    if [ -f "$DOCKER_COMPOSE_DEV" ]; then
        COMPOSE_CMD="docker-compose -f $DOCKER_COMPOSE_DEV"
    else
        COMPOSE_CMD="docker-compose -f $DOCKER_COMPOSE_PROD"
    fi
fi

echo "🏗️  Igniting Manifold with BuildKit..."
$COMPOSE_CMD up --build -d

echo "⏳ Waiting for Database to stabilize..."
MAX_RETRIES=30
COUNT=0
until $COMPOSE_CMD ps --format json | grep -q '"Service":"postgres","Health":"healthy"'; do
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "❌ Timeout: Database failed to reach healthy state."
        exit 1
    fi
    printf "."
    sleep 2
    COUNT=$((COUNT + 1))
done
echo "✅ Database is Healthy."

echo "🔍 Verifying all critical services..."
CRITICAL_SERVICES=("api" "auth-service" "redis" "rabbitmq")
for SERVICE in "${CRITICAL_SERVICES[@]}"; do
    if ! $COMPOSE_CMD ps --format json | grep -q "\"Service\":\"$SERVICE\",\"Health\":\"healthy\""; then
        echo "⚠️  Warning: $SERVICE is not yet healthy. Checking logs..."
        # Non-blocking warning as some services might take longer
    fi
done

echo "=========================================================="
echo "⚡ BS-OPT Manifold Ignition Complete!"
echo "=========================================================="
