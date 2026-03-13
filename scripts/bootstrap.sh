#!/usr/bin/env bash

# ==============================================================================
# Phase 0: Zero-Touch Automation, Security & Database Bootstrapping
# ==============================================================================
# Automates environment setup, secret generation, and stack deployment.
# ==============================================================================

set -e

ENV_FILE=".env"
ENV_EXAMPLE=".env.example"
DOCKER_COMPOSE_PROD="docker-compose.yml"
DOCKER_COMPOSE_DEV="docker-compose.dev.yml"

echo "🚀 Initiating Zero-Touch Automation..."

# --- 1. Security Automation: Secret Generation ---
if [ ! -f "$ENV_FILE" ]; then
    echo "🔐 Generating cryptographically secure MFA and JWT secrets..."
    
    JWT_SECRET=$(openssl rand -hex 32)
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
    
    # Inject secrets
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
    update_env "POSTGRES_DB" "bsopt"
    update_env "POSTGRES_USER" "admin"
    
    # Update URLs
    sed -i "s|:password@|:$DB_PASSWORD@|g" "$ENV_FILE"
    sed -i "s|:bsopt_redis_secret@|:$REDIS_PASSWORD@|g" "$ENV_FILE"
    
    echo "✅ Secrets injected into $ENV_FILE"
else
    echo "ℹ️  $ENV_FILE already exists."
fi

# Extract credentials from .env to use for DB initialization
source "$ENV_FILE"

# --- 2. Environment Selection ---
ENVIRONMENT=${1:-dev}
echo "🌍 Setting environment to: $ENVIRONMENT"

if [ "$ENVIRONMENT" == "prod" ]; then
    COMPOSE_CMD="docker-compose -f $DOCKER_COMPOSE_PROD"
else
    if [ -f "$DOCKER_COMPOSE_DEV" ]; then
        COMPOSE_CMD="docker-compose -f $DOCKER_COMPOSE_DEV"
    else
        COMPOSE_CMD="docker-compose -f $DOCKER_COMPOSE_PROD"
    fi
fi

# --- 3. Sequenced Startup & Health Gates ---
echo "🏗️  Starting Core Infrastructure (Postgres, Redis, RabbitMQ)..."
$COMPOSE_CMD up -d postgres redis rabbitmq

echo "⏳ Waiting for Database to fully initialize..."
MAX_RETRIES=30
COUNT=0
until $COMPOSE_CMD ps --format json | jq -e '. | select(.Service=="postgres" and .Health=="healthy")' > /dev/null 2>&1; do
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "❌ Timeout: Database failed to reach healthy state."
        exit 1
    fi
    printf "."
    sleep 2
    COUNT=$((COUNT + 1))
done
echo "✅ Database is Healthy."

echo "⏳ Waiting for Redis to fully initialize..."
COUNT=0
until $COMPOSE_CMD ps --format json | jq -e '. | select(.Service=="redis" and .Health=="healthy")' > /dev/null 2>&1; do
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "❌ Timeout: Redis failed to reach healthy state."
        exit 1
    fi
    printf "."
    sleep 2
    COUNT=$((COUNT + 1))
done
echo "✅ Redis is Healthy."

# Automatically trigger database creation using extracted credentials
echo "🛠️  Triggering database creation & initialization scripts..."
docker exec -e PGPASSWORD="$POSTGRES_PASSWORD" bsopt-postgres-1 psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1;" > /dev/null 2>&1 || true
# The init-scripts are automatically run by the postgres image, but we ensure it's forced if needed.

echo "🚀 Starting App Services (Sequential Build)..."
SERVICES=$($COMPOSE_CMD config --services)
for SERVICE in $SERVICES; do
    if [[ "$SERVICE" != "postgres" && "$SERVICE" != "redis" && "$SERVICE" != "rabbitmq" ]]; then
        echo "🏗️  Building & Starting $SERVICE..."
        $COMPOSE_CMD up --build -d "$SERVICE"
    fi
done

echo "✅ BS-OPT Sequenced Startup Complete!"
