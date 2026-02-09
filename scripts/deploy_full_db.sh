#!/bin/bash
set -e

# Deploy all database initialization scripts in order
echo " Deploying full database schema..."

DB_HOST=${POSTGRES_HOST:-localhost}
DB_PORT=${POSTGRES_PORT:-5432}
DB_USER=${POSTGRES_USER:-postgres}
DB_NAME=${POSTGRES_DATABASE:-bsopt}

export PGPASSWORD=$POSTGRES_PASSWORD

if ! command -v psql &> /dev/null; then
    echo "❌ 'psql' command not found."
    exit 1
fi

for script in init-scripts/*.sql; do
    echo "📜 Running $script..."
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$script"
done

echo "✅ Database deployment complete."
