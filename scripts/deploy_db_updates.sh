#!/bin/bash
set -e

# Deployment script for database updates
# Usage: ./deploy_db_updates.sh

echo "🚀 Deploying database updates..."

if [ -z "$POSTGRES_HOST" ]; then
  echo "⚠️  POSTGRES_HOST is not set. Assuming local docker-compose environment or manual execution."
  echo "    If running manually, please set POSTGRES_HOST, POSTGRES_USER, and POSTGRES_DB."
fi

# Default values for local dev
DB_HOST=${POSTGRES_HOST:-localhost}
DB_PORT=${POSTGRES_PORT:-5432}
DB_USER=${POSTGRES_USER:-postgres}
DB_NAME=${POSTGRES_DATABASE:-bsopt}

echo "📍 Target Database: $DB_HOST:$DB_PORT/$DB_NAME"

# Check if psql is installed
if ! command -v psql &> /dev/null; then
    echo "❌ 'psql' command not found. Please install postgresql-client."
    exit 1
fi

# Run the optimization script
echo "📜 Running optimization scripts..."
export PGPASSWORD=$POSTGRES_PASSWORD

# Apply optimization scripts in order
SCRIPTS=(
    "init-scripts/05-optimize-indices.sql"
    "init-scripts/07-market-stats-mv.sql"
    "init-scripts/08-continuous-aggregates.sql"
    "init-scripts/09-portfolio-summary-mv.sql"
    "init-scripts/10-trading-stats-mv.sql"
    "init-scripts/11-drift-metrics-mv.sql"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo "  ➡️ Applying $script..."
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$script"
        if [ $? -ne 0 ]; then
            echo "  ❌ Failed to apply $script."
            exit 1
        fi
    else
        echo "  ⚠️  Warning: $script not found, skipping."
    fi
done

echo "✅ Database optimized and materialized views created successfully."

echo "🎉 Database deployment complete."
