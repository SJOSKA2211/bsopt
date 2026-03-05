#!/bin/bash
set -e

# Deployment script for database updates
# Usage: ./deploy_db_updates.sh

echo "🚀 Deploying database updates..."

# Default values for local dev
DB_HOST=${POSTGRES_HOST:-localhost}
DB_PORT=${POSTGRES_PORT:-5432}
DB_USER=${POSTGRES_USER:-postgres}
DB_NAME=${POSTGRES_DATABASE:-bsopt}

if [ -z "$POSTGRES_PASSWORD" ]; then
    echo "❌ ERROR: POSTGRES_PASSWORD is not set. Execution halted."
    exit 1
fi

echo "📍 Target Database: $DB_HOST:$DB_PORT/$DB_NAME"

# Check if psql is installed
if ! command -v psql &> /dev/null; then
    echo "❌ ERROR: 'psql' command not found. Please install postgresql-client."
    exit 1
fi

# Secure password handling using PGPASSFILE
PGPASSFILE_TMP=$(mktemp)
chmod 0600 "$PGPASSFILE_TMP"
echo "$DB_HOST:$DB_PORT:$DB_NAME:$DB_USER:$POSTGRES_PASSWORD" > "$PGPASSFILE_TMP"
export PGPASSFILE="$PGPASSFILE_TMP"

# Ensure cleanup on exit
cleanup() {
    rm -f "$PGPASSFILE_TMP"
}
trap cleanup EXIT

# Apply optimization scripts in order
SCRIPTS=(
    "init-scripts/05-indexes.sql"
    "init-scripts/07-continuous-aggregates.sql"
    "init-scripts/08-materialized-views.sql"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo "  ➡️ Applying $script..."
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$script"
    else
        echo "  ⚠️  Warning: $script not found, skipping."
    fi
done

echo "✅ Database optimized and materialized views created successfully."
echo "🎉 Database deployment complete."
