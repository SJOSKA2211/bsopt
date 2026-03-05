#!/bin/bash
set -e

# Deploy all database initialization scripts in order
echo "🚀 Deploying full database schema..."

DB_HOST=${POSTGRES_HOST:-localhost}
DB_PORT=${POSTGRES_PORT:-5432}
DB_USER=${POSTGRES_USER:-postgres}
DB_NAME=${POSTGRES_DATABASE:-bsopt}

if [ -z "$POSTGRES_PASSWORD" ]; then
    echo "❌ ERROR: POSTGRES_PASSWORD is not set. Execution halted."
    exit 1
fi

if ! command -v psql &> /dev/null; then
    echo "❌ ERROR: 'psql' command not found."
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

# Execute scripts in strict alphanumeric order
for script in $(find init-scripts/ -name "*.sql" | sort); do
    echo "📜 Running $script..."
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$script"
done

echo "✅ Full database deployment complete."
