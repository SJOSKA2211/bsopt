#!/bin/bash
set -e

# ==============================================================================
# BS-OPT: THE GOD MODE DB UPDATER (v2.0)
# ==============================================================================
# Applies incremental optimizations and schema refreshes.
# ==============================================================================

echo "🥒 Pressurizing the Manifold (Incremental Updates)..."

# Prioritize PG* variables for compatibility, then POSTGRES_*, then defaults
DB_HOST=${PGHOST:-${POSTGRES_HOST:-localhost}}
DB_PORT=${PGPORT:-${POSTGRES_PORT:-5432}}
DB_USER=${PGUSER:-${POSTGRES_USER:-admin}}
DB_NAME=${PGDATABASE:-${POSTGRES_DATABASE:-bsopt}}

if [ -z "$POSTGRES_PASSWORD" ] && [ -z "$PGPASSWORD" ]; then
    echo "❌ ERROR: Neither POSTGRES_PASSWORD nor PGPASSWORD is set. Execution halted."
    exit 1
fi
POSTGRES_PASSWORD=${PGPASSWORD:-$POSTGRES_PASSWORD}

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

# Apply optimization scripts in order
SCRIPTS=(
    "init-scripts/05-indexes.sql"
    "init-scripts/06-compression-retention.sql"
    "init-scripts/07-continuous-aggregates.sql"
    "init-scripts/08-materialized-views.sql"
    "init-scripts/09-security.sql"
    "init-scripts/10-missing-tables.sql"
    "init-scripts/11-scheduled-jobs.sql"
    "init-scripts/12-performance-dashboard.sql"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo "  🥒 Applying $script..."
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$script"
    else
        echo "  ⚠️  Warning: $script not found, skipping."
    fi
done

# Force a maintenance pass
echo "🚀 Running Maintenance Pass..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "VACUUM (ANALYZE, VERBOSE);"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT refresh_all_continuous_aggregates();"

echo "✅ Manifold pressurized and optimized. Solenya-tight! 🥒"
