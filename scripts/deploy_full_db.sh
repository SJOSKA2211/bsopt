#!/bin/bash
set -e

# ==============================================================================
# BS-OPT: THE GOD MODE DB DEPLOYER (v2.0)
# ==============================================================================
# Orchestrates full initialization, incremental migrations, and fine-tuning.
# ==============================================================================

echo "🥒 Launching Full-Scale Manifold Deployment..."

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

# Phase 1: Core Initialization
echo "🚀 Phase 1: Core Initialization..."
for script in $(find init-scripts/ -name "*.sql" | sort); do
    echo "  📜 Running $script..."
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$script"
done

# Phase 2: Incremental Migrations
echo "🚀 Phase 2: Structural Migrations..."
for script in $(find src/migrations/ -name "*.sql" | sort); do
    echo "  📜 Applying $script..."
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$script"
done

# Phase 3: Final Fine-Tuning (Pressurizing)
echo "🚀 Phase 3: Manifold Fine-Tuning..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "VACUUM (ANALYZE, VERBOSE);"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT refresh_all_continuous_aggregates();"

echo "✅ Full database deployment and optimization complete. Solenya-tight! 🥒"
