#!/bin/bash
set -e

# ==============================================================================
# BS-OPT: THE GOD MODE DB UPDATER (v2.1)
# ==============================================================================
# Applies incremental optimizations and schema refreshes.
# ==============================================================================

LOG_FILE="logs/db_deploy_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

log() {
    local message="[$(date +'%Y-%m-%dT%H:%M:%S%z')] $1"
    echo "$message"
    echo "$message" >> "$LOG_FILE"
}

log "🥒 Pressurizing the Manifold (Incremental Updates)..."

# Prioritize PG* variables for compatibility, then POSTGRES_*, then defaults
DB_HOST=${PGHOST:-${POSTGRES_HOST:-localhost}}
DB_PORT=${PGPORT:-${POSTGRES_PORT:-5432}}
DB_USER=${PGUSER:-${POSTGRES_USER:-admin}}
DB_NAME=${PGDATABASE:-${POSTGRES_DATABASE:-bsopt}}
POSTGRES_PASSWORD=${PGPASSWORD:-$POSTGRES_PASSWORD}

if [ -z "$POSTGRES_PASSWORD" ]; then
    log "❌ ERROR: POSTGRES_PASSWORD is not set. Execution halted."
    exit 1
fi

# Detect environment: Local vs Docker
USE_DOCKER=false
if ! command -v psql &> /dev/null; then
    if docker ps | grep -q "bsopt-postgres-1"; then
        log "  🥒 Local 'psql' not found. Using 'docker exec' fallback..."
        USE_DOCKER=true
    else
        log "❌ ERROR: 'psql' command not found and 'bsopt-postgres-1' container not running."
        exit 1
    fi
else
    # Auto-detect port from Docker if not specified
    if [ "$DB_PORT" = "5432" ] && docker ps | grep -q "bsopt-postgres-1"; then
        DETECTED_PORT=$(docker inspect --format='{{(index (index .NetworkSettings.Ports "5432/tcp") 0).HostPort}}' bsopt-postgres-1 2>/dev/null || echo "5432")
        if [ "$DETECTED_PORT" != "5432" ] && [ -n "$DETECTED_PORT" ]; then
            log "  🥒 Auto-detected Docker port: $DETECTED_PORT (Overriding $DB_PORT)"
            DB_PORT=$DETECTED_PORT
        fi
    fi
fi

run_sql_file() {
    local file=$1
    if [ "$USE_DOCKER" = true ]; then
        docker exec -i bsopt-postgres-1 psql -U "$DB_USER" -d "$DB_NAME" < "$file" >> "$LOG_FILE" 2>&1
    else
        PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$file" >> "$LOG_FILE" 2>&1
    fi
}

run_sql_cmd() {
    local cmd=$1
    if [ "$USE_DOCKER" = true ]; then
        docker exec -i bsopt-postgres-1 psql -U "$DB_USER" -d "$DB_NAME" -c "$cmd" >> "$LOG_FILE" 2>&1
    else
        PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "$cmd" >> "$LOG_FILE" 2>&1
    fi
}

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
        log "  🥒 Applying $script..."
        if ! run_sql_file "$script"; then
            log "  ❌ FAILED: $script"
            exit 1
        fi
    else
        log "  ⚠️  Warning: $script not found, skipping."
    fi
done

# Force a maintenance pass
log "🚀 Running Maintenance Pass..."
run_sql_cmd "VACUUM (ANALYZE, VERBOSE);"
run_sql_cmd "CALL refresh_all_continuous_aggregates();"

log "✅ Manifold pressurized and optimized. Solenya-tight! 🥒"
log "Log written to: $LOG_FILE"
