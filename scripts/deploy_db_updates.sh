#!/bin/bash
set -eo pipefail

# ==============================================================================
# BS-OPT: THE HIGH-PERFORMANCE DB UPDATER (v3.0)
# ==============================================================================
# Applies incremental optimizations and schema refreshes with audit tracking.
# ==============================================================================

LOG_FILE="logs/db_deploy_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

log() {
    local message="[$(date +'%Y-%m-%dT%H:%M:%S%z')] $1"
    echo "$message"
    echo "$message" >> "$LOG_FILE"
}

error_handler() {
    log "❌ FATAL ERROR occurred on line $1. Check $LOG_FILE for details."
    exit 1
}

trap 'error_handler $LINENO' ERR

# Load environment variables if .env exists
if [ -f .env ]; then
    log "   Loading .env..."
    export $(grep -v '^#' .env | xargs)
fi

# Pre-flight Validation
log " Pre-flight check..."
if ! query_sql "SELECT 1" > /dev/null 2>&1; then
    log "❌ ERROR: Cannot connect to database. Is it running?"
    exit 1
fi

EXT_CHECK=$(query_sql "SELECT count(*) FROM pg_extension WHERE extname IN ('timescaledb', 'vector');")
if [ "$EXT_CHECK" != "2" ]; then
    log "⚠️  Warning: Core extensions (timescaledb/vector) missing. Check init-scripts."
fi
DB_HOST=${PGHOST:-${POSTGRES_HOST:-127.0.0.1}}
DB_PORT=${PGPORT:-${POSTGRES_PORT:-5432}}
DB_USER=${PGUSER:-${POSTGRES_USER:-admin}}
DB_NAME=${PGDATABASE:-${POSTGRES_DATABASE:-bsopt}}
POSTGRES_PASSWORD=${PGPASSWORD:-$POSTGRES_PASSWORD}

if [ -z "$POSTGRES_PASSWORD" ]; then
    log "❌ ERROR: POSTGRES_PASSWORD is not set. Execution halted."
    exit 1
fi

# Detect environment: Prefer Docker if container is running
USE_DOCKER=false
if docker ps | grep -q "bsopt-postgres-1"; then
    log "   Container 'bsopt-postgres-1' detected. Using 'docker exec'..."
    USE_DOCKER=true
elif ! command -v psql &> /dev/null; then
    log "❌ ERROR: 'psql' command not found and 'bsopt-postgres-1' container not running."
    exit 1
fi

query_sql() {
    local cmd=$1
    if [ "$USE_DOCKER" = true ]; then
        docker exec -i -e PGPASSWORD="$POSTGRES_PASSWORD" bsopt-postgres-1 psql -t -v ON_ERROR_STOP=1 -U "$DB_USER" -d "$DB_NAME" -c "$cmd" 2>/dev/null | xargs
    else
        PGPASSWORD="$POSTGRES_PASSWORD" psql -t -v ON_ERROR_STOP=1 -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "$cmd" 2>/dev/null | xargs
    fi
}

run_sql_cmd() {
    local cmd=$1
    if [ "$USE_DOCKER" = true ]; then
        docker exec -i -e PGPASSWORD="$POSTGRES_PASSWORD" bsopt-postgres-1 psql -v ON_ERROR_STOP=1 -U "$DB_USER" -d "$DB_NAME" -c "$cmd" >> "$LOG_FILE" 2>&1
    else
        PGPASSWORD="$POSTGRES_PASSWORD" psql -v ON_ERROR_STOP=1 -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "$cmd" >> "$LOG_FILE" 2>&1
    fi
}

run_sql_file() {
    local file=$1
    if [ "$USE_DOCKER" = true ]; then
        docker exec -i -e PGPASSWORD="$POSTGRES_PASSWORD" bsopt-postgres-1 psql -v ON_ERROR_STOP=1 -U "$DB_USER" -d "$DB_NAME" < "$file" >> "$LOG_FILE" 2>&1
    else
        PGPASSWORD="$POSTGRES_PASSWORD" psql -v ON_ERROR_STOP=1 -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$file" >> "$LOG_FILE" 2>&1
    fi
}

# Ensure deployment_history table exists
log "   Initializing audit history..."
run_sql_cmd "CREATE TABLE IF NOT EXISTS deployment_history (
    id SERIAL PRIMARY KEY,
    script_name TEXT UNIQUE NOT NULL,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);"

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
    "init-scripts/13-pg16-diagnostics.sql"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        # Check if already applied
        ALREADY_APPLIED=$(query_sql "SELECT 1 FROM deployment_history WHERE script_name = '$script';")
        
        if [ "$ALREADY_APPLIED" = "1" ]; then
            log "  ⏭️  Skipping $script (Already applied)"
        else
            log "   Applying $script..."
            if run_sql_file "$script"; then
                run_sql_cmd "INSERT INTO deployment_history (script_name) VALUES ('$script');"
            else
                log "  ❌ FAILED: $script"
                exit 1
            fi
        fi
    else
        log "  ⚠️  Warning: $script not found, skipping."
    fi
done

# Force a maintenance pass
log " Running Maintenance Pass..."
run_sql_cmd "VACUUM (ANALYZE, VERBOSE);"

log " Refreshing Continuous Aggregates..."
VIEWS=$(query_sql "SELECT view_name FROM timescaledb_information.continuous_aggregates;")
for view in $VIEWS; do
    log "   Refreshing: $view..."
    run_sql_cmd "CALL refresh_continuous_aggregate('$view', NULL, NULL);"
done

log " Manifold pressurized and optimized. production-ready! "
log "Log written to: $LOG_FILE"
