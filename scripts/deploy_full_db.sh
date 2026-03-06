#!/bin/bash
set -e

# ==============================================================================
# BS-OPT: THE GOD MODE DB DEPLOYER (v2.1)
# ==============================================================================
# Orchestrates full initialization, incremental migrations, and fine-tuning.
# ==============================================================================

LOG_FILE="logs/db_deploy_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

log() {
    local message="[$(date +'%Y-%m-%dT%H:%M:%S%z')] $1"
    echo "$message"
    echo "$message" >> "$LOG_FILE"
}

log "🥒 Launching Full-Scale Manifold Deployment..."

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

# Phase 1: Core Initialization
log "🚀 Phase 1: Core Initialization..."
for script in $(find init-scripts/ -name "*.sql" | sort); do
    log "  📜 Running $script..."
    if ! run_sql_file "$script"; then
        log "  ❌ FAILED: $script"
        exit 1
    fi
done

# Phase 2: Incremental Migrations
log "🚀 Phase 2: Structural Migrations..."
if [ -d "src/migrations/" ]; then
    for script in $(find src/migrations/ -name "*.sql" | sort); do
        log "  📜 Applying $script..."
        if ! run_sql_file "$script"; then
            log "  ❌ FAILED: $script"
            exit 1
        fi
    done
else
    log "  ⚠️  No migrations found in src/migrations/"
fi

# Phase 3: Final Fine-Tuning (Pressurizing)
log "🚀 Phase 3: Manifold Fine-Tuning..."
run_sql_cmd "VACUUM (ANALYZE, VERBOSE);"
run_sql_cmd "CALL refresh_all_continuous_aggregates();"

log "✅ Full database deployment and optimization complete. Solenya-tight! 🥒"
log "Log written to: $LOG_FILE"
