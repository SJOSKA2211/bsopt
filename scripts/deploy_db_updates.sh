#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🗄️ Executing Production Database Schema Synchronization..."

# Load Production environment
source scripts/utils_env.sh
load_decrypted_secrets

# Pre-flight Validation
if ! command -v psql > /dev/null; then
    echo "❌ Error: psql client not found. Mandatory for Production deployment."
    exit 1
fi

# Using the service-specific URL from the environment
export PGPASSWORD=${POSTGRES_PASSWORD:-}

echo "🩺 Verifying Database Connectivity..."
if ! psql "$DATABASE_URL" -c "SELECT 1" > /dev/null 2>&1; then
    echo "❌ Error: Cannot connect to database substrate."
    exit 1
fi

# Apply Materialized View Refreshes and Optimizations
echo "🔄 Refreshing Production Analytical Views..."
psql "$DATABASE_URL" <<EOF
-- Explicitly refresh the ML comparison dashboard view
REFRESH MATERIALIZED VIEW CONCURRENTLY ml_comparison_stats;
-- Ensure TimescaleDB maintenance is active
SELECT run_job(job_id) FROM timescaledb_information.jobs WHERE proc_name = 'policy_compression';
EOF

echo "✅ Database Substrate Synchronized."
