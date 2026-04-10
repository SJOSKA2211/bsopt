#!/bin/bash
set -euo pipefail

# Wrapper to execute the full request
source scripts/utils_env.sh
load_decrypted_secrets

echo "--- STEP 1: Running Redis until healthy ---"
# We need to export the decrypted REDIS_PASSWORD so the python script can use it
python3 scripts/run_redis_healthy.py

echo "--- STEP 2: Ensuring PgBouncer is ACTIVE and OPTIMIZED ---"
bash scripts/run_pgbouncer.sh

echo "--- STEP 3: Revamping and fully optimizing the engine ---"
# This script initializes the DB, sets tuning params, and optimizes hypertables
python3 scripts/engine_revamp_god_mode.py

echo "--- STEP 4: Final Engine Health Report ---"
python3 scripts/engine_health.py --auto-fix
