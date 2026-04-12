#!/bin/bash
set -euo pipefail

# Ensure we have the environment correctly set
source scripts/utils_env.sh
load_decrypted_secrets

export BSOPT_ALLOW_WEAK_SECRETS=true
export PGBOUNCER_ENABLED=true

echo "--- STEP 1: Running NGINX until healthy ---"
./.venv/bin/python scripts/run_nginx_healthy.py

echo "--- STEP 2: Ensuring PgBouncer is ACTIVE and OPTIMIZED ---"
bash scripts/run_pgbouncer.sh

echo "--- STEP 3: Revamping and fully optimizing the engine ---"
./.venv/bin/python scripts/engine_revamp_god_mode.py

echo "--- STEP 4: Final Engine Health Report ---"
./.venv/bin/python scripts/engine_health.py --auto-fix
