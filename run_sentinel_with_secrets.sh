#!/bin/bash
source scripts/utils_env.sh
load_decrypted_secrets
export PGBOUNCER_HOST=127.0.0.1
export PGBOUNCER_PORT=6432
export PGBOUNCER_ADMIN_USER=${POSTGRES_USER:-admin}
export PGBOUNCER_ADMIN_PASSWORD=${POSTGRES_PASSWORD}
export REDIS_PASSWORD=${REDIS_PASSWORD}
.venv/bin/python scripts/system_sentinel.py
