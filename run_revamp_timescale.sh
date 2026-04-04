#!/bin/bash
source scripts/utils_env.sh
load_decrypted_secrets
export PGBOUNCER_ENABLED=False
export DATABASE_URL="postgresql://admin:${POSTGRES_PASSWORD}@127.0.0.1:5435/bsopt?sslmode=disable"
.venv/bin/python scripts/revamp_timescale_fully.py
