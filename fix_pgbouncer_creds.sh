#!/bin/bash
set -e
source scripts/utils_env.sh
load_decrypted_secrets

# Update PGBOUNCER_ADMIN_PASSWORD in .env to match the decrypted POSTGRES_PASSWORD
if [ -n "${POSTGRES_PASSWORD:-}" ]; then
    echo "Updating PGBOUNCER_ADMIN_PASSWORD in .env..."
    # Remove old entry if exists
    sed -i '/^PGBOUNCER_ADMIN_PASSWORD=/d' .env
    echo "PGBOUNCER_ADMIN_PASSWORD=\"$POSTGRES_PASSWORD\"" >> .env
    echo "✅ PGBOUNCER_ADMIN_PASSWORD updated."
else
    echo "❌ POSTGRES_PASSWORD not found."
    exit 1
fi
