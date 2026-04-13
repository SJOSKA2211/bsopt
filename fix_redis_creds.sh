#!/bin/bash
set -e
source scripts/utils_env.sh
load_decrypted_secrets

if [ -n "${REDIS_PASSWORD:-}" ]; then
    echo "Updating REDIS_PASSWORD in .env..."
    sed -i '/^REDIS_PASSWORD=/d' .env
    echo "REDIS_PASSWORD=\"$REDIS_PASSWORD\"" >> .env
    echo " REDIS_PASSWORD updated."
else
    echo " REDIS_PASSWORD not found after decryption."
    exit 1
fi
