#!/bin/bash
source scripts/utils_env.sh
load_decrypted_secrets
if [ -n "${RABBITMQ_USER:-}" ]; then
    echo "RABBITMQ_USER: $RABBITMQ_USER"
else
    echo "RABBITMQ_USER is NOT set"
fi
if [ -n "${RABBITMQ_PASSWORD:-}" ]; then
    echo "RABBITMQ_PASSWORD is set"
    echo "Length: ${#RABBITMQ_PASSWORD}"
else
    echo "RABBITMQ_PASSWORD is NOT set"
fi
