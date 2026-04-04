#!/bin/bash
set -euo pipefail
source scripts/utils_env.sh
load_decrypted_secrets
export PYTHONPATH=$PYTHONPATH:.
uv run python report_rabbitmq_health.py
