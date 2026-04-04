source .venv/bin/activate
#!/bin/bash
# run_api_local.sh

# Load shared environment utilities
source scripts/utils_env.sh

# Load and decrypt secrets into current shell
load_decrypted_secrets

# Override some variables for local run (pointing to Docker mapped ports)
export DATABASE_URL="postgresql://admin:${POSTGRES_PASSWORD}@localhost:5435/bsopt?sslmode=disable"
export REDIS_HOST="localhost"
export REDIS_PORT="6380"
export REDIS_URL="redis://:${REDIS_PASSWORD}@localhost:6380/0"
export PGBOUNCER_ENABLED=True
export INSIDE_DOCKER=0
export PYTHONPATH=$(pwd)

# Start API
granian --interface asgi api.index:app --host 0.0.0.0 --port 8000
