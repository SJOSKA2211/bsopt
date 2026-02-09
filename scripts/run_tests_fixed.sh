#!/bin/bash
export ENVIRONMENT=test
export DATABASE_URL="sqlite:///:memory:"
export REDIS_URL="redis://localhost:6379/0"
export JWT_SECRET="test_secret_key_change_me_in_prod"
export NUMBA_DISABLE_JIT=1

echo " Running tests with Optimized Environment..."
pytest --cov=src --cov-report=xml --cov-report=term-missing "$@"
