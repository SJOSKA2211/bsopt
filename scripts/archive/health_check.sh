#!/bin/bash
# Wrapper for ml-health CLI tool

# Set default URL if not provided
export ML_HEALTH_URL=${ML_HEALTH_URL:-"http://localhost:8000/ml/health"}

# Run the CLI tool
python3 bin/ml-health "$@"