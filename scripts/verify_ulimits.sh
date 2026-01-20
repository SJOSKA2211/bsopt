#!/bin/bash
# Verify ulimits for a running container
if [ -z "$1" ]; then
  echo "Usage: ./verify_ulimits.sh <container_name>"
  exit 1
fi

CONTAINER=$1
echo "🔍 Checking ulimits for $CONTAINER..."
docker exec $CONTAINER sh -c "ulimit -n"
