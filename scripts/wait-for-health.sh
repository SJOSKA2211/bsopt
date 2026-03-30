#!/bin/bash
# scripts/wait-for-health.sh
containers=("$@")
while true; do
  all_healthy=true
  for container in "${containers[@]}"; do
    status=$(docker inspect "$container" --format='{{json .State.Health.Status}}' 2>/dev/null || echo "unhealthy")
    if [[ "$status" != "\"healthy\"" ]]; then
      echo "Waiting for $container ($status)..."
      all_healthy=false
      break
    fi
  done
  if $all_healthy; then
    echo "All containers are healthy!"
    exit 0
  fi
  sleep 5
done
