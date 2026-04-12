#!/bin/sh
# BS-OPT CI Heartbeat Wrapper

HEARTBEAT_FILE="/tmp/ci_heartbeat"

# Start heartbeat in background
echo "{\"time\": $(date +%s), \"metrics\": {\"health\": \"ACTIVE\", \"mode\": \"PARALLEL\"}}" > "$HEARTBEAT_FILE"
(
  while true; do
    echo "{\"time\": $(date +%s), \"metrics\": {\"health\": \"ACTIVE\", \"mode\": \"PARALLEL\"}}" > "$HEARTBEAT_FILE"
    sleep 5
  done
) &
HEARTBEAT_PID=$!

# Run the provided command
"$@"
RESULT=$?

# Cleanup heartbeat
kill $HEARTBEAT_PID 2>/dev/null
rm -f "$HEARTBEAT_FILE"

exit $RESULT
