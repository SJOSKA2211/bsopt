#!/bin/bash
# BS-OPT CI Heartbeat Wrapper

HEARTBEAT_FILE="/tmp/ci_heartbeat"

# Start heartbeat in background
(
  while true; do
    echo "{\"time\": $(date +%s.%N), \"metrics\": {\"health\": \"ACTIVE\", \"mode\": \"PARALLEL\"}}" > "$HEARTBEAT_FILE"
    sleep 5
  done
) &
HEARTBEAT_PID=$!

# Run the provided command (usually pytest)
"$@"
RESULT=$?

# Cleanup heartbeat
kill $HEARTBEAT_PID
rm -f "$HEARTBEAT_FILE"

exit $RESULT
