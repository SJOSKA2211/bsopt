#!/bin/sh
# ==
# BSOPT: CI HEARTBEAT SENTINEL
# ==
# This script is designed to run within containers to provide a heartrate 
# signal for the autonomous deployment loop.
# ==

HEARTBEAT_FILE=${1:-/tmp/manifold_heartbeat}
INTERVAL=${2:-60}

echo "Starting CI Heartbeat Sentinel on $HEARTBEAT_FILE with interval $INTERVAL s..."

while true; do
    date +%s > "$HEARTBEAT_FILE"
    sleep "$INTERVAL"
done
