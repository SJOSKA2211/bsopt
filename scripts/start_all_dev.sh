#!/bin/bash
echo "🥒 Pickle Rick's Dev Stack Launcher 🥒"

# Trap Ctrl-C to kill all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

./scripts/start_infra.sh &
PID_INFRA=$!

sleep 5

./scripts/start_auth.sh &
PID_AUTH=$!

./scripts/start_api.sh &
PID_API=$!

./scripts/start_frontend.sh &
PID_FRONT=$!

wait
