#!/bin/bash
# scripts/run_geth_manual.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONTAINER_NAME="geth-bsopt"
IMAGE="docker.io/ethereum/client-go:stable"
NETWORK="bsopt_bsopt-network"

echo " Starting Geth container: $CONTAINER_NAME..."
docker run -d --name "$CONTAINER_NAME" \
  -p 8545:8545 -p 8546:8546 \
  --network "$NETWORK" \
  --memory 2gb \
  "$IMAGE" --dev --http --http.addr 0.0.0.0 --http.vhosts "*" --http.api eth,net,web3,debug

echo "⏳ Waiting for Geth (8545) to become healthy..."
RETRIES=60
SUCCESS=false
until [ $RETRIES -eq 0 ]; do
    if curl -s -X POST -H "Content-Type: application/json" \
      --data '{"jsonrpc":"2.0","method":"net_version","params":[],"id":67}' \
      http://localhost:8545 | grep -q result; then
        SUCCESS=true
        break
    fi
    echo "Waiting for JSON-RPC at http://localhost:8545 ($RETRIES retries left)..."
    sleep 5
    ((RETRIES--))
done

if [ "$SUCCESS" = false ]; then
    echo " Fatal: Geth failed to reach stable state within timeout."
    docker logs "$CONTAINER_NAME"
    exit 1
fi

echo " Geth is Online and Healthy!"
echo "Report: Geth engine version $(curl -s -X POST -H "Content-Type: application/json" --data '{"jsonrpc":"2.0","method":"web3_clientVersion","params":[],"id":1}' http://localhost:8545 | jq -r .result)"
