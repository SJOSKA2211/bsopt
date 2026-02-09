#!/bin/bash
#  AF_XDP Market Ingestion Loader
# Compiles and loads the eBPF/XDP filter to bypass the kernel stack.

INTERFACE=$1
if [ -z "$INTERFACE" ]; then
    echo "Usage: $0 <interface>"
    exit 1
fi

echo "🏗️ Compiling XDP Filter..."
clang -O2 -g -target bpf -c scripts/xdp_filter.c -o scripts/xdp_filter.o

echo " Loading XDP Filter onto $INTERFACE..."
# Load the filter using iproute2
sudo ip link set dev "$INTERFACE" xdp obj scripts/xdp_filter.o sec xdp_market_filter

echo "✅ XDP Kernel Bypass Active on $INTERFACE (Port 5555)"
