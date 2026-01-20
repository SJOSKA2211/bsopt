#!/bin/bash
# 🛡️ High-Concurrency Kernel Optimization for C100k
# This script tunes the host OS kernel limits to handle 100,000+ concurrent connections.

# Ensure script is run as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (sudo)"
  exit 1
fi

echo "🚀 Applying kernel optimizations for C100k..."

# 1. Increase Max Open Files (File Descriptors)
# Default is often 1024. We need >100k for Nginx + Sockets.
sysctl -w fs.file-max=2097152
ulimit -n 2097152

# 2. Tune TCP Stack for High Concurrency
# Allow reusing sockets in TIME_WAIT state (fast recycling)
sysctl -w net.ipv4.tcp_tw_reuse=1

# Increase ephemeral port range (allow more outgoing connections if needed)
sysctl -w net.ipv4.ip_local_port_range="1024 65535"

# Increase backlog (pending connections queue)
# Prevents "Connection Refused" during traffic spikes (Thundering Herd)
sysctl -w net.core.somaxconn=65535
sysctl -w net.ipv4.tcp_max_syn_backlog=65535

# 3. Memory Tuning for TCP
# Increase TCP buffer sizes for 10Gbps+ networks
sysctl -w net.core.rmem_max=16777216
sysctl -w net.core.wmem_max=16777216
sysctl -w net.ipv4.tcp_rmem="4096 87380 16777216"
sysctl -w net.ipv4.tcp_wmem="4096 65536 16777216"

# 4. Enable TCP Fast Open (Reduces latency by 1 RTT)
sysctl -w net.ipv4.tcp_fastopen=3

echo "✅ Kernel optimized for C100k concurrency"
