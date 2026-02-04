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
sysctl -w net.core.somaxconn=65535
sysctl -w net.ipv4.tcp_max_syn_backlog=65535
sysctl -w net.ipv4.ip_local_port_range="1024 65535"
sysctl -w net.ipv4.tcp_tw_reuse=1
sysctl -w net.ipv4.tcp_fin_timeout=15

# 3. Memory Tuning for TCP
sysctl -w net.core.rmem_max=16777216
sysctl -w net.core.wmem_max=16777216
sysctl -w net.ipv4.tcp_rmem="4096 87380 16777216"
sysctl -w net.ipv4.tcp_wmem="4096 65536 16777216"

# 4. Enable TCP Fast Open and Low Latency
sysctl -w net.ipv4.tcp_fastopen=3
sysctl -w net.ipv4.tcp_low_latency=1

# 5. Adrenaline Tuning: Zero-Jitter Hardware Performance
# Force CPU to Performance Mode (requires cpupower or direct sysfs access)
echo "🔥 Locking CPU cores to Performance mode..."
if command -v cpupower > /dev/null; then
    cpupower frequency-set -g performance
else
    for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo "performance" > "$i"
    done
fi

# Disable CPU C-States (prevents cores from entering low-power sleep modes)
# This requires kernel boot parameters ideally, but we can hint via sysfs
if [ -f /sys/module/intel_idle/parameters/max_cstate ]; then
    echo 0 > /sys/module/intel_idle/parameters/max_cstate
fi

# 6. I/O Singularity: NVMe Optimization
# Set scheduler to 'none' for NVMe drives (bypass kernel scheduling overhead)
echo "⚡ Optimizing NVMe I/O paths..."
for i in /sys/block/nvme*/queue/scheduler; do
    echo "none" > "$i"
done

# 7. Hugepages Allocation (Optimization for high-RAM financial workloads)
...
# 6. Low-Latency Tuning
# Disable swap to prevent latency spikes
swapoff -a
# SOTA: Set memlock limits to prevent paging for high-performance workers
ulimit -l unlimited

# Optimize IRQ affinity for network cards (if available)
if command -v irqbalance > /dev/null; then
    systemctl stop irqbalance
    echo "irqbalance stopped - recommend manual IRQ binding for high-frequency workers"
fi

# CPU Topology Detection
CPU_CORES=$(nproc)
echo "💻 Detected $CPU_CORES cores. Recommended worker pinning:"
for i in $(seq 0 $((CPU_CORES-1))); do
    echo "  - Worker $i -> taskset -c $i"
done

echo "✅ Kernel optimized for Extreme Performance"
