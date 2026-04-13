# ==============================================================================
# Manifold: UNIFIED CPU-ONLY BASE (v12.0) - HARDENED & HYPER-FAST
# ==============================================================================

# --- STAGE 1: BUILDER (Heavy lifting, glibc-based for wheels) ---
FROM python:3.12-slim AS builder

# Optimized for BuildKit caching
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install essential build deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libssl-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for blisteringly fast installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# 1. Install Rust Core (CPU-Optimized)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
ENV PATH="/root/.cargo/bin:${PATH}"
COPY src/math_kernel/rust-core ./rust-core
RUN cd rust-core && uv pip install maturin && maturin build --release --out ../wheels

# 2. Build Python Environment
COPY pyproject.toml uv.lock ./
# Nuke GPU logic by forcing CPU-only torch
RUN uv pip install --no-cache torch --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-cache .
RUN uv pip install --no-cache ./wheels/*.whl

# --- STAGE 2: PRODUCTION RUNTIME (Minimalist Distroless-style Slim) ---
FROM python:3.12-slim AS latest

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

# Install runtime utilities for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && GRPC_HEALTH_PROBE_VERSION=v0.4.31 && \
    curl -fL -o /usr/local/bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
    chmod +x /usr/local/bin/grpc_health_probe && \
    rm -rf /var/lib/apt/lists/*

# Copy ONLY site-packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY . .

# Hardening: Dedicated non-root user
RUN groupadd -r Manifold && useradd -r -g Manifold Manifold
RUN chown -R Manifold:Manifold /app

USER Manifold

# Stage labels for downstream Dockerfiles
LABEL stage=production-base
