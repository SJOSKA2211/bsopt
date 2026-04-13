# Manifold: UNIFIED CPU-ONLY BASE (v13.0)
# Optimized for pure CPU execution on Alpine Linux

# --- STAGE 1: BUILDER ---
FROM python:3.12.13-alpine AS builder

# Optimized for BuildKit caching
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build dependencies with cache mount
RUN --mount=type=cache,target=/var/cache/apk \
    apk add --no-cache \
    build-base \
    curl \
    openssl-dev \
    pkgconfig \
    git \
    patchelf \
    rust \
    cargo

WORKDIR /app

# Install uv core
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 1. Build Rust Math Kernel with cache mount
COPY src/math_kernel/rust-core ./rust-core
RUN --mount=type=cache,target=/root/.cache/uv \
    cd rust-core && uv pip install maturin && maturin build --release --out ../wheels

# 2. Build Python Environment with cache mounts
COPY pyproject.toml uv.lock ./
# Force CPU-only torch
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install . && \
    uv pip install ./wheels/*.whl

# --- STAGE 2: PRODUCTION RUNTIME ---
FROM python:3.12.13-alpine AS latest

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

# Runtime utilities
RUN apk add --no-cache curl

# Copy ONLY site-packages and binaries
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY . .

# Hardening
RUN addgroup -S manifold && adduser -S manifold -G manifold
RUN chown -R manifold:manifold /app

USER manifold

LABEL stage=production-base
