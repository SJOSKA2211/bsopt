# Manifold: UNIFIED CPU-ONLY BASE (v14.0)
# Optimized for pure CPU execution on distroless

# --- STAGE 1: BUILDER ---
FROM python:3.12.13-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libssl-dev \
    libffi-dev \
    patchelf \
    git \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Compile Rust Core Kernel (Performance Substrate)
COPY src/math_kernel/rust-core ./rust-core
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install maturin[patchelf] && \
    cd rust-core && maturin build --release --out ../wheels

# Install Python dependencies into a temporary location
COPY pyproject.toml uv.lock ./
ARG EXTRAS="api,auth,shared,observability,ml,quant,distributed"
# We skip dev dependencies for production image to speed up build
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install ".[${EXTRAS}]" && \
    uv pip install ./wheels/*.whl

# --- RUNTIME STAGE ---
FROM python:3.12.13-alpine AS latest

WORKDIR /app

# Install runtime dependencies
RUN apk add --no-cache curl openssl ca-certificates

# Environment Defaults
ENV PYTHONPATH=/app

# Copy installed packages from builder
COPY --from=builder /usr/local /usr/local

# Copy application source
COPY . .

# Set default command
CMD ["python", "api/index.py"]
