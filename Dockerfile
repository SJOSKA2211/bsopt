# Manifold: UNIFIED CPU-ONLY BASE (v14.0)
# Optimized for pure CPU execution on distroless/slim

# --- STAGE 1: BUILDER ---
FROM python:3.12.13-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_SYSTEM_PYTHON=0 \
    VIRTUAL_ENV=/opt/venv

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

# Create virtual environment
RUN uv venv /opt/venv

# Compile Rust Core Kernel (Performance Substrate)
COPY src/math_kernel/rust-core ./rust-core
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install maturin[patchelf] && \
    cd rust-core && maturin build --release --out ../wheels

# Install Python dependencies into virtual environment
COPY pyproject.toml uv.lock ./
ARG EXTRAS="api,auth,shared,observability,ml,quant,distributed"
# We skip dev dependencies for production image to speed up build
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install ".[${EXTRAS}]" && \
    uv pip install ./wheels/*.whl

# --- RUNTIME STAGE ---
FROM python:3.12.13-slim AS latest

WORKDIR /app

# Install runtime dependencies (curl needed for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Environment Defaults
ENV PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application source
COPY . .

# Set default command
CMD ["python", "api/index.py"]