# Manifold: UNIFIED CPU-ONLY BASE (v13.0)
# Optimized for pure CPU execution on Debian Slim

# --- STAGE 1: BUILDER ---
FROM python:3.12.13-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1

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
    cd rust-core && uv pip install maturin && maturin build --release --out ../wheels

# Install Python dependencies into a temporary location
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install ".[api,auth,shared,observability,ml,quant,distributed,dev]" && \
    uv pip install ./wheels/*.whl

# --- RUNTIME STAGE ---
FROM python:3.12.13-slim AS runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY . .

# Environment Defaults
ENV PORT=3001 \
    PYTHONPATH=/app \
    ENVIRONMENT=production \
    LOG_LEVEL=info

EXPOSE 3001 5001 50051

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["python", "api/auth_index.py"]
