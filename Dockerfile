# ==============================================================================
# BSOPT: Multi-Stage Production Dockerfile (v8.3 - CPU-Only)
# ==============================================================================

# --- Stage 1: Rust Core Builder ---
FROM python:3.12.13-slim AS rust-builder

RUN for i in 1 2 3; do apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libssl-dev \
    pkg-config \
    git \
    && break || sleep 5; done \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build/rust-core
COPY src/math_kernel/rust-core /build/rust-core
RUN pip install --no-cache-dir maturin
RUN PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin build --release --out /build/wheels

# --- Stage 2: Base Runtime Image ---
FROM python:3.12.13-slim AS runtime-base

RUN for i in 1 2 3; do apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    wget \
    && break || sleep 5; done \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# --- Stage 3: Python Dependency Builder (using uv for speed) ---
FROM runtime-base AS py-builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN for i in 1 2 3; do apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    git \
    && break || sleep 5; done \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

# Extract requirements and install with retries
RUN for i in 1 2 3 4 5; do \
    uv pip compile pyproject.toml --extra api --extra auth --extra shared --extra distributed --extra quant --extra observability -o requirements-all.txt && break || \
    if [ $i -lt 5 ]; then echo "uv compile attempt $i failed, retrying..." && sleep 5; else exit 1; fi; \
    done

RUN for i in 1 2 3 4 5; do \
    uv pip install --system --no-cache -r requirements-all.txt && break || \
    if [ $i -lt 5 ]; then echo "uv install attempt $i failed, retrying..." && sleep 5; else exit 1; fi; \
    done

# Copy Rust wheels from stage 1 and install
COPY --from=rust-builder /build/wheels/*.whl /tmp/
RUN uv pip install --system --no-cache /tmp/*.whl && rm /tmp/*.whl

# --- Stage 4: Final Optimized Runtime ---
FROM runtime-base AS final

# Copy installed python packages from py-builder
COPY --from=py-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=py-builder /usr/local/bin /usr/local/bin

# Copy source code
COPY . .

# Expose ports for various services
EXPOSE 50051 3001 8000

# Default entrypoint (overridden by docker-compose)
CMD ["python", "src/auth/grpc_server.py"]
