# ==============================================================================
# BSOPT: Multi-Stage Production Dockerfile (v8.3 - CPU-Only - Alpine Optimized)
# ==============================================================================

# --- Stage 1: Rust Core Builder ---
# Use a slim Alpine base for Rust build dependencies
FROM python:3.12.13-alpine AS rust-builder

# Install build tools and Python for maturin
RUN apk add --no-cache 
    build-base 
    curl 
    openssl-dev 
    pkgconfig 
    git 
    python3 
    python3-dev 
    && rm -rf /var/lib/apt/lists/* # Clean up apt cache (though apk doesn't use it, good practice for consistency)

# Install Rust
ENV RUSTUP_HOME=/usr/local/rustup 
    CARGO_HOME=/usr/local/cargo 
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal --no-modify-path 
    && rustup default stable

# Install target for musl if needed (for cross-compilation, though building on Alpine should be native)
# RUN rustup target add x86_64-unknown-linux-musl

WORKDIR /build/rust-core
COPY src/math_kernel/rust-core /build/rust-core
RUN pip install --no-cache-dir maturin
# Removed PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 as it might be legacy. Modern PYO3 should handle this.
# Target x86_64-unknown-linux-musl is appropriate for Alpine Linux.
RUN maturin build --release --target x86_64-unknown-linux-musl --out /build/wheels

# --- Stage 2: Base Runtime Image (Alpine) ---
FROM python:3.12.13-alpine AS runtime-base

RUN apk add --no-cache 
    openssl 
    ca-certificates 
    wget 
    python3 
    python3-dev 
    && rm -rf /var/lib/apt/lists/* # Clean up cache

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# --- Stage 3: Python Dependency Builder (using uv for speed) ---
FROM runtime-base AS py-builder

# Install uv directly into the image
RUN curl -LsSf https://uv-rs.github.io/release/latest/x86_64-unknown-linux-musl/uv.tar.gz | tar xz -C /usr/local/bin --strip-components=1

RUN apk add --no-cache 
    build-base 
    openssl-dev 
    git 
    python3 
    python3-dev 
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

# Extract requirements and install with retries
RUN for i in 1 2 3 4 5; do 
    uv pip compile pyproject.toml --extra api --extra auth --extra shared --extra distributed --extra quant --extra observability -o requirements-all.txt && break || 
    if [ $i -lt 5 ]; then echo "uv compile attempt $i failed, retrying..." && sleep 5; else exit 1; fi; 
    done

RUN for i in 1 2 3 4 5; do 
    uv pip install --system --no-cache -r requirements-all.txt && break || 
    if [ $i -lt 5 ]; then echo "uv install attempt $i failed, retrying..." && sleep 5; else exit 1; fi; 
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
