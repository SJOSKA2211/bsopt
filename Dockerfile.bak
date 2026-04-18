FROM python:3.12.13-slim as rust-builder

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libssl-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build/rust-core
COPY src/math_kernel/rust-core /build/rust-core
RUN pip install --no-cache-dir maturin
RUN maturin build --release --out /build/wheels

# --- Stage 2: Base Runtime Image ---
FROM python:3.12.13-slim as runtime-base

RUN apt-get update --fix-missing && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# --- Stage 3: Python Dependency Builder (using uv for speed) ---
FROM runtime-base as py-builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN apt-get update --fix-missing && apt-get install -y \
    build-essential \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
# Extract requirements and install
RUN uv pip compile pyproject.toml --extra api --extra auth --extra shared --extra distributed --extra quant --extra observability -o requirements-all.txt
RUN uv pip install --system --no-cache -r requirements-all.txt

# Copy Rust wheels from stage 1 and install
COPY --from=rust-builder /build/wheels/*.whl /tmp/
RUN uv pip install --system --no-cache /tmp/*.whl && rm /tmp/*.whl

# --- Stage 4: Final Optimized Runtime ---
FROM runtime-base as final

# Copy installed python packages from py-builder
COPY --from=py-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=py-builder /usr/local/bin /usr/local/bin

# Copy source code
COPY . .

# Expose ports for various services
EXPOSE 50051 3001 8000

# Default entrypoint (overridden by docker-compose)
CMD ["python", "src/auth/auth_server.py"]
