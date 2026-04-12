# Stage 1: Build Rust Core
FROM python:3.11-alpine as builder

RUN apk add --no-cache 
    build-essential 
    curl 
    openssl-dev 
    pkgconfig 
    && rm -rf /var/cache/apk/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
COPY src/math_kernel/rust-core /build/rust-core
WORKDIR /build/rust-core
RUN pip install maturin
RUN maturin build --release --out /build/wheels

# Stage 2: Final Runtime - Python API Server (Auth gRPC server)
FROM python:3.11-alpine as runtime

# Install Python dependencies
# Copy requirements.txt first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install grpcio grpcio-tools

# Install Rust Core wheel
COPY --from=builder /build/wheels/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Copy source code
COPY . .

# Environment setup
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Ensure the app directory exists and set it as WORKDIR
WORKDIR /app

# Expose gRPC port
EXPOSE 50051

# Default entrypoint (overridden by compose)
CMD ["python", "src/auth/grpc_server.py"]
