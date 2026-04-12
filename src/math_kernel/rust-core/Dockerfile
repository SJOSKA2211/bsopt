# Stage 1: Rust Core Builder
FROM rust:1.77-alpine as rust_builder

# Install build dependencies for Rust core (e.g., pkg-config, openssl-dev)
RUN apk add --no-cache 
    pkg-config 
    openssl-dev 
    libffi-dev 
    && rm -rf /var/cache/apk/*

WORKDIR /build/rust-core

# Copy the Rust project's Cargo.toml and Cargo.lock first to cache dependencies
COPY src/math_kernel/rust-core/Cargo.toml ./
COPY src/math_kernel/rust-core/Cargo.lock ./

# Build dependencies (creates Cargo.lock if it doesn't exist)
RUN cargo build --release --bins --jobs $(nproc) --target x86_64-unknown-linux-musl --profile=release

# Copy the rest of the Rust source code
COPY src/math_kernel/rust-core/src ./src/

# Build the Rust core for release
RUN cargo build --release --bins --jobs $(nproc) --target x86_64-unknown-linux-musl --profile=release

# Stage 2: Final Runtime
# Use a minimal base image for the runtime, e.g., Alpine
FROM alpine:latest as runtime

# Install runtime dependencies if any (e.g., libssl)
RUN apk add --no-cache 
    openssl 
    libffi 
    && rm -rf /var/cache/apk/*

WORKDIR /app

# Copy the compiled Rust binary from the builder stage
# Adjust the path based on the actual binary name and build output location
COPY --from=rust_builder /build/rust-core/target/x86_64-unknown-linux-musl/release/manifold-core /usr/local/bin/manifold-core

# Copy any necessary configuration files
# Example: COPY config/manifold-core/config.yaml /etc/manifold-core/config.yaml

# Expose ports if the Rust core service listens on any
# EXPOSE 8080

# Set the entrypoint to run the compiled Rust binary
ENTRYPOINT ["/usr/local/bin/manifold-core"]
