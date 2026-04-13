# Use the hardened base image
FROM /home/kamau/bsopt/infrastructure/orchestration/Dockerfile.base AS builder

# Set working directory within the builder stage
WORKDIR /app/auth_api

# Copy application source code
COPY src/auth/ /app/auth_api/src/auth/
COPY src/common/ /app/auth_api/src/common/
COPY src/database/ /app/auth_api/src/database/
COPY src/shared/protos/ /app/auth_api/src/shared/protos/
COPY src/shared/ /app/auth_api/src/shared/
COPY requirements-auth.txt . # Copy specific requirements for auth service

# Install auth-specific Python dependencies using uv
# This assumes requirements-auth.txt lists dependencies not covered by base
RUN uv pip install --no-cache-dir -r requirements-auth.txt 
    && uv pip install --no-cache-dir -e . # Install the auth package itself if needed

# --- Runtime Stage ---
# Inherit from the production base image
FROM /home/kamau/bsopt/infrastructure/orchestration/Dockerfile.base AS production

# Set working directory in the final image
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application source code and configs from the builder
# Copy only necessary parts to keep the image lean
COPY --from=builder /app/auth_api/src/auth /app/src/auth
COPY --from=builder /app/auth_api/src/common /app/src/common
COPY --from=builder /app/auth_api/src/database /app/src/database
COPY --from=builder /app/auth_api/src/shared/protos /app/src/shared/protos
COPY --from=builder /app/auth_api/src/shared /app/src/shared

# Copy necessary configuration files or assets if not already in base
# Example: if .pki certificates are needed and not handled by compose mounts
# COPY .pki /app/.pki

# Ensure correct ownership
RUN chown -R Manifold:Manifold /app /opt/venv

# Expose the gRPC port
EXPOSE 50051

# Set the entrypoint and command to run the gRPC server
# Using tini to ensure proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "src/auth/grpc_server.py"]
