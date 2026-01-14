# syntax=docker/dockerfile:1
# STAGE 1: Builder (Compile dependencies)
FROM python:3.10-slim-bookworm as builder

WORKDIR /build

# Install system build deps (GCC, OpenMP for Qiskit/PyTorch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Optimize pip caching by copying only requirements first
# Use BuildKit cache mount for pip to speed up repeated local builds
COPY requirements/ml.txt ./requirements/
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements/ml.txt

# STAGE 2: Runtime (Slim & Secure)
FROM python:3.10-slim-bookworm as runtime

# Create non-root user
RUN groupadd -r algo && useradd -r -g algo algo_trader

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Runtime libraries only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source code separately to ensure that changing code doesn't invalidate dependency layers
COPY --chown=algo_trader:algo src/ ./src/

# Hardening
USER algo_trader

ENV QISKIT_PARALLEL=TRUE \
    OMP_NUM_THREADS=8 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

CMD ["python", "src/ml/autonomous_pipeline.py"]
