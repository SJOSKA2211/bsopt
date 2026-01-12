# STAGE 1: Builder (Compile dependencies)
FROM python:3.10-slim-bookworm as builder

WORKDIR /build

# Install system build deps (GCC, OpenMP for Qiskit/PyTorch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into a virtual env to keep them isolated
COPY requirements/ml.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r ml.txt

# STAGE 2: Runtime (Slim & Secure)
FROM python:3.10-slim-bookworm as runtime

# Create non-root user (Security Best Practice)
RUN groupadd -r algo && useradd -r -g algo algo_trader

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Runtime libraries only (no GCC)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --chown=algo_trader:algo src/ ./src/

# Hardening
USER algo_trader

ENV QISKIT_PARALLEL=TRUE \
    OMP_NUM_THREADS=8 \
    PYTHONUNBUFFERED=1

CMD ["python", "src/ml/autonomous_pipeline.py"]