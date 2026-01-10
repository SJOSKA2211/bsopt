# --- Build Stage ---
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --user --no-cache-dir cython setuptools wheel && \
    pip install --user --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --user --no-cache-dir .

# --- Final Stage ---
FROM python:3.13-slim

WORKDIR /app

# Create a non-root user
RUN useradd -m -u 1000 appuser || true

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy source code
COPY src /app/src

# Create artifacts directory and set permissions
USER root
RUN mkdir -p /mlartifacts && chown -R 1000:1000 /mlartifacts
USER 1000

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["python", "src/ml/autonomous_pipeline.py"]
