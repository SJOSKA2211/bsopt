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

# Copy installed packages from builder to the appuser's home directory
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy source code
COPY src /app/src

# Create artifacts directory and set permissions
RUN mkdir -p /mlartifacts && \
    chown -R appuser:appuser /mlartifacts /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

USER appuser

CMD ["python", "src/ml/autonomous_pipeline.py"]
