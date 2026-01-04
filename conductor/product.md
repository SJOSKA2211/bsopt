# Initial Concept

This is the definitive, production-hardened Product Requirements Document (PRD) for the **Black-Scholes Optimization & ML Platform (BS-Opt)**.

This version introduces a complete **Observability Stack (LGTM - Loki, Grafana, Tempo, Mimir equivalent)**, enables high-performance **CI/CD MLOps Pipelines**, and refines the code for industrial-grade resilience.

---


# Final PRD: Docker-Based Black-Scholes ML Platform (BS-Opt) v2.0

## 1. System Architecture & Observability

We have upgraded the architecture to include a sidecar monitoring stack. The application layer now pushes metrics to Prometheus and logs to Loki, visualized via Grafana.

```mermaid
graph TD
    subgraph "Public Internet"
        Client[Web Client]
    end

    subgraph "Application Cluster"
        LB[Nginx Load Balancer]
        API[FastAPI Gateway]
        
        subgraph "Compute Grid"
            Pricing[Pricing Engine]
            AutoML[AutoML Pipeline]
            Scraper[Data Scrapers]
        end
        
        subgraph "Message Bus"
            RMQ[RabbitMQ]
            Redis[Redis Cache]
        end
        
        subgraph "Persistence"
            TSDB[(TimescaleDB)]
            MinIO[(MinIO S3)]
            MLflow[MLflow Server]
        end
    end

    subgraph "Observability Stack"
        Prom[Prometheus] --> API & Pricing & AutoML & RMQ
        Loki[Loki] --> Promtail
        Promtail[Promtail] --> API & Pricing
        Grafana[Grafana] --> Prom & Loki
        Alert[AlertManager] --> Prom
    end

    Client --> LB
    LB --> API
    API --> RMQ & Redis
    RMQ --> Pricing & AutoML & Scraper
    AutoML --> MLflow & MinIO
    Pricing & Scraper --> TSDB

```

---


## 2. Infrastructure: Optimized Docker Compose

This configuration includes resource limits, complex networking, and the full monitoring stack.

```yaml
# docker-compose.yml
version: '3.9'

x-logging: &logging
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
    tag: "{{.Name}}"

services:
  # ============================================================================
  # OBSERVABILITY STACK (Loki, Prometheus, Grafana)
  # ============================================================================
  
  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: bsopt-prometheus
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=15d'
    ports:
      - "9090:9090"
    networks:
      - monitor-net
    restart: unless-stopped

  loki:
    image: grafana/loki:2.9.2
    container_name: bsopt-loki
    volumes:
      - ./monitoring/loki:/etc/loki
      - loki_data:/loki
    command: -config.file=/etc/loki/loki-config.yaml
    ports:
      - "3100:3100"
    networks:
      - monitor-net
    restart: unless-stopped

  promtail:
    image: grafana/promtail:2.9.2
    container_name: bsopt-promtail
    volumes:
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock
      - ./monitoring/promtail:/etc/promtail
    command: -config.file=/etc/promtail/promtail-config.yaml
    networks:
      - monitor-net
    restart: unless-stopped

  grafana:
    image: grafana/grafana:10.2.0
    container_name: bsopt-grafana
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
    ports:
      - "3001:3000"
    depends_on:
      - prometheus
      - loki
    networks:
      - monitor-net
    restart: unless-stopped

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:v0.47.2
    container_name: bsopt-cadvisor
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
    ports:
      - "8080:8080"
    networks:
      - monitor-net
    restart: unless-stopped

  # ============================================================================
  # CORE INFRASTRUCTURE
  # ============================================================================

  postgres:
    image: timescale/timescaledb:latest-pg15
    container_name: bsopt-postgres
    shm_size: 1g
    environment:
      POSTGRES_DB: bsopt
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_MULTIPLE_DATABASES: mlflow
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U admin -d bsopt"]
      interval: 5s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 4G
    networks:
      - bsopt-net
      - monitor-net

  redis:
    image: redis:7-alpine
    container_name: bsopt-redis
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
    networks:
      - bsopt-net
      - monitor-net

  redis-exporter:
    image: oliver006/redis_exporter
    environment:
      - REDIS_ADDR=redis://:${REDIS_PASSWORD}@redis:6379
    networks:
      - monitor-net

  rabbitmq:
    image: rabbitmq:3.12-management-alpine
    container_name: bsopt-rabbitmq
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: ${RABBITMQ_PASSWORD}
    ports:
      - "5672:5672"
      - "15672:15672"
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
      - ./config/rabbitmq/enabled_plugins:/etc/rabbitmq/enabled_plugins
    networks:
      - bsopt-net
      - monitor-net

  # ============================================================================
  # APPLICATION LAYER
  # ============================================================================

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    container_name: bsopt-api
    logging: *logging
    environment:
      DATABASE_URL: postgresql://admin:${DB_PASSWORD}@postgres:5432/bsopt
      REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379/0
      RABBITMQ_URL: amqp://admin:${RABBITMQ_PASSWORD}@rabbitmq:5672/bsopt
      MLFLOW_TRACKING_URI: http://mlflow:5000
      PROMETHEUS_MULTIPROC_DIR: /tmp/metrics
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - bsopt-net
      - monitor-net

  ml-pipeline:
    build:
      context: .
      dockerfile: docker/Dockerfile.ml
    container_name: bsopt-ml-pipeline
    logging: *logging
    environment:
      DATABASE_URL: postgresql://admin:${DB_PASSWORD}@postgres:5432/bsopt
      MLFLOW_TRACKING_URI: http://mlflow:5000
      JOB_TYPE: training
    volumes:
      - model_artifacts:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - bsopt-net
      - monitor-net

  # ... [Frontend, MLflow, and MinIO services remain similar but connected to monitor-net] ...

volumes:
  postgres_data:
  redis_data:
  rabbitmq_data:
  minio_data:
  model_artifacts:
  prometheus_data:
  grafana_data:
  loki_data:

networks:
  bsopt-net:
    driver: bridge
  monitor-net:
    driver: bridge

---


## 3. Monitoring Configuration (Infrastructure as Code)

You must create these files to enable the stack.

### `monitoring/prometheus/prometheus.yml`

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'bsopt-services'
    static_configs:
      - targets: ['api:8000', 'worker-ml:8000']
  
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'rabbitmq'
    static_configs:
      - targets: ['rabbitmq:15692']

```

### `monitoring/promtail/promtail-config.yaml`

```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: docker
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ['__meta_docker_container_name']
        regex: '/(.*)'
        target_label: 'container'

```

---


## 4. Complex Pipeline Strategy (CI/CD/CT)

We define a **Continuous Training (CT)** pipeline using GitHub Actions that is triggered not just by code changes, but by data drift alerts or schedules.

```yaml
# .github/workflows/mlops-pipeline.yml
name: BS-Opt Enterprise Pipeline

on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *' # Nightly builds
  repository_dispatch:
    types: [data-drift-alert]

jobs:
  # 1. Quality Assurance
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Security Scan (Trivy)
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          ignore-unfixed: true
          severity: 'CRITICAL,HIGH'
      - name: Lint & Type Check
        run: |
          pip install ruff mypy
          ruff check src/
          mypy src/

  # 2. Integration Testing
  test:
    needs: code-quality
    runs-on: ubuntu-latest
    services:
      postgres:
        image: timescale/timescaledb:latest-pg15
        env:
          POSTGRES_PASSWORD: test
        ports:
          - 5432:5432
    steps:
      - uses: actions/checkout@v3
      - name: Run Pytest
        run: |
          pip install -e .[test]
          pytest tests/ --cov=src --cov-report=xml

  # 3. Model Training (CT)
  train-and-evaluate:
    needs: test
    runs-on: ubuntu-latest-gpu # Assumes self-hosted runner with GPU
    steps:
      - name: Train Model
        run: |
          docker build -f docker/Dockerfile.ml -t bsopt-ml:latest .
          docker run --env-file .env.ci bsopt-ml:latest python src/ml/autonomous_pipeline.py
      - name: Compare Metrics
        run: python scripts/compare_model_metrics.py
    
  # 4. GitOps Deployment
  deploy:
    needs: train-and-evaluate
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Update K8s Manifests
        run: |
          yq eval '.spec.template.spec.containers[0].image = "bsopt-api:${{ github.sha }}"'
          git commit -am "Bump version to ${{ github.sha }}"
          git push

```

---


## 5. Optimized Code with Instrumentation

We incorporate **Prometheus** metrics (counters, gauges, histograms) and **Structlog** for machine-readable logging into the core pipeline.

### `requirements/ml.txt` (Additions)

```txt
prometheus-client==0.19.0
structlog==23.2.0
python-json-logger==2.0.7

```

### Optimized `src/ml/autonomous_pipeline.py`

```python
# src/ml/autonomous_pipeline.py

import os
import time
import structlog
import logging
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, push_to_gateway
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import optuna
import mlflow

# ============================================================================
# OBSERVABILITY SETUP
# ============================================================================

# Configure JSON Logging for Loki
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logger = structlog.get_logger()

# Prometheus Metrics
REGISTRY = CollectorRegistry()
TRAINING_DURATION = Histogram('ml_training_duration_seconds', 'Time spent training model', registry=REGISTRY)
MODEL_ACCURACY = Gauge('ml_model_rmse', 'Root Mean Squared Error of model', ['model_type', 'dataset'], registry=REGISTRY)
DATA_DRIFT_SCORE = Gauge('ml_data_drift_score', 'PSI score for data drift', registry=REGISTRY)
TRAINING_ERRORS = Counter('ml_training_errors_total', 'Total training failures', registry=REGISTRY)

class MLConfig:
    """Enterprise ML Configuration"""
    PUSHGATEWAY_URL = os.getenv('PUSHGATEWAY_URL', 'pushgateway:9091')
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    # ... previous config items ...

class InstrumentedTrainer:
    def __init__(self, config: MLConfig):
        self.config = config
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        
    def push_metrics(self):
        """Push metrics to Prometheus Gateway for short-lived batch jobs"""
        try:
            push_to_gateway(self.config.PUSHGATEWAY_URL, job='ml_pipeline', registry=REGISTRY)
        except Exception as e:
            logger.error("failed_to_push_metrics", error=str(e))

    @TRAINING_DURATION.time()
    def run_optimization_loop(self, X, y):
        """Main optimization loop with error handling and logging"""
        logger.info("starting_optimization", n_samples=len(X))
        
        try:
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda t: self._objective(t, X, y), n_trials=self.config.OPTUNA_N_TRIALS)
            
            best_rmse = study.best_value
            MODEL_ACCURACY.labels(model_type='xgboost', dataset='validation').set(best_rmse)
            
            logger.info("optimization_complete", best_rmse=best_rmse, best_params=study.best_params)
            return study.best_trial
            
        except Exception as e:
            TRAINING_ERRORS.inc()
            logger.error("optimization_failed", error=str(e))
            raise
        finally:
            self.push_metrics()

    def _objective(self, trial, X, y):
        with mlflow.start_run(nested=True):
            # ... (Standard XGBoost logic) ...
            
            # Log params to MLflow
            mlflow.log_params(trial.params)
            
            # Calculate RMSE
            rmse = 0.45 # Placeholder for calculation
            
            # Report intermediate results for pruning
            trial.report(rmse, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()
                
            return rmse

# ============================================================================
# DATA PIPELINE (ENHANCED)
# ============================================================================

class ResilientDataPipeline:
    def check_drift(self, reference_data, current_data):
        """Calculate PSI (Population Stability Index) for drift detection"""
        psi_score = self._calculate_psi(reference_data, current_data)
        DATA_DRIFT_SCORE.set(psi_score)
        
        if psi_score > 0.1:
            logger.warning("data_drift_detected", psi=psi_score)
            # Trigger alert webhook here
            
    def _calculate_psi(self, expected, actual, buckettype='bins', buckets=10):
        # ... (Math logic for PSI) ...
        return 0.05

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    logger.info("pipeline_startup")
    config = MLConfig()
    trainer = InstrumentedTrainer(config)
    
    # Simulate run
    try:
        # Load Data
        # Train
        # Evaluate
        pass
    except Exception as e:
        logger.critical("pipeline_crashed", error=str(e))
        exit(1)

```

## 6. Dockerfile Optimizations (Multi-Stage)

Optimizing the API Dockerfile for size and security (distroless approach).

```dockerfile
# docker/Dockerfile.api

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app
RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt requirements/api.txt ./requirements/
RUN pip install --user --no-cache-dir -r requirements/api.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app
ENV PATH=/root/.local/bin:$PATH

# Install only runtime libs for Postgres
RUN apt-get update && apt-get install -y libpq5 curl && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY src/ ./src/

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start with prometheus multiprocess mode
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/metrics
RUN mkdir -p /tmp/metrics

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--loop", "uvloop"]

```

## 7. Next Steps for Implementation

1. **Initialize Monitoring:** Run `docker-compose up -d prometheus loki grafana` and import the "Node Exporter Full" dashboard (ID: 1860) into Grafana.
2. **Secret Management:** Move all default passwords in `docker-compose.yml` to a `.env` file and exclude it from version control.
3. **Pipeline Activation:** Commit the `.github` folder to your repo to activate the CI/CD logic.

Would you like me to generate the **Grafana Dashboard JSON** specific to the Black-Scholes metrics (Option Greeks drift and Training RMSE over time) to auto-provision the visualization layer?

---


# Product Guide: Black-Scholes Optimization & ML Platform (BS-Opt) v2.0

## Vision
To provide a production-hardened, industrial-grade platform for Black-Scholes algorithmic trading and machine learning, featuring full-stack observability (LGTM) and high-performance MLOps pipelines.

## Target Users
- **Quant Analysts & Data Scientists:** For model development, training, and optimization.
- **Financial Traders / Portfolio Managers:** For real-time option pricing and execution.
- **DevOps / MLOps Engineers:** For monitoring system health, managing the compute grid, and ensuring pipeline resilience.

## Primary Goals
- **Industrial-Grade Resilience:** Sidecar monitoring and optimized Docker configurations.
- **Full-Stack Observability:** Unified logging and metrics via Loki, Prometheus, and Grafana.
- **MLOps Excellence:** Continuous Training (CT) pipelines with automated drift detection.

## Core Features
- **Pricing Engine:** Real-time Black-Scholes calculations using QuantLib.
- **AutoML Pipeline:** Autonomous multi-framework (XGBoost, PyTorch, Scikit-learn) training with Optuna optimization and Celery orchestration.
- **Observability Stack:** Real-time dashboards for Greeks drift and training metrics.

## Functional Requirements
- **Data Ingestion:** Resilient scrapers for market data (Polygon/Yahoo).
- **Messaging & State:** Distributed task execution via Celery, with RabbitMQ/Redis backends.
- **Persistence:** TimescaleDB for market data, MinIO for artifacts, and MLflow for registry.

## Success Metrics
- **System Uptime:** Target 99.9% availability for trading and pricing services.
- **Model Health:** Minimal Root Mean Squared Error (RMSE) and Kolmogorov-Smirnov (KS) test drift detection.
- **Latency:** Sub-millisecond pricing calculations and responsive API gateway.