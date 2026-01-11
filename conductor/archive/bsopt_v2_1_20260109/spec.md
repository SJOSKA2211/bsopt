# Specification: Black-Scholes Optimization Platform v2.1 Implementation

## Overview
This track implements the "Final, Production-Grade PRD v2.1" for the Black-Scholes Optimization Platform. The primary goals are to establish a robust "Sidecar Observability Pattern" using the LGTM stack (Loki, Grafana, Prometheus, Promtail), harden the infrastructure security, and establish a Continuous Training (CT) pipeline.

## Functional Requirements

### 1. Infrastructure & Security
-   **Docker Compose:** Implement the v2.1 `docker-compose.yml` with:
    -   **Observability Stack:** Prometheus, Loki, Promtail, Grafana.
    -   **Core Infrastructure:** Postgres (TimescaleDB), Redis, RabbitMQ.
    -   **Application Layer:** API (FastAPI), ML Pipeline.
    -   **Exporters:** `redis-exporter` (Critical addition).
-   **Security Hardening:**
    -   Apply `security_opt: [no-new-privileges:true]` to all relevant services.
    -   Enforce read-only root filesystems where applicable.
    -   Implement strict secret management (Docker secrets or Vault integration preferred over plain env vars).
    -   Define strict network policies (custom bridge networks `bsopt-net` and `monitor-net`) to isolate monitoring traffic.

### 2. Observability (LGTM Stack)
-   **Configuration:**
    -   Create `monitoring/prometheus/prometheus.yml` to scrape API, ML workers, Redis, and RabbitMQ.
    -   Create `monitoring/promtail/promtail-config.yaml` to ship Docker container logs to Loki.
    -   Create `monitoring/loki/loki-config.yaml` and `grafana/provisioning` setups.
-   **Instrumentation:**
    -   Refactor `src/ml/autonomous_pipeline.py` to use `structlog` for structured JSON logging (consumed by Loki).
    -   Integrate `prometheus_client` to expose metrics:
        -   `ml_training_duration_seconds` (Histogram)
        -   `ml_model_rmse` (Gauge)
        -   `ml_data_drift_score` (Gauge)
        -   `ml_training_errors_total` (Counter)
    -   Implement `pushgateway` integration for batch ML jobs.

### 3. CI/CD/CT Pipeline
-   **GitHub Actions:** Create `.github/workflows/mlops-pipeline.yml` with:
    -   **Quality Gate:** Trivy security scan (CRITICAL/HIGH severity) and Ruff linting.
    -   **Integration Test:** Run Pytest with a service container for Postgres.
    -   **Continuous Training (CT):**
        -   Configure the job to support a self-hosted GPU runner (`self-hosted-gpu-runner`).
        -   **Fallback:** Include logic/documentation to mock GPU steps on standard runners if the self-hosted runner is unavailable.
        -   Include a step/doc to verify or set up the runner environment.
    -   **Deployment:** GitOps-style image build and push (mocked or registry-ready).

## Non-Functional Requirements
-   **Resilience:** All services must have `restart: always` or `on-failure` and configured `healthcheck` blocks.
-   **Performance:** The API and ML pipeline must run as non-root users (`appuser`).
-   **Maintainability:** Infrastructure as Code (IaC) principles for all monitoring configs.

## Acceptance Criteria
-   `docker-compose up` starts all services (including the new `redis-exporter`) without security privilege warnings.
-   Grafana (`localhost:3001`) displays metrics from Prometheus and logs from Loki.
-   The ML pipeline runs, pushes metrics to the Pushgateway, and logs structured JSON that appears in Loki.
-   The GitHub Actions pipeline passes locally (using `act` or similar) or on the repository, including the security scan and mocked training step.
-   Secrets are not exposed in plain text in `docker-compose.yml` (moved to `.env` or secrets manager).

## Out of Scope
-   Distributed Tracing (Tempo) implementation (deferred to a future track).
-   Production Kubernetes deployment (this track focuses on the Docker Compose environment).
