# Specification: Observability Stack (LGTM) Implementation

## Overview
This track involves setting up the full-stack observability infrastructure (Loki, Grafana, Tempo, Mimir equivalent) and instrumenting the core ML pipeline and API layer with structured logging and Prometheus metrics. This aligns the project with industrial-grade MLOps standards.

## Functional Requirements
1.  **Observability Infrastructure:**
    -   Configure and deploy Prometheus for metrics collection.
    -   Configure and deploy Loki for log aggregation.
    -   Configure and deploy Promtail for log shipping from Docker containers.
    -   Configure and deploy Grafana for visualization.
    -   Ensure all services are networked correctly in Docker Compose (`monitor-net`).
2.  **Instrumentation:**
    -   Update `src/ml/autonomous_pipeline.py` to use `structlog` for JSON-formatted logging.
    -   Implement Prometheus metrics (Counters, Gauges, Histograms) in `src/ml/autonomous_pipeline.py` to track:
        -   Training duration.
        -   Model RMSE.
        -   Data drift (PSI) score.
        -   Training errors.
    -   Ensure metrics are pushed to the Prometheus Pushgateway for batch ML jobs.
3.  **API Optimization:**
    -   Update `docker/Dockerfile.api` to a multi-stage build for security and size optimization.
    -   Enable Prometheus multiprocess mode for the FastAPI application.

## Non-Functional Requirements
-   **Performance:** Metrics collection should have minimal impact on training latency.
-   **Security:** API Docker image should use a non-root user and minimal runtime dependencies.
-   **Resilience:** Sidecar containers should have `restart: unless-stopped` or `restart: always`.

## Acceptance Criteria
-   `docker-compose up` successfully starts all observability services.
-   Grafana is accessible at `localhost:3001`.
-   Prometheus targets include the API and ML pipeline.
-   Loki receives structured JSON logs from the ML pipeline.
-   `src/ml/autonomous_pipeline.py` execution results in metrics appearing in Prometheus.
-   API Docker image size is reduced and passes security best practices.

## Out of Scope
-   Setting up Tempo for distributed tracing (reserved for a future track).
-   Production deployment to Kubernetes (GKE/EKS).
-   External secret management (HashiCorp Vault).
