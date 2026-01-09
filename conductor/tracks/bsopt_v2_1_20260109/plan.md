# Implementation Plan: Black-Scholes Optimization Platform v2.1

## Phase 1: Infrastructure & Security Hardening [checkpoint: 7041813]
- [x] Task: Move hardcoded secrets to `.env` and configure Docker Compose to use environment variables
- [x] Task: Implement `redis-exporter` service in `docker-compose.yml` and verify metrics exposition
- [x] Task: Apply `no-new-privileges` security options and `read_only` root filesystems to core services
- [x] Task: Configure custom bridge networks (`bsopt-net`, `monitor-net`) for service isolation
- [x] Task: Verify all services start with healthy status and enforced resource limits
- [x] Task: Conductor - User Manual Verification 'Phase 1: Infrastructure & Security Hardening' (Protocol in workflow.md) (7041813)

## Phase 2: Observability Stack (LGTM) Configuration [checkpoint: 06324c9]
- [x] Task: Create `monitoring/prometheus/prometheus.yml` with scrape configs for all exporters and services
- [x] Task: Create `monitoring/promtail/promtail-config.yaml` and verify Docker log collection
- [x] Task: Configure Loki and Grafana provisioning (Datasources and basic Dashboards)
- [x] Task: Verify end-to-end metrics flow from services to Grafana dashboards
- [x] Task: Verify end-to-end log flow from Docker containers to Loki/Grafana
- [x] Task: Conductor - User Manual Verification 'Phase 2: Observability Stack (LGTM) Configuration' (Protocol in workflow.md) (06324c9)

## Phase 3: ML Pipeline Instrumentation (TDD)
- [x] Task: Write unit tests for `structlog` JSON formatting in `src/ml/autonomous_pipeline.py`
- [x] Task: Implement `structlog` configuration in the ML pipeline
- [x] Task: Write unit tests for Prometheus metrics (RMSE, PSI, Duration) logic
- [x] Task: Implement `prometheus_client` instrumentation and Pushgateway integration in `InstrumentedTrainer`
- [x] Task: Refactor `docker/Dockerfile.api` to use a multi-stage, non-root build 45678cc
- [ ] Task: Verify metrics appear in Prometheus after a manual pipeline execution
- [ ] Task: Conductor - User Manual Verification 'Phase 3: ML Pipeline Instrumentation (TDD)' (Protocol in workflow.md)

## Phase 4: CI/CD/CT Pipeline Implementation
- [ ] Task: Create `.github/workflows/mlops-pipeline.yml` with Trivy security scan and Ruff linting jobs
- [ ] Task: Configure integration test job with Postgres service container
- [ ] Task: Implement the Continuous Training (CT) job with mocked GPU steps and runner setup documentation
- [ ] Task: Implement the GitOps deployment step (build and mock push)
- [ ] Task: Verify the full GitHub Actions pipeline execution (success/failure triggers)
- [ ] Task: Conductor - User Manual Verification 'Phase 4: CI/CD/CT Pipeline Implementation' (Protocol in workflow.md)
