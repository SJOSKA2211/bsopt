# Implementation Plan: Observability Stack (LGTM)

## Phase 1: Observability Infrastructure Setup
- [x] Task: Create monitoring configuration directories (`monitoring/prometheus`, `monitoring/loki`, `monitoring/promtail`, `monitoring/grafana`) (2d76e0f)
- [x] Task: Create `monitoring/prometheus/prometheus.yml` with service scrape configs (45f9d3b)
- [x] Task: Create `monitoring/promtail/promtail-config.yaml` for Docker log shipping (4a6f1a4)
- [x] Task: Update `docker-compose.yml` with Prometheus, Loki, Promtail, Grafana, and cAdvisor services (f0f77c2)
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Observability Infrastructure Setup' (Protocol in workflow.md)

## Phase 2: ML Pipeline Instrumentation (TDD)
- [ ] Task: Write unit tests for `InstrumentedTrainer` metrics logic and `structlog` configuration
- [ ] Task: Implement `structlog` configuration and Prometheus metrics in `src/ml/autonomous_pipeline.py`
- [ ] Task: Write unit tests for `ResilientDataPipeline` drift detection metrics
- [ ] Task: Implement drift metrics (PSI) in `src/ml/autonomous_pipeline.py`
- [ ] Task: Conductor - User Manual Verification 'Phase 2: ML Pipeline Instrumentation' (Protocol in workflow.md)

## Phase 3: API & Docker Optimization
- [ ] Task: Write unit tests for API metrics endpoint and instrumentation
- [ ] Task: Implement Prometheus instrumentation in the FastAPI API layer
- [ ] Task: Refactor `docker/Dockerfile.api` to use a multi-stage, distroless-style build
- [ ] Task: Conductor - User Manual Verification 'Phase 3: API & Docker Optimization' (Protocol in workflow.md)

## Phase 4: Final Integration & Dashboards
- [ ] Task: Verify end-to-end log flow from ML Pipeline to Loki/Grafana
- [ ] Task: Verify end-to-end metrics flow from Pushgateway to Prometheus/Grafana
- [ ] Task: Import basic "Node Exporter Full" and custom BS-Opt metrics dashboards into Grafana
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Final Integration & Dashboards' (Protocol in workflow.md)
