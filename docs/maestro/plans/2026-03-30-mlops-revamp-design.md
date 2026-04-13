---
design_depth: deep
task_complexity: complex
---

# Design Document: BS-OPT MLOps Revamp (Integrated Health Mesh)

## 1. Problem Statement
The BS-OPT MLOps system currently suffers from architectural fragmentation and technical debt. Critical import paths are broken (specifically in the feature store), rendering the core ML pipeline unstable. Existing tests reference legacy orchestrator classes that no longer exist in the source tree. Furthermore, the autonomous detection and remediation capabilities are under-utilized due to a lack of unified visibility—there is no centralized "Health Report" that combines model drift (MLflow), system metrics (Prometheus), and autonomous engine state. This initiative aims to revamp the engine into an Integrated Health Mesh that provides real-time diagnostic reporting and fully automated self-healing in a dockerized sandbox environment.

## 2. Requirements

### Functional Requirements
- **REQ-1**: Repair all broken import paths in `src/ml/feature_store/` and unify `Feature` definitions.
- **REQ-2**: Update unit and integration tests to align with the current `AutonomousEngine` architecture.
- **REQ-3**: Implement a `HealthReporter` class that aggregates state from MLflow (model status), Prometheus (system health), and Redis (anomaly history).
- **REQ-4**: Expose a REST API endpoint `/ml/health` in the ML service for structured diagnostic reports.
- **REQ-5**: Create a CLI tool `bin/ml-health` for manual health inspections.
- **REQ-6**: Enable and verify the Transformer-based anomaly detection engine in `AutonomousEngine`.

### Non-Functional Requirements
- **Observability**: High-fidelity logging and structured status reporting.
- **Automation**: Fully autonomous retraining and remediation (Auto-Heal).
- **Environment**: Must operate correctly in a Docker Compose based sandbox.

## 3. Approach

### Selected Approach: Integrated Health Mesh
The revamp will focus on unifying the detection, remediation, and reporting layers into a cohesive mesh.

**Key Design Decisions:**
- **Centralized Health Reporter** — We will introduce a `src/ml/aiops/health_reporter.py` to aggregate disparate signals. *[Rationale: Simplifies reporting logic by decoupling data collection from presentation. Traces To: REQ-3]*
- **Dual-Mode Interface** — A unified schema will power both the `/ml/health` API and the `bin/ml-health` CLI. *[Rationale: Ensures consistency between developer tools and automated monitors. Traces To: REQ-4, REQ-5]*
- **Transformer-First Detection** — We will transition the default anomaly detection to use the Transformer-based sequential analyzer. *[Rationale: Provides better capture of temporal patterns in trading metrics compared to simple statistical isolation. Traces To: REQ-6]*
- **Incremental Repair** — We will apply a "Fix-as-we-Go" strategy for imports and tests, ensuring each fix is validated by the revamped engine. *[Rationale: Minimizes disruption while ensuring structural health. Traces To: REQ-1, REQ-2]*

### Alternatives Considered
- **MLflow-Only Reporting** *(considered: rejected because MLflow doesn't capture real-time system metrics like Prometheus)*
- **External Dashboard (Grafana only)** *(considered: rejected because we need a programmatic way for the engine to verify its own health during remediation cycles)*

## 4. Architecture

### Component Layout
1.  **Orchestration Layer**: `AutonomousEngine` & `AutonomousGuardian` manage the loop.
2.  **Reporting Layer**: `HealthReporter` aggregates data from:
    - `MLflowClient`: Model stage, drift metrics, last run status.
    - `PrometheusClient`: CPU/Mem, latency (p95), error rates.
    - `RedisCache`: Recent anomaly history and remediation results.
3.  **Interface Layer**:
    - `FastAPI (main.py)`: Exposes `GET /ml/health`.
    - `CLI (bin/ml-health)`: Shell script/Python entry point for local diagnostics.
4.  **Detection Layer**: `AnomalyDetector` using the `Transformer` engine.

### Data Flow
- `AutonomousEngine` runs detection cycle → results saved to Redis.
- `HealthReporter` queries MLflow, Prometheus, and Redis on-demand.
- `/ml/health` returns a JSON object following the diagnostic schema.

## 5. Agent Team
- **coder**: Responsible for the core revamp of `AutonomousEngine` and the new `HealthReporter` class.
- **refactor**: Will handle the surgical repair of import paths in `src/ml/feature_store/` and unify the `Feature` classes.
- **tester**: Tasked with updating the legacy tests in `tests/unit/test_aiops_engine.py` and creating new integration tests for the Health Mesh.
- **devops_engineer**: Ensures the environment (Docker sidecars) is correctly configured for Prometheus/MLflow integration.
- **code_reviewer**: Performs the final quality gate on all MLOps changes.

## 6. Risk Assessment

| Risk | Impact | Mitigation Strategy |
|------|--------|--|
| **Sidecar Connectivity** | High | Use robust retry logic in `PrometheusClient` and provide mock fallbacks for local-only execution. |
| **Import Circularity** | Medium | Centralize base classes in `src/ml/feature_store/base.py` and strictly follow the new import pattern. |
| **Detection Flakiness** | Medium | Calibrate Transformer thresholds during the initial implementation phase using historical data. |
| **Service Downtime** | Low | Implement all changes in a dockerized sandbox environment first before any promotion logic is activated. |

## 7. Success Criteria
1.  **Green Tests**: `pytest tests/unit/test_aiops_engine.py` passes without import errors or mock failures.
2.  **API Diagnostic**: `curl http://localhost:8000/ml/health` returns a valid JSON diagnostic object containing MLflow and Prometheus metrics.
3.  **CLI Discovery**: `bin/ml-health` correctly prints the current health summary to the terminal.
4.  **Auto-Heal Verified**: `AutonomousEngine` successfully detects a simulated drift/anomaly and records the remediation attempt in Redis.
5.  **Healthy Report**: The final report from the revamped engine indicates "System Health: Optimal" (or accurately identifies remaining blockers).
