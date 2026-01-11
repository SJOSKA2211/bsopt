# Specification: AIOps & Self-Healing (Anomaly Detection and Automated Remediation)

## Overview
Implement an intelligent, autonomous AIOps engine that monitors the BS-Opt ecosystem for performance anomalies and data drift. This system will utilize Machine Learning (Isolation Forests and Autoencoders) to identify deviations from normal behavior across API, ML, and Streaming layers, and execute automated remediation tasks (restarts, retrains, cache purges) to maintain high availability and accuracy.

## Functional Requirements
1.  **AIOps Orchestrator Service**: A standalone Python service responsible for the detection-remediation loop.
2.  **ML-Driven Anomaly Detection**:
    *   **Isolation Forest**: For detecting outliers in individual metrics (e.g., API latency, memory usage).
    *   **Autoencoder**: For multivariate anomaly detection across system-wide health signals.
3.  **Cross-Layer Detection Scope**:
    *   **API**: High error rates (5xx) and p95 latency spikes.
    *   **ML**: Data drift (PSI/KS scores) and training pipeline failures.
    *   **Infrastructure**: Resource exhaustion (CPU/Memory) and Kafka consumer lag.
4.  **Automated Remediation Engine**:
    *   **Docker Integration**: Capability to restart unhealthy containers via Docker SDK.
    *   **ML Integration**: Automated triggering of the `AutonomousMLPipeline` upon drift detection.
    *   **State Management**: Capability to purge Redis caches if stale data is detected.
    *   **Mocked Scaling**: Logic to simulate service scaling under heavy load.
5.  **Observability Loop**: Feed remediation events back into Grafana dashboards and log actions to Loki for auditable self-healing history.

## Non-Functional Requirements
1.  **Reliability**: The AIOps service itself must be lightweight and fail-safe (it should not cause more harm than the issues it fixes).
2.  **Latency**: Detection-to-remediation loop should complete within 60 seconds.
3.  **Traceability**: Every automated action must be logged with the "Why" (the specific anomaly detected).

## Acceptance Criteria
1.  The AIOps service successfully pulls metrics from Prometheus and identifies simulated anomalies.
2.  Simulating a high error rate in the API service triggers an automated restart of the container.
3.  Injecting "drifted" data results in an automated trigger of the ML retraining pipeline.
4.  Remediation events are visible in a dedicated "Self-Healing" panel in Grafana.

## Out of Scope
1.  Full Kubernetes operator implementation (remediation will focus on Docker/Docker Compose).
2.  Complex predictive scaling (forecasting future load days in advance).
