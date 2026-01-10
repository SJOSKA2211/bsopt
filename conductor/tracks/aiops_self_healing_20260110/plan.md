# Plan: AIOps & Self-Healing (Track: aiops_self_healing_20260110)

## Phase 1: Foundation & Observability Loop (TDD) [checkpoint: 46f596a]
- [x] Task: Create `src/aiops` service structure and configure Prometheus connectivity 62528d7
- [x] Task: Write TDD tests for `PrometheusClient` wrapper (fetching 5xx and latency metrics) b425093
- [x] Task: Implement `PrometheusClient` with robust error handling and logging to Loki 7425565
- [x] Task: Configure Grafana "Self-Healing" dashboard with initial empty state fc94187
- [x] Task: Conductor - User Manual Verification 'Phase 1: Foundation' (Protocol in workflow.md) 65c0fbc

## Phase 2: ML-Driven Anomaly Detection (TDD) [checkpoint: d012261]
- [x] Task: Write TDD tests for `IsolationForestDetector` (univariate outlier detection) be60db7
- [x] Task: Implement `IsolationForestDetector` using `scikit-learn` for latency/resource metrics be60db7
- [x] Task: Write TDD tests for `AutoencoderDetector` (multivariate anomaly detection) d34de9d
- [x] Task: Implement `AutoencoderDetector` using `PyTorch` for system-wide health signals d34de9d
- [x] Task: Implement detection logic for "ML Data Drift" (integrating PSI/KS scores from ML pipeline) 4060457
- [ ] Task: Conductor - User Manual Verification 'Phase 2: ML Detection' (Protocol in workflow.md)

## Phase 3: Automated Remediation Engine (TDD)
- [ ] Task: Write TDD tests for `DockerRemediator` (simulating service restarts)
- [ ] Task: Implement `DockerRemediator` using the Docker SDK to restart unhealthy containers
- [ ] Task: Write TDD tests for `MLPipelineTrigger` (triggering retraining)
- [ ] Task: Implement `MLPipelineTrigger` to invoke `AutonomousMLPipeline` upon drift detection
- [ ] Task: Implement `RedisRemediator` for automated cache purges
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Remediation Engine' (Protocol in workflow.md)

## Phase 4: Integration & Verification
- [ ] Task: Implement the main `AIOpsOrchestrator` loop (Detect -> Analyze -> Remediate)
- [ ] Task: Integrate "Remediation Events" with Grafana annotations and Loki logs
- [ ] Task: Simulate E2E failure scenarios (API spike, ML drift) and verify automated recovery
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Full Integration' (Protocol in workflow.md)
