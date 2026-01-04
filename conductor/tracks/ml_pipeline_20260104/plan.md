# Track Plan: Build the Autonomous ML Pipeline

## Phase 1: Foundation and Data Ingestion [checkpoint: b7b96e6]
- [x] Task: Project structure and environment setup e428b2a
    - [x] Write Tests: Verify base directories and basic dependency loading.
    - [x] Implement: Create `src/ml`, `src/shared` and configure base Python environment.
- [x] Task: Resilient Data Scraper Implementation 9b32997
    - [x] Write Tests: Mock API responses and verify scraper retry logic and error handling.
    - [x] Implement: Create `MarketDataScraper` supporting Polygon and Yahoo Finance APIs.
- [x] Task: Hybrid Storage Implementation (TimescaleDB & MinIO) b507604
    - [x] Write Tests: Verify CRUD operations for TimescaleDB and Parquet file handling for MinIO.
    - [x] Implement: Setup SQLAlchemy models for "hot" data and MinIO client for "cold" storage.
- [x] Task: Conductor - User Manual Verification 'Foundation and Data Ingestion' (Protocol in workflow.md) b7b96e6

## Phase 2: Core ML Pipeline and Optimization [checkpoint: 7be2a22]
- [x] Task: Multi-Framework Training Loop Scaffolding c331ec4
    - [x] Write Tests: Verify XGBoost, PyTorch, and Scikit-learn model initialization and basic fit.
    - [x] Implement: Create modular training classes for supported frameworks.
- [x] Task: Optuna Integration for Hyperparameter Tuning 0e61696
    - [x] Write Tests: Verify Optuna study creation and successful trial completion.
    - [x] Implement: Integrate `InstrumentedTrainer` with Optuna optimization loops.
- [x] Task: Model Performance Drift Detection 38b98df
    - [x] Write Tests: Verify drift detection triggers when performance metrics degrade.
    - [x] Implement: Logic to track and compare current RMSE/MAE against rolling historical baselines.
- [x] Task: Conductor - User Manual Verification 'Core ML Pipeline and Optimization' (Protocol in workflow.md) 7be2a22

## Phase 3: Tracking and Observability Integration
- [x] Task: MLflow Experiment Tracking 725d837
    - [x] Write Tests: Verify logging of parameters, metrics, and model artifacts to MLflow.
    - [x] Implement: Detailed MLflow instrumentation including model lineage and feature importance plots.
- [x] Task: Prometheus Metrics and Structured Logging 5650bac
    - [x] Write Tests: Verify metric emission format and JSON log structure for Loki compliance.
    - [x] Implement: `structlog` and `prometheus-client` integration throughout the pipeline.
- [~] Task: Alerting Rules and Dashboard JSON
    - [ ] Write Tests: Verify alerting rule logic and dashboard schema validity.
    - [ ] Implement: Generate Grafana dashboard JSON and AlertManager configuration for pipeline health.
- [ ] Task: Conductor - User Manual Verification 'Tracking and Observability Integration' (Protocol in workflow.md)

## Phase 4: Pipeline Orchestration and Refinement
- [ ] Task: End-to-End Autonomous Pipeline Integration
    - [ ] Write Tests: Full integration test simulating the entire flow from scraping to model registry.
    - [ ] Implement: Create `autonomous_pipeline.py` script to orchestrate Phase 1-3.
- [ ] Task: Celery Grid Readiness and Final Audit
    - [ ] Write Tests: Verify the pipeline can be executed as a distributed Celery task.
    - [ ] Implement: Refactor for Celery worker execution and perform final coverage/quality gate checks.
- [ ] Task: Conductor - User Manual Verification 'Pipeline Orchestration and Refinement' (Protocol in workflow.md)
