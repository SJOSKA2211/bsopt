# Track Spec: Build the Autonomous ML Pipeline

## Overview
This track focuses on implementing the end-to-end autonomous machine learning pipeline for the BS-Opt platform. It includes resilient data ingestion, a hybrid persistence layer (TimescaleDB/MinIO), an automated training and optimization loop supporting multiple frameworks (XGBoost, PyTorch, Scikit-learn), and a comprehensive observability and tracking integration.

## Goals
- Establish a resilient market data ingestion layer.
- Implement a hybrid storage strategy for time-series and historical data.
- Develop an autonomous training loop with Optuna-based hyperparameter optimization.
- Integrate MLflow for experiment tracking and model lineage.
- Instrumentation for full-stack observability (Prometheus, Structlog).

## Functional Requirements
### 1. Data Ingestion & Persistence
- **Scrapers:** Implement resilient scrapers for Polygon/Yahoo APIs with retry logic.
- **Hybrid Storage:**
    - **TimescaleDB:** Store recent "hot" market data for fast SQL access.
    - **MinIO:** Offload historical/cold data as Parquet/CSV files for batch processing.

### 2. Autonomous Training Loop
- **Multi-Framework Support:** Implement training loops for XGBoost, PyTorch (LSTMs), and Scikit-learn baseline models.
- **Optimization:** Utilize Optuna for automated hyperparameter tuning across all supported frameworks.
- **Drift Detection:** Implement Model Performance Drift monitoring (detecting degradation in RMSE/MAE over time).

### 3. Tracking & Observability
- **MLflow Integration:**
    - Log tuned hyperparameters and fixed configurations.
    - Log performance metrics (RMSE, MAE) and performance drift scores.
    - Store model artifacts (.bin/.pt) and feature importance plots.
    - Track data lineage (dataset version/timestamp range).
- **Instrumentation:**
    - **Prometheus:** Emit gauges and histograms for training duration and model RMSE.
    - **Structlog:** Implement structured JSON logging for all pipeline events.
    - **Alerting:** Define basic AlertManager rules for training failures or significant performance drift.

## Non-Functional Requirements
- **Resilience:** The pipeline must handle API outages and database reconnections gracefully.
- **Observability:** Logs must be machine-readable (JSON) for Loki indexing.
- **Scalability:** The training loop should be Docker-ready for execution in the Celery compute grid.

## Acceptance Criteria
- [ ] Data scrapers successfully persist data to both TimescaleDB and MinIO.
- [ ] Optuna optimization loop completes successfully for an XGBoost model.
- [ ] MLflow dashboard displays all logged parameters, metrics, and plots for each run.
- [ ] Prometheus metrics are visible at the `/metrics` endpoint during/after training.
- [ ] Performance drift correctly triggers a warning log/metric update.
