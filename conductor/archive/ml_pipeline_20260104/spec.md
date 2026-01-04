# Track Spec: Build the Autonomous ML Pipeline

## Overview
This track aims to implement the end-to-end machine learning lifecycle for the BS-Opt platform. It encompasses resilient data ingestion, a hybrid persistence layer, an autonomous model optimization loop, and full-stack experiment tracking and observability.

## Goals
- Establish a resilient data ingestion layer for financial market data.
- Implement a hybrid storage solution using TimescaleDB and MinIO.
- Develop an autonomous model training loop with hyperparameter optimization and drift detection.
- Integrate experiment tracking and pipeline observability.

## Functional Requirements
### 1. Data Ingestion & Scrapers
- Implement scrapers for Polygon and Yahoo Finance APIs.
- Prioritize fetching Daily OHLCV data, Option Chains (Greeks and IV), and Macro Indicators (Interest rates, Dividend yields).
- Implement robust retry logic and error handling for API rate limits and network issues.

### 2. Hybrid Persistence Layer
- Store recent/hot market data in **TimescaleDB** for high-performance querying.
- Offload historical/cold data to **MinIO** in efficient formats (e.g., Parquet).
- Provide a unified interface for data retrieval across both storage layers.

### 3. Autonomous Training Loop
- Use **XGBoost** as the primary model engine.
- Implement hyperparameter optimization using **Optuna**.
- Integrate **Kolmogorov-Smirnov (KS) Test** for detecting distribution drift between training and inference data.

### 4. MLflow Experiment Tracking
- Log all **Optuna hyperparameters** for every trial.
- Log performance metrics: **RMSE, MAE**, and **KS drift scores**.
- Store artifacts: **Trained model files** and **feature importance plots**.
- Track **Data Lineage** by logging the specific dataset versions or timestamp ranges used.

### 5. Observability & Instrumentation
- Emit pipeline metrics to **Prometheus** (e.g., training duration, scraper success rates).
- Implement structured logging with **Structlog** for machine-readable logs compatible with Loki.

## Acceptance Criteria
- [ ] Scrapers successfully fetch and parse required market data.
- [ ] Data is correctly partitioned between TimescaleDB and MinIO.
- [ ] Optuna study completes successfully and selects the best hyperparameters.
- [ ] KS Test correctly identifies significant drift in synthetic test data.
- [ ] MLflow dashboard displays all logged parameters, metrics, and plots for each run.
- [ ] Prometheus metrics are visible and structured logs are emitted.

## Out of Scope
- Frontend visualization of metrics (to be handled in a separate track).
- Integration with live trading execution engines.
- Advanced model architectures beyond XGBoost (e.g., Deep Learning models).
