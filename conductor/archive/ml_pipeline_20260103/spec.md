# Track Spec: Build the Autonomous ML Pipeline

## Overview
This track focuses on establishing the core machine learning lifecycle for the BS-Opt platform. It includes data ingestion from external financial APIs, a resilient data pipeline, an autonomous model training and optimization loop using Optuna and XGBoost, and comprehensive experiment tracking with MLflow. All components will be instrumented with Prometheus metrics and structured logging for the LGTM observability stack.

## Goals
- Implement resilient market data scrapers (Polygon/Yahoo).
- Build a data pipeline with drift detection (PSI).
- Develop an autonomous model optimization loop using Optuna.
- Integrate MLflow for experiment and model version tracking.
- Instrumentation with Prometheus metrics and structured logging (Structlog).

## Functional Requirements
- **Scrapers:** Able to fetch historical and real-time option data.
- **Data Pipeline:** Store market data in TimescaleDB and check for PSI drift.
- **Training Loop:** Automatically tune hyperparameters using Optuna.
- **MLflow Integration:** Log parameters, metrics, and artifacts for every run.
- **Observability:** Emit metrics for training duration, RMSE, and drift scores.

## Technical Constraints
- **Language:** Python 3.11+
- **Database:** TimescaleDB (PostgreSQL 15)
- **ML Libraries:** XGBoost, Optuna, Scikit-learn, Pandas, NumPy
- **Tracking:** MLflow
- **Observability:** Prometheus Client, Structlog
- **Environment:** Docker-ready

## Success Criteria
- Successful data ingestion into TimescaleDB.
- Autonomous training run completes and logs to MLflow.
- Metrics are visible in Prometheus/Grafana (placeholders/mocks where stack isn't up).
- Code coverage >80% for all new modules.
