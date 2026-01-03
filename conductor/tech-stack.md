# Technology Stack: Black-Scholes Optimization & ML Platform (BS-Opt)

## Core Language & Runtime
- **Python 3.11+:** The primary language for the API, ML pipelines, and quant engines, leveraging its mature ecosystem for data science and finance.

## Application Layer
- **FastAPI:** A modern, high-performance web framework for building the gateway API, utilizing asynchronous capabilities for low-latency request handling.
- **React:** The frontend framework used to build a responsive and data-centric user interface for traders and analysts.

## Quantitative & ML Engine
- **QuantLib:** The industry-standard library for quantitative finance, used for precise Black-Scholes calculations and option greeks.
- **XGBoost & PyTorch:** Core libraries for building and training advanced machine learning models.
- **Optuna:** An open-source hyperparameter optimization framework used for autonomous model tuning.

## Message Processing & State
- **RabbitMQ:** The primary message bus for distributed task processing and event-driven communication between services.
- **Redis Cluster:** Used for high-speed caching and as a secondary broker to manage session state and transient data.

## Persistence & Artifacts
- **TimescaleDB (PostgreSQL 15):** A time-series database optimized for fast ingest and complex queries of market and trading data.
- **MinIO (S3 Compatible):** High-performance object storage for persisting model artifacts, datasets, and logs.
- **MLflow:** A dedicated platform for the machine learning lifecycle, including experimentation, reproducibility, and a central model registry.

## Observability Stack (LGTM)
- **Prometheus:** Collects and stores real-time metrics from the application and infrastructure.
- **Loki & Promtail:** A log aggregation system inspired by Prometheus, used for high-efficiency structured logging.
- **Grafana:** The unified visualization layer for monitoring system health, Greeks drift, and ML metrics.
- **cAdvisor:** Provides container users an understanding of the resource usage and performance characteristics of their running containers.

## Infrastructure & DevOps
- **Docker & Docker Compose:** Containerization and orchestration for local development and production-like environments.
- **GitHub Actions:** Automates the CI/CD/CT (Continuous Training) pipelines, ensuring seamless integration, testing, and deployment.
