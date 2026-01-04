# Tech Stack: BS-Opt

## Core Languages & Runtimes
*   **Python (>=3.11):** Primary programming language for API, pricing engine, and ML pipelines.

## Machine Learning & Data Science
*   **Optuna:** Hyperparameter optimization.
*   **XGBoost & PyTorch:** Primary ML frameworks for model training.
*   **Scikit-learn:** General-purpose ML utilities and metrics.
*   **MLflow:** Model tracking, versioning, and registry.
*   **Pandas & Numpy:** Data manipulation and numerical computation.

## Data Storage & Message Broking
*   **TimescaleDB (PostgreSQL 15):** Time-series database for market data and application state.
*   **Redis (7):** Caching and Celery result backend.
*   **MinIO:** S3-compatible object storage for model artifacts and large datasets.
*   **RabbitMQ:** Primary message broker for the asynchronous task queue.

## Application Frameworks
*   **FastAPI:** High-performance web framework for the API layer.
*   **Celery:** Distributed task queue for long-running compute jobs (Pricing, AutoML).

## Observability & Monitoring (LGTM Stack)
*   **Prometheus:** Metrics collection and alerting.
*   **Loki:** Log aggregation.
*   **Promtail:** Log shipping from Docker containers.
*   **Grafana:** Unified visualization and dashboarding.
*   **Structlog:** Structured JSON logging for machine readability.

## Infrastructure & DevOps
*   **Docker & Docker Compose:** Containerization and local orchestration.
*   **GitHub Actions:** CI/CD/CT pipelines.