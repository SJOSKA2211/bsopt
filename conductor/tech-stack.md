# Tech Stack: BS-Opt

## Core Languages & Runtimes
*   **Python (>=3.11):** Primary programming language for API, pricing engine, and ML pipelines.
*   **Rust:** Language for high-performance WebAssembly modules.
*   **Node.js (>=18):** Runtime for the Apollo GraphQL Gateway.

## Machine Learning & Data Science
*   **Optuna:** Hyperparameter optimization.
*   **XGBoost & PyTorch:** Primary ML frameworks for model training.
*   **Ray & RLLib:** Distributed training and scaling for reinforcement learning.
*   **Gymnasium:** Standardized environment interface for RL agents.
*   **Stable-Baselines3:** Reliable implementation of TD3 and other RL algorithms.
*   **Flower (flwr):** Framework for federated learning and distributed model aggregation.
*   **IBM Qiskit:** Primary framework for quantum circuit construction and simulation.
*   **Qiskit-Aer:** High-performance local quantum simulators.
*   **Qiskit-Algorithms:** Implementation of Amplitude Estimation and other quantum algorithms.
*   **Scikit-learn:** General-purpose ML utilities and metrics.
*   **MLflow:** Model tracking, versioning, and registry.
*   **Pandas & Numpy:** Data manipulation and numerical computation.

## Data Storage & Message Broking
*   **TimescaleDB (PostgreSQL 15):** Time-series database for market data and application state.
*   **Redis (7):** Caching and Celery result backend.
*   **MinIO:** S3-compatible object storage for model artifacts and large datasets.
*   **RabbitMQ:** Primary message broker for the asynchronous task queue.
*   **Apache Kafka (3.5+):** Real-time event streaming platform for market data.
*   **ksqlDB:** Streaming SQL engine for real-time data filtering and transformation.
*   **Confluent Schema Registry:** Avro schema management and enforcement.
*   **Faust-streaming:** Python library for building streaming applications.

## Application Frameworks
*   **FastAPI:** High-performance web framework for the API layer.
*   **React (>=19) & Vite:** Modern frontend framework and build tool for the platform UI.
*   **Celery:** Distributed task queue for long-running compute jobs (Pricing, AutoML).

## Observability & Monitoring (LGTM Stack)
*   **Prometheus:** Metrics collection and alerting.
*   **Loki:** Log aggregation.
*   **Promtail:** Log shipping from Docker containers.
*   **Tempo:** Distributed tracing for microservices.
*   **Grafana:** Unified visualization and dashboarding.
*   **Structlog:** Structured JSON logging for machine readability.

## Infrastructure & DevOps
*   **Docker & Docker Compose:** Containerization and local orchestration.
*   **Docker SDK (Python):** Programmatic container management for self-healing workflows.
*   **GitHub Actions:** CI/CD/CT pipelines.
*   **Wasm-pack:** Toolchain for building and packaging Rust-generated WebAssembly.
*   **Trivy:** Container and filesystem security scanning.
*   **Locust:** High-performance load testing and benchmarking.
*   **Ruff:** High-performance Python linting and formatting.
*   **Uvloop:** High-performance event loop for the FastAPI layer.