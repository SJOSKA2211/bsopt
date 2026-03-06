# Codebase Analysis Report

## 1. Overview

This document provides a comprehensive analysis of the `bsopt` codebase. The goal is to provide a clear understanding of the project's architecture, dependencies, and conventions to facilitate a major feature revamp.

## 2. Architecture

The `bsopt` platform is a sophisticated, low-latency financial trading system built on a microservices architecture. The system is designed for high performance and real-time data processing.

Key architectural patterns identified from documentation and configuration files include:

*   **Microservices**: The application is composed of multiple services, each with a specific responsibility (e.g., `api`, `auth-service`, `portfolio`, `worker`, `scraper`, `neural-pricing`). These services are defined in `docker-compose.yml` and `docker-compose.dev.yml`.
*   **Shared Memory IPC ("Market Mesh")**: A shared memory manager (`src/shared/shm_manager.py`) is used for inter-process communication, enabling extremely low-latency data exchange between services like the `scraper` and the `api`.
*   **Hybrid Worker System (Revamped)**: The system uses a combination of Celery and Ray for task distribution.
    *   **Celery** (`src/ml/celery_app.py`, `src/workers/math_worker.py`) is used for non-blocking orchestration using the `BaseAsyncTask` pattern.
    *   **Ray Pool** (`src/workers/ray_workers.py`) is used for computationally intensive tasks, using a robust `RayActorPool` with thread-safe round-robin delegation.
*   **JIT Compilation & Silicon-Risk**: Numba (`@njit`) is used in performance-critical sections. The risk layer now features an **Incremental Delta Tracker** in the `OrderEngine` hot loop, achieving O(1) risk enforcement with < 350ns latency.
*   **Optimized Persistence (Postgres 16)**: The primary database is PostgreSQL 16 with TimescaleDB 2.17+. It is fine-tuned for high-write financial workloads with optimized memory parameters (`work_mem`, `shared_buffers`) and advanced compression policies.
*   **DeFi Options Layer (God-tier)**: The blockchain integration now includes a **High-Frequency Hybrid Oracle** (WebSockets + RPC) and a **Gas-Aware Smart Order Router (SOR)** for optimal execution across decentralized venues.
*   **Quantum Pricing (QAE-v2)**: A cutting-edge quantum engine using **Iterative Amplitude Estimation (IAE)** provides a quadratic speedup over classical methods for derivative pricing.
*   **Neural Engine (DT-v2)**: Advanced **Decision Transformer** implementation with **Flash Attention**, spectral feature engineering (Fourier kernels), and auxiliary loss stabilization for high-fidelity offline RL.
*   **GraphQL Federation**: The `app-gateway` service uses GraphQL Federation to combine multiple GraphQL APIs (`api`, `portfolio`, `neural-pricing`) into a single, unified data graph for the frontend.
*   **Event-Driven Architecture**: RabbitMQ is used as a message broker for asynchronous communication between services, and Kafka is available as a specialized streaming service.

## 3. Core Components & Services

The `docker-compose.yml` file defines the following key services:

| Service | Description | Key Technologies |
| :--- | :--- | :--- |
| **`postgres`** | Primary database for the application. | PostgreSQL, TimescaleDB |
| **`pgbouncer`** | Connection pooler for PostgreSQL. | PgBouncer |
| **`redis`** | In-memory data store for caching and session storage. | Redis |
| **`rabbitmq`** | Message broker for asynchronous tasks. | RabbitMQ |
| **`auth-service`** | Handles user authentication and authorization. | Node.js, Better-Auth |
| **`api`** | Main API service, providing GraphQL endpoints. | FastAPI, Uvicorn, GraphQL |
| **`portfolio`** | Service for managing user portfolios. | FastAPI, Uvicorn |
| **`worker`** | Celery worker for processing background tasks. | Celery |
| **`scraper`** | Service for scraping financial data. | (Custom Python) |
| **`neural-pricing`** | Service for ML-based derivative pricing. | FastAPI, Uvicorn, PyTorch |
| **`app-gateway`** | GraphQL Federation gateway. | Node.js, Apollo Federation |
| **`frontend`** | The main web user interface. | (Likely a JS framework like React/Vue) |

The project also includes specialized services for `blockchain`, `streaming` (Kafka), `ml` (Ray, MLflow), and `hft` (High-Frequency Trading).

## 4. Dependencies

The `pyproject.toml` file lists the project's dependencies. Key libraries and frameworks include:

*   **Core**: `numpy`, `pandas`, `scipy` for numerical and data manipulation.
*   **Performance**: `numba` for JIT compilation.
*   **Web Framework**: `fastapi`, `uvicorn`, `gunicorn` for the Python-based API services.
*   **Database**: `sqlalchemy`, `asyncpg`, `psycopg2-binary`, `pgvector` for interacting with PostgreSQL.
*   **ML**: `torch`, `lightning`, `pytorch-forecasting`, `xgboost`, `scikit-learn`, `mlflow`, `optuna`, `stable-baselines3` for machine learning tasks.
*   **Distributed Computing**: `ray`, `celery`, `confluent-kafka` for distributed and parallel processing.
*   **GraphQL**: `strawberry-graphql` for the Python GraphQL APIs.
*   **Authentication**: `bcrypt`, `cryptography`, `PyJWT`, `pyotp` for security and authentication.
*   **Development & Tooling**: `pytest`, `ruff`, `mypy`, `bandit`, `black` for testing, linting, and formatting.

## 5. Source Code Structure (`src`)

The `src` directory is well-organized by feature/domain:

*   **`src/api`**: Contains the main FastAPI application, including routes, schemas, and GraphQL resolvers.
*   **`src/auth-service`**: A Node.js service for authentication.
*   **`src/database`**: Contains database-related code (not present in the `ls` output, but expected).
*   **`src/ml`**: A large module containing code for machine learning, including architectures, training pipelines, and serving.
*   **`src/pricing`**: Contains the logic for financial instrument pricing, including Black-Scholes, Monte Carlo, and ML-based models.
*   **`src/shared`**: Contains shared utilities and modules, including the `shm_manager.py` for the Market Mesh.
*   **`src/workers`**: Contains the Celery and Ray workers.
*   **`src/trading`**: Contains the core trading logic, including the order engine and risk kernels.
*   **`src/scrapers`**: Contains the data scraping engine.
*   **`src/frontend`**: Contains the frontend application source code.

## 6. Next Steps for Feature Revamp

Based on this analysis, here are the proposed next steps for a feature revamp:

1.  **Define the Scope**: Clearly define the feature to be revamped. Identify which services and components will be affected.
2.  **Architectural Review**: For the affected components, perform a deeper architectural review. This may involve creating sequence diagrams or data flow diagrams to understand the existing implementation.
3.  **Local Development Environment**: Set up a local development environment using `docker-compose.dev.yml` to run the application and test changes.
4.  **Create a Detailed Plan**: Create a detailed implementation plan for the revamp. This plan should break down the work into smaller, manageable tasks.
5.  **Implementation**: Begin implementing the changes, following the project's existing conventions and best practices.

This codebase is complex and highly optimized. A successful feature revamp will require careful planning and a deep understanding of the existing architecture.
