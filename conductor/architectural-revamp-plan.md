# Implementation Plan: Massive Architectural Revamp & Optimization

## Phase 0: Complete Stack Automation & Security Bootstrapping
- **Bootstrap Script**: Create `scripts/bootstrap.sh` to fully automate the setup.
  - Automatically generate cryptographically secure MFA and JWT secrets using `openssl rand -hex 32`.
  - Inject these private keys and necessary configuration variables into a `.env` file on first run.
- **Environment Separation**: Ensure clear boundaries between Dev and Prod.
  - Audit and refine `docker-compose.dev.yml` (development with hot-reloading, local DBs) and `docker-compose.yml` (production with hardened settings, pre-built images).

## Phase 1: Codebase Revamp, bsopt-api Fix & Math Kernels
- **bsopt-api Fix**:
  - **Root Cause**: In `src/api/routes/pricing.py`, there is a call to `await asyncio.gather(...)` but the `asyncio` module is not imported, causing a `NameError` which breaks the API `/calculate` endpoint.
  - **Fix**: Add `import asyncio` to `src/api/routes/pricing.py`. Audit other endpoints for similar missing imports or exception handling.
- **Asynchronous Refactoring**:
  - Ensure all endpoints, WebSockets (`src/api/websockets`), and data scrapers use `async/await` patterns with `httpx` or `aiohttp` to avoid blocking the `uvloop` event loop.
- **Math Kernels (Black-Scholes)**:
  - Create/Update `src/shared/math_utils.py` to use fully vectorized, strictly typed Numba functions (`@njit(fastmath=True, cache=True)`) for computing $d_1, d_2$, Call/Put Prices, and Greeks. This avoids Python loop overhead and leverages C-level speed.

## Phase 2: Infrastructure & Database (Reuse & Optimize)
- **Docker Optimization**:
  - Reuse existing `docker/Dockerfile.*` (e.g., `Dockerfile.api`, `Dockerfile.worker`) and optimize them using multi-stage builds.
  - Minimize layers, use lightweight base images (e.g., `python:3.12-slim`), and leverage Docker build caching.
- **PostgreSQL / TimescaleDB**:
  - Enhance the existing `init-scripts/` to create a normalized schema optimized for massive time-series data (market data, option chains).
  - Configure connection pooling (e.g., PgBouncer integration) for handling massive concurrent connections.

## Phase 3: High-Volume Data Ingestion & Monitoring
- **Concurrent Scraping**:
  - Update `src/scrapers/` to asynchronously trigger NSE scraping and massive `yfinance` bulk downloads.
  - Integrate `aiolimiter` and `tenacity` for strict rate-limiting, exponential backoff, and robust error handling.
- **Monitoring Stack**:
  - Enhance the Prometheus and Grafana setup (defined in `docker-compose`) to explicitly track ingestion health (records/sec), API rate limits (HTTP 429s), and database load.

## Phase 4: Full-Lifecycle Machine Learning (Pre, Mid, & Post-Training)
- **Pre-Training**:
  - Async data extraction from Postgres.
  - Robust handling of NaNs and missing data.
  - Feature Engineering: Append the dynamically calculated Black-Scholes Greeks as features using the vectorized math kernels.
  - Implement temporal train/test splitting to prevent data leakage.
- **Training**:
  - Scalable distributed training loop (e.g., using Ray or PyTorch Lightning as referenced in existing files).
- **Post-Training**:
  - Integrate `MLflow` for logging model metrics, hyperparameters, and artifact storage.
  - Implement model evaluation against a champion/challenger baseline.

## Phase 5: CI/CD/CT Pipelines
- **CI/CD**:
  - Establish a GitHub Actions workflow (`.github/workflows/main.yml`) for running the test suite (`make test-all`), linting, and building optimized Docker images.
- **Continuous Training (CT)**:
  - Implement a database trigger or Celery beat task that monitors the volume of new market data. When a specific threshold is reached, it will automatically trigger the MLflow pipeline for model retraining.

## Verification
- Run `scripts/bootstrap.sh` and verify `.env` generation.
- Start API using `docker compose` and test `/calculate` endpoint to confirm the `bsopt-api` error is fixed.
- Run benchmark scripts (`scripts/benchmark_db.py`, `scripts/benchmark_risk_kernels.py`) to validate math kernel and DB optimization.
- Check Prometheus/Grafana dashboards for active metrics collection.
