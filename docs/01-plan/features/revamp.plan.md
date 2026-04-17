# Comprehensive System Revamp Plan

## 1. Objective
Refactor the BSOPT codebase to make it lightweight, Vercel-compatible, and free of redundant logic, hardcoded secrets, and unused dependencies (Fastify, Kafka). Enhance ML pipelines, infrastructure, and test coverage to ensure a lean and dynamic system.

## 2. Work Streams

### 2.1. Frontend & Vercel Migration
*   **Vercel Preparation**: Ensure the Vite React app is fully prepared for Vercel deployment.
*   **Remove Fastify**: Remove `services/api` Fastify logic. Migrate required backend logic into Vercel Serverless Functions (`api/` directory) or equivalent Next.js API routes if migrating.
*   **Ticker Tape Revamp**: Refactor `TickerStrip` in `src/frontend/src/components/layout/Layout.tsx` into a dedicated `TickerTape` or `PriceTicker` component. Connect it to live dynamic logic (Server-Sent Events / WebSockets via Vercel) instead of placeholder data.

### 2.2. ML, Models & Pipelines
*   **AIOps & MLOps**: Revamp `src/ml` by removing redundant training/serving scripts.
*   **Placeholder Removal**: Remove all `mock_model.zip`, fake prediction functions, and dummy logic. Fully implement real inference paths.

### 2.3. Kafka Removal
*   **Purge Kafka**: Delete all Kafka workers (e.g., `src/workers/streaming/kafka_producer.py`, `kafka_consumer.py`).
*   **Alternative Messaging**: If queuing is still necessary for real-time data, propose a lightweight Vercel-compatible alternative (e.g., Redis Pub/Sub or HTTP-based webhooks).

### 2.4. Infrastructure & Scripts
*   **Makefiles & Scripts**: Port Bash scripts (`scripts/*.sh`, `scripts/*.py`) into a unified `Makefile`. Delete redundant scripts to maintain a clean root.
*   **Protoc Revamp**: Streamline protobuf compilation logic within the Makefile.
*   **Docker Revamp**: Refactor Dockerfiles for minimal footprint (multi-stage builds, alpine/distroless bases). 
*   **Dependencies**: Clean up and optimize `requirements.txt` / `uv.lock` by removing Kafka, Fastify, and other deprecated packages.

### 2.5. Security & Tokens
*   **Dynamic Secrets**: Replace all hardcoded secrets, credentials, and symbols across the codebase with environment variable lookups (`os.getenv`, `process.env`).
*   **Token Generation**: Revamp token generation logic (JWT) to ensure robust and dynamic signature verification.

### 2.6. Testing & Quality Assurance
*   **Unit Tests**: Standardize all Python tests on `pytest`.
*   **Integration Tests**: Add an integration tests layer to verify cross-module functionality without full E2E overhead.
*   **Playwright E2E**: Revamp `tests/e2e` to eliminate mock logic and test real paths where possible.
*   **Cleanup**: Remove redundant comments and dead code across the entire codebase to make it as lean as possible.

## 3. Execution Strategy
Due to the vast scope of these changes, the work will be divided into modular tasks. The first step will focus on stripping out Fastify and Kafka, followed by the frontend/ticker revamp, and then the ML and testing optimizations.

## 4. Verification
*   The application starts cleanly locally (`vercel dev` or `npm run dev`).
*   The Ticker Tape displays dynamic real-time data.
*   `make test` successfully runs the `pytest` and `playwright` suites.
*   Zero occurrences of `kafka` or `fastify` remain in the codebase.