# Design: MLOps Health & Revamp

## 1. System Architecture
The system relies on a distributed MLOps stack:
- **MLflow**: Tracking server for experiments and models.
- **Ray**: Distributed compute for training and serving.
- **Inference Service**: Real-time prediction paths.
- **Worker**: Asynchronous task processing.

## 2. Infrastructure Design
- **Backplane**: Postgres (metadata), Redis (cache/queue), RabbitMQ (messaging), MinIO (artifact store).
- **Service Mesh**: Envoy for routing and security.
- **Orchestration**: Docker Compose with profiles (`core`, `ml`).

## 3. Implementation Strategy

### Phase 1: Health Diagnostics
- **Bootstrapping**: PKI certificate generation and service health checks.
- **Diagnostics**: `bin/ml-health` utility to aggregate health status from MLflow and Ray.

### Phase 2: Codebase Revamp
- **ML Directory Refactor**: Transition from placeholder scripts to structured modules in `src/ml`.
- **Inference Path**: Implement standard inference interfaces for real models.
- **Cleanup**: Purge Kafka and Fastify dependencies to align with the Vercel-native goal.

## 4. Security & Configuration
- **Environment Variables**: Use `.env` for all secrets and configuration.
- **Authentication**: JWT-based auth for services.

## 5. Verification Plan
- **Automated Tests**: Pytest for ML modules and integration tests.
- **Manual Verification**: Access dashboards and run end-to-end inference tests.
