# Design: Comprehensive System Revamp

## 1. Objective
Stabilize the BS-OPT system by removing redundant legacy components (Fastify, Kafka) and optimizing for a Vercel-native, lightweight architecture.

## 2. Component Revamp

### 2.1. API Layer
- **Current State**: FastAPI implementation in `api/index.py`.
- **Target**: Ensure all routes in `api/routes/` are stateless and compatible with Vercel serverless environment.
- **Action**: Purge any remaining `fastify` references from lock files and documentation.

### 2.2. Messaging & Workers
- **Action**: Delete `src/workers/streaming/` (if it existed) and any code attempting to import `kafka`.
- **Alternative**: Use Vercel Cron jobs or Redis-based queues for lightweight background tasks.

### 2.3. ML Pipeline
- **Action**: Ensure `src/ml/serving` uses real model paths instead of placeholders.
- **Cleanup**: Remove any dummy models or mock prediction logic found during implementation.

### 2.4. Infrastructure Consolidation
- **Action**: Create a unified `Makefile` that replaces multiple shell scripts in `scripts/`.
- **Tasks to include**: build, test, lint, start-api, health-check.

## 3. Security Hardening
- **Action**: Verify all secrets are pulled from environment variables.
- **Tool**: Use a script to scan for hardcoded credentials.

## 4. Verification Plan
- **Health Check**: Engine health must be verified using `scripts/engine_health.py --simulate` if Docker is unavailable.
- **Tests**: All `pytest` suites must pass.
