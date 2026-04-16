# Design: MLOps Stack Health & Revamp

## 1. Objective
Ensure the MLOps infrastructure (MLflow, Ray, Workers) is fully operational and healthy. Report its comprehensive health status and perform necessary "revamp" steps to clean up legacy artifacts.

## 2. Approach
- **Phase 1: Stabilization**: Complete the `bootstrap.sh` and `start_mlops.sh` sequence.
- **Phase 2: Health Verification**:
    - **Container Level**: `docker-compose ps` status check.
    - **Service Level (MLflow)**: `curl -f http://localhost:5000/health`.
    - **Service Level (Ray)**: `ray status` and `curl -f http://localhost:8265/healthz`.
    - **Application Level**: `bin/ml-health` check.
- **Phase 3: Revamp**:
    - Port legacy scripts to `Makefile`.
    - Clean up redundant training/serving placeholders.
    - Ensure `.env` is fully secured.

## 3. Architecture Context
- **Tracking**: MLflow (Port 5000)
- **Compute**: Ray Head (Port 8265) & Workers
- **Storage**: MinIO (S3 compatible)
- **Database**: TimescaleDB via PGBouncer

## 4. Verification Plan
- [ ] Core Infra (DB, Redis, MinIO) healthy.
- [ ] MLflow reports `/health` 200 OK.
- [ ] Ray cluster status shows all workers active.
- [ ] `ml-health` script returns "HEALTHY".
