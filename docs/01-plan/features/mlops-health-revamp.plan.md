# Plan: MLOps Health & Revamp

## Objective
Run the MLOps engine until it is healthy, report its status, and then revamp the ML components as per the system revamp plan.

## Background
The system requires a healthy MLOps infrastructure (MLflow, Ray, etc.) for its core functionalities. A revamp is also planned to remove redundant scripts and implement real inference paths.

## Scope
- Infrastructure: Postgres, Redis, RabbitMQ, MinIO, MLflow, Ray.
- Services: API, Auth, ML Inference, Worker.
- Refactoring: `src/ml` directory, removal of placeholders.

## Implementation Steps

### Phase 1: MLOps Health (Plan: docs/01-plan/run-mlops-until-healthy.md)
1. **Infrastructure Bootstrap**: Run `./bootstrap.sh` to ensure core services are healthy.
2. **MLOps Startup**: Run `./scripts/start_mlops.sh` to launch ML profile services.
3. **Health Reporting**: Run `bin/ml-health` and verify "HEALTHY" status.

### Phase 2: MLOps Revamp (Plan: docs/01-plan/revamp.md)
1. **Cleanup src/ml**: Remove redundant training/serving scripts.
2. **Remove Placeholders**: Delete `mock_model.zip` and fake prediction logic.
3. **Real Inference Paths**: Implement real inference paths in `src/ml`.
4. **Makefile Integration**: Ensure ML tasks are integrated into the unified `Makefile`.

## Verification Criteria
- `bin/ml-health` reports "HEALTHY".
- `make test` passes for ML components.
- Redundant scripts and placeholders are removed from `src/ml`.
- MLflow and Ray dashboards are accessible.
