# Plan: Cloud-Native Build Pipeline & Docker Optimization

## Phase 1: Immediate Mitigation & Documentation [checkpoint: fbf4149]
- [x] Task: Document "Anti-Freeze" solutions in `docs/mlops/anti-freeze.md`. Include `COMPOSE_PARALLEL_LIMIT`, Docker resource limits, and `remote-builder` setup. [5375d2e]
- [x] Task: Update main `README.md` to point to the new Anti-Freeze guide. [5ce7ae0]
- [x] Task: Conductor - User Manual Verification 'Phase 1: Immediate Mitigation & Documentation' (Protocol in workflow.md) [fbf4149]

## Phase 2: Dockerfile Optimization (Layer Caching) [checkpoint: 8634145]
- [x] Task: Refactor `docker/Dockerfile.ml` to implement multi-stage builds and separate dependency installation (pip install) from source code copy. [a06e013]
- [x] Task: Refactor `docker/Dockerfile.api`, `docker/Dockerfile.scraper`, and `docker/Dockerfile.gateway` with similar caching optimizations. [a06e013]
- [x] Task: Refactor `docker/Dockerfile.mlflow`, `docker/Dockerfile.ray`, and `docker/Dockerfile.wasm` for consistency and caching. [a06e013]
- [x] Task: Verify local build performance improvements and layer reuse using `docker build --progress=plain`. [a06e013]
- [x] Task: Conductor - User Manual Verification 'Phase 2: Dockerfile Optimization (Layer Caching)' (Protocol in workflow.md) [8634145]

## Phase 3: CI/CD Pipeline Implementation (GHA + GHCR) [checkpoint: 9dcf7c1]
- [x] Task: Create `.github/workflows/build-and-push.yml` with matrix strategy for all services (`api`, `ml`, `scraper`, `gateway`, `mlflow`, `ray`, `wasm`). [793acfb]
- [x] Task: Configure GitHub Actions to authenticate with GHCR and use short Commit SHA for image tagging. [793acfb]
- [x] Task: Enable Docker Buildx and GHA cache backend (`type=gha`) in the workflow to speed up CI builds. [793acfb]
- [x] Task: Conductor - User Manual Verification 'Phase 3: CI/CD Pipeline Implementation (GHA + GHCR)' (Protocol in workflow.md) [9dcf7c1]

## Phase 4: Docker Compose & Local Workflow Integration
- [x] Task: Refactor `docker-compose.yml` and `docker-compose.prod.yml` to use YAML anchors (`x-images`) for GHCR image paths. [25f474c]
- [x] Task: Implement `pull_policy: always` and `cache_from` configurations in Compose files to prefer cloud artifacts. [25f474c]
- [x] Task: Add a shell script `scripts/dev-setup.sh` to automate the `docker-compose pull` and environment initialization. [25f474c]
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Docker Compose & Local Workflow Integration' (Protocol in workflow.md)

## Phase 5: Final Verification & Benchmarking
- [ ] Task: Perform a "Clean Start" test: Remove all local images and run the new pull-based workflow.
- [ ] Task: Benchmark the time from "Push to Git" to "Local Up and Running" for a minor code change.
- [ ] Task: Verify that all services correctly report their versions/SHAs in logs or health checks.
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Final Verification & Benchmarking' (Protocol in workflow.md)
