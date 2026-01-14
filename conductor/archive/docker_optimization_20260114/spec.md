# Specification: Cloud-Native Build Pipeline & Docker Optimization

## Overview
This track aims to resolve system performance issues ("freezing") during the Docker build process by offloading heavy image builds to a cloud-native CI/CD pipeline (GitHub Actions) and optimizing local development workflows. It introduces a "Build in Cloud, Pull to Local" strategy, utilizing GitHub Container Registry (GHCR) and advanced Docker layer caching.

## Goals
- Eliminate local CPU/RAM saturation during build phases.
- Reduce build times through optimized Dockerfile layering and BuildKit caching.
- Standardize the deployment artifact pipeline via GHCR.
- Provide developers with immediate "Anti-Freeze" mitigation strategies.

## Functional Requirements
- **Automated CI Pipeline:** Implement a GitHub Actions workflow to build and push images for all core services (`api`, `ml`, `scraper`, `gateway`, `mlflow`, `ray`, `wasm`).
- **GHCR Integration:** Securely push build artifacts to GitHub Container Registry using short Commit SHAs for tagging.
- **Optimized Dockerfiles:** Refactor Dockerfiles (prioritizing `ml`) to separate system dependencies from application requirements and source code to maximize layer reuse.
- **Enhanced Docker Compose:** Update `docker-compose.yml` to support pulling pre-built images with optional build fallbacks and BuildKit cache integration.
- **Documentation:** Incorporate "Anti-Freeze" solutions (concurrency limits, resource allocation, and remote builder setup) into the project's developer guides.

## Technical Details
- **Registry:** `ghcr.io`
- **Tagging Strategy:** Commit SHA (e.g., `ghcr.io/org/repo-service:sha-1234567`).
- **Build Tools:** Docker Buildx with GHA cache backend.
- **Services Covered:** `api`, `ml`, `scraper`, `gateway`, `mlflow`, `ray`, `wasm`.

## Acceptance Criteria
- [ ] GitHub Actions successfully builds and pushes all service images to GHCR on push to main/develop.
- [ ] Local `docker-compose pull` fetches pre-built images without requiring a local compilation of heavy extensions.
- [ ] Changing application source code does not trigger a re-installation of heavy Python/System dependencies in the Docker build process.
- [ ] Documentation for `COMPOSE_PARALLEL_LIMIT` and `remote-builder` contexts is provided and verified.
