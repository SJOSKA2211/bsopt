# Plan: Docker Containerization for Production Hardening

## Phase 1: Core Dockerfile Hardening & Optimization

- [x] Task: Update `docker/Dockerfile.ml` to multi-stage build with non-root user and shared memory configuration.
  - [x] Sub-task: Write failing test for `Dockerfile.ml` to verify non-root user and stripped image size. (Verification skipped by user instruction)
  - [x] Sub-task: Implement changes in `docker/Dockerfile.ml` as per spec.
  - [x] Sub-task: Verify test passes and image size is reduced. (Verification skipped by user instruction)
- [x] Task: Update `docker/Dockerfile.wasm` to multi-stage build with non-root user and Nginx configuration.
  - [x] Sub-task: Write failing test for `Dockerfile.wasm` to verify non-root user and stripped image size. (Verification skipped by user instruction)
  - [x] Sub-task: Implement changes in `docker/Dockerfile.wasm` and `docker/nginx/wasm.conf` as per spec.
  - [x] Sub-task: Verify test passes and image size is reduced. (Verification skipped by user instruction)
- [x] Task: Review and apply multi-stage and hardening to `docker/Dockerfile.api`.
  - [x] Sub-task: Write failing test for `Dockerfile.api` to verify non-root user and stripped image size. (Verification skipped by user instruction)
  - [x] Sub-task: Implement changes in `docker/Dockerfile.api`.
  - [x] Sub-task: Verify test passes and image size is reduced. (Verification skipped by user instruction)
- [x] Task: Review and apply multi-stage and hardening to `docker/Dockerfile.gateway`.
  - [x] Sub-task: Write failing test for `Dockerfile.gateway` to verify non-root user and stripped image size. (Verification skipped by user instruction)
  - [x] Sub-task: Implement changes in `docker/Dockerfile.gateway`.
  - [x] Sub-task: Verify test passes and image size is reduced. (Verification skipped by user instruction)
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Core Dockerfile Hardening' (Protocol in workflow.md)

## Phase 2: Docker Compose & Infrastructure Optimization

- [ ] Task: Update `docker-compose.prod.yml` with Ray shared memory (shm_size) settings.
  - [ ] Sub-task: Write failing test to confirm Ray shared memory is configured correctly.
  - [ ] Sub-task: Implement `shm_size` in `docker-compose.prod.yml` for Ray services.
  - [ ] Sub-task: Verify test passes.
- [ ] Task: Implement Kafka KRaft mode configuration in `docker-compose.prod.yml`.
  - [ ] Sub-task: Write failing test to verify Kafka cluster starts in KRaft mode and Zookeeper is removed.
  - [ ] Sub-task: Implement Kafka KRaft configuration in `docker-compose.prod.yml`.
  - [ ] Sub-task: Verify test passes.
- [ ] Task: Incorporate Geth and Quantum Simulator configurations into `docker-compose.prod.yml`.
  - [ ] Sub-task: Write failing tests for Geth and Quantum Simulator configurations.
  - [ ] Sub-task: Implement Geth and Quantum Simulator services in `docker-compose.prod.yml`.
  - [ ] Sub-task: Verify tests pass.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Docker Compose Optimization' (Protocol in workflow.md)

## Phase 3: Network Policies & Validation

- [ ] Task: Define and implement Docker network isolation rules.
  - [ ] Sub-task: Write failing test to verify network isolation between selected services.
  - [ ] Sub-task: Implement network policies using Docker features (e.g., custom networks, `--internal` flag).
  - [ ] Sub-task: Verify test passes.
- [ ] Task: Run comprehensive performance benchmarks to validate optimizations.
  - [ ] Sub-task: Define key performance indicators (KPIs) and establish baseline.
  - [ ] Sub-task: Execute benchmarks for startup time, memory usage, and application responsiveness.
  - [ ] Sub-task: Analyze results against acceptance criteria.
- [ ] Task: Execute container security scans (Trivy or similar) and review reports.
  - [ ] Sub-task: Configure Trivy (or similar) scan within CI/CD.
  - [ ] Sub-task: Run scans and ensure no critical/high vulnerabilities related to container setup.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Network Policies & Validation' (Protocol in workflow.md)
