# MLOps Health & Revamp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the MLOps stack, verify its health, and revamp legacy artifacts into a unified structure.

**Architecture:** A three-phase approach starting with environment stabilization, followed by deep health verification across tracking, compute, and application layers, and concluding with a systemic revamp of scripts and placeholders.

**Tech Stack:** Docker Compose, MLflow, Ray, TimescaleDB, Python, Makefile.

---

### Task 1: Stabilize Core Infrastructure

**Files:**
- Modify: `infrastructure/orchestration/docker-compose.yml`
- Execute: `./bootstrap.sh`

- [ ] **Step 1: Verify PostgreSQL and PGBouncer**
Run: `docker ps --filter "name=postgres" --filter "name=pgbouncer" --format "{{.Names}}: {{.Status}}"`
Expected: Both containers show "Up" and "healthy".

- [ ] **Step 2: Initialize Performance Backplane**
Run: `docker-compose -f infrastructure/orchestration/docker-compose.yml up -d redis rabbitmq minio`
Expected: Containers start successfully.

- [ ] **Step 3: Secure Environment**
Run: `bash bootstrap.sh`
Expected: Completion of security layer setup and core service building.

- [ ] **Step 4: Commit Infrastructure State**
```bash
git add .env
git commit -m "infra: stabilize core infrastructure services"
```

### Task 2: Launch and Verify MLOps Stack

**Files:**
- Execute: `scripts/start_mlops.sh`
- Verify: `bin/ml-health`

- [ ] **Step 1: Start MLOps Services**
Run: `bash scripts/start_mlops.sh`
Expected: Services `mlflow`, `ray-head`, `mlops-worker` are reported as "Healthy".

- [ ] **Step 2: Deep Health Check (MLflow)**
Run: `curl -sf http://localhost:5000/health`
Expected: Returns "OK".

- [ ] **Step 3: Deep Health Check (Ray)**
Run: `ray status`
Expected: Shows active nodes and resource allocation.

- [ ] **Step 4: Run Application Health Reporter**
Run: `python bin/ml-health`
Expected: Status reported as "HEALTHY".

- [ ] **Step 5: Commit Health Status**
```bash
git add docs/superpowers/specs/2026-03-31-mlops-health-check-design.md
git commit -m "mlops: verify stack health and connectivity"
```

### Task 3: System Revamp - Script Porting

**Files:**
- Modify: `Makefile`
- Delete: Redundant scripts in `scripts/`

- [ ] **Step 1: Port MLops startup to Makefile**
Add to `Makefile`:
```makefile
mlops-up:
	bash scripts/start_mlops.sh

mlops-status:
	python bin/ml-health
```

- [ ] **Step 2: Verify ported commands**
Run: `make mlops-status`
Expected: Same output as `bin/ml-health`.

- [ ] **Step 3: Clean up redundant scripts**
Run: `rm scripts/start_mlflow_pipeline.sh` (if redundant)
Expected: Cleanup of legacy entry points.

- [ ] **Step 4: Commit Revamp**
```bash
git add Makefile
git commit -m "revamp: port mlops scripts to Makefile"
```
