# MLOps V2: The Financial Manifold

## Overview
BS-OPT MLOps V2 is a hyper-optimized, containerized, and automated machine learning lifecycle designed for zero-latency financial applications. It leverages **MLflow** for orchestration, **Ray** for distributed compute, and **Docker** for environment parity.

## 🏛️ Architecture

### 1. Standardization: MLflow Projects
The root `MLproject` file defines all entry points. This ensures that any model can be retrained with a single command, regardless of the underlying framework.

**Entry Points:**
- `train_regressor`: Distributed HPO (XGBoost/NN) via Ray Tune.
- `train_rl`: Offline/Online RL training (TD3/DT-v2).
- `detect_drift`: Automated drift analysis for AIOps triggers.

### 2. Compute: Ray & Docker
- **Base Image**: `rayproject/ray:latest-py312-cpu` (optimized for SHM and inter-node communication).
- **Package Manager**: `uv` is used for ultra-fast dependency resolution during builds.
- **Orchestration**: `docker-compose.yml` manages the `mlflow` tracking server, `ray-head`, and `mlops-worker`.

### 3. AIOps: Closed-Loop Retraining
The `AIOpsOrchestrator` implements a "Self-Healing ML" loop:
1. **Monitor**: Prometheus metrics track error rates and latency.
2. **Detect**: `DriftTrigger` and `MultivariateDriftDetector` identify data/concept drift.
3. **Act**: `AIOpsOrchestrator` triggers `mlflow.projects.run` to retrain models.
4. **Promote**: `MLPipeline` automatically registers and promotes the "champion" model to `Production`.

## ⚡ Performance Optimizations

### Vectorized Kernels (Numba)
All financial indicators and RL environment steps are implemented as **Fused Numba Kernels** (`src/ml/indicators.py`, `src/ml/reinforcement_learning/kernels.py`).
- **Zero Allocation**: Steps are executed in machine code with minimal memory overhead.
- **Spectral Features**: Market micro-structure is captured using multi-scale Fourier kernels.

### RL Policy Synchronization
`SHMWeightSyncCallback` uses **Linux Shared Memory** to broadcast policy weights from the training worker to the execution engine in near real-time.

## 🚀 Operational Guide

### Build the Runtime
```bash
docker build -f docker/Dockerfile.mlops -t bsopt-mlops-runtime:latest .
```

### Start the Stack
```bash
docker compose --profile ml up -d
```

### Execute a Training Job
```bash
# Example: Regressor HPO
mlflow run . -e train_regressor -P ticker=TSLA -P n_trials=20

# Example: RL Training
mlflow run . -e train_rl -P timesteps=10000
```

### View Results
- **Tracking Server**: `http://localhost:5000`
- **Ray Dashboard**: `http://localhost:8265`

---
*Maintained by the AI Engineering Team.*
