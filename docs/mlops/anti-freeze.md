# Anti-Freeze Guide: Solving System Freezing during Builds

This guide provides solutions for developers experiencing system freezes or extreme slowdowns when building the BS-Opt platform locally. These issues are typically caused by high CPU and RAM consumption during compilation or container builds.

## 1. Limit Build Concurrency

The most effective way to prevent system freezes is to limit the number of parallel build jobs.

### Python Environment
When installing heavy dependencies like `torch`, `numpy`, or `scikit-learn`, limit the number of worker processes:
```bash
# Example for some package managers
export MAKEFLAGS="-j1"
pip install -r requirements.txt
```

### Docker Builds
Limit Docker's resource usage during builds:
```bash
docker build --cpus=2 --memory=4g -t bsopt-api .
```

## 2. CI/CD Pipeline Performance

The BS-Opt platform uses a dedicated `mlops-training.yml` workflow. If this workflow is causing issues during local simulation:
- Ensure `ENVIRONMENT` is set to `dev`.
- Reduce the `n_samples` parameter in `MLOrchestrator` to speed up local testing.

## 3. Dedicated Resource Limits

If you are using a virtual environment, ensure your system has at least 8GB of free RAM before starting a full build.

---
*Maintained by the DevOps Team.*
