1. **Fix Trivy Scanner Build Path**
   - The CI failed in `.github/workflows/blue-green-deploy.yml` due to `lstat docker: no such file or directory`.
   - Update `docker build -t bsopt-api-scan:latest -f docker/Dockerfile.api .` to `docker build --target builder -t manifold-base:builder -f infrastructure/orchestration/Dockerfile.base . && docker build -t manifold-base:latest -f infrastructure/orchestration/Dockerfile.base . && docker build -t bsopt-api-scan:latest -f infrastructure/orchestration/Dockerfile.api .` per memory instructions.
2. **Fix Missing `grpcio-health-checks` in Backend Unit Tests**
   - The CI failed in `.github/workflows/equaflow-institutional.yml` (Backend Tests) due to `Could not find a version that satisfies the requirement grpcio-health-checks>=1.0.0; extra == "dev" (from bsopt[api,dev,ml])`.
   - Wait, `grpcio-health-checks` was required by `bsopt[dev]` in `pyproject.toml`. Let's remove it from `pyproject.toml`'s dependencies, or add `grpcio-health-checks` via a direct installation if we can't find it. Actually, `grpcio-health-checks` is available as `grpcio-health-checking` on PyPI. `grpcio-health-checks` doesn't exist. Let's fix the typo in `pyproject.toml`!
3. **Fix Frontend Unit Tests Node Version Error**
   - The CI failed in `.github/workflows/equaflow-institutional.yml` (Frontend Tests) due to `EBADENGINE Unsupported engine { package: 'camera-controls@3.1.2', required: { node: '>=22.0.0', npm: '>=10.5.1' }, current: { node: 'v20.20.2', npm: '10.8.2' } }`.
   - Change `NODE_VERSION: "20"` to `NODE_VERSION: "22"` in both `.github/workflows/equaflow-institutional.yml` and `.github/workflows/blue-green-deploy.yml` (if it exists there).
4. **Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.**
   - Run tests and verifications.
5. **Submit PR**
