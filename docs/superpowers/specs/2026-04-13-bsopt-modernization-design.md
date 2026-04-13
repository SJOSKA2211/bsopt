# BSOpt Modernization & Sequential Orchestration Design

## Goal
Transform the BSOpt repository into a lean, production-grade, CPU-optimized architecture with 100% true-integration test coverage and a robust sequential deployment pipeline.

## 1. Workspace Purge & File Structure
A "Deep Purge" will be executed to remove all non-essential assets, ensuring a minimalist and logical structure.

### Purge List:
- **Temporary Files**: `src/frontend/eslint-errors.json`, `src/frontend/eslint.json`, `src/frontend/debug-screenshot.png`.
- **Build Artifacts**: `bsopt.egg-info`, `dist`, `.bkit`, `.venv` (root level).
- **Redundant Scripts**: All `scripts/*.sh` except `scripts/bootstrap.sh`. The bootstrap script will absorb ALL environment setup logic.
- **Fluff**: All console emojis, ASCII art banners, and redundant inline comments across the codebase.

### Target Structure:
- `api/`: FastAPI entry points.
- `src/`: Core logic (math, quant, workers, frontend, ml).
- `infrastructure/`: Nginx, Docker, and core orchestration.
- `scripts/`: Only `bootstrap.sh`.

## 2. Full-Stack Modernization
### Python 3.12.13 Codebase
- Enforce `uv` for dependency management.
- Remove all remaining GPU/CUDA references.
- Audit all shims/polyfills and replace with native Python 3.12 features (e.g., improved typing, `asyncio` optimizations).

### Modern Frontend
- Nuke all mock service workers (`msw`) and `mock-socket`.
- Standardize on React 19 hooks for state and data fetching (React Query).
- Ensure 100% compatibility with the live backend container network.

## 3. Infrastructure & Sequential Orchestration
### Dynamic Environment
- `bootstrap.sh` will dynamically generate a `.env` file using `openssl rand -base64` for secure default passwords.
- Existing PKI certs in `.pki/` will be strictly enforced in the gRPC network.

### Sequential Build & Deploy Loop
- A custom loop in `bootstrap.sh` will:
  1. Build a container.
  2. Start the container.
  3. Wait for `health_status == "healthy"`.
  4. Perform the next step only upon success.
- `docker-compose.yml` will use `depends_on: { service_healthy }` for all stack dependencies.

## 4. Zero-Mock Integration Testing
- The testing framework will be rewritten for **True Integration**.
- **No Mocks**: No `unittest.mock`. All tests will target the live containerized services.
- **Playwright**: Headless E2E tests for the frontend.
- **Coverage**: Iterative execution until 100% coverage and stability.

## User Review Required
> [!IMPORTANT]
> The purge is AGGRESSIVE. I will be deleting most existing scripts in `scripts/` and consolidating them into a modern `bootstrap.sh`. If you have custom local scripts you need kept, please label them now.

---
Approved? (Respond with "Approve" or specific feedback)
