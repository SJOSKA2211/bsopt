# Plan: Infrastructure Overhaul

## Steps
1.  **Refactor Docker Compose**:
    -   Define a single `bsopt-net` network.
    -   Define `pgdata` volume.
    -   Add healthchecks for `db` and `redis`.
    -   Consolidate `api` and `worker` environments.
    -   Ensure `auth-service` builds correctly.

2.  **Create Makefile**:
    -   `up`: Launch services.
    -   `down`: Stop services.
    -   `build`: Rebuild images.
    -   `test`: Run tests inside the `test-runner` container.
    -   `clean`: Remove volumes and pycache.

3.  **Update .dockerignore**:
    -   Exclude `venv`, `__pycache__`, `.git`.
    -   Explicitly exclude `node_modules`.

## Validation
-   Running `make up` should succeed.
-   Running `make build` should succeed.
-   Running `make clean` should remove `pgdata`.
