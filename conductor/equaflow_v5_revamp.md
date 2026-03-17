# EquaFlow v5: Institutional-Grade Revamp & Automation Plan

## Objective
Architect, refactor, and fully automate the EquaFlow platform following strict institutional-grade mandates: zero local environments, container engine agnosticism, and autonomous self-healing.

## Key Files & Context
- **Orchestration:** `Makefile`, `bootstrap.sh`, `infra/orchestration/docker-compose.yml`
- **Kernels:** `services/quant/rust-core/`, `services/quant/gpu_math.py`
- **Auth:** `services/auth/` (Argon2id, Asymmetric JWT, RBAC)
- **Database:** `init-scripts/` (TimescaleDB tuning, Hierarchical CAGGs)
- **MLOps:** `scripts/mlflow_ray_watchdog.py`, Ray Serve 2.x
- **Testing:** `make test-all`, `scripts/self_healing_test_runner.py`

---

## Phased Implementation Plan

### Phase 0: Zero-Touch Live Database Bootstrapping & Engine Detection
- **Container Agnosticism:** Refine `Makefile` and `bootstrap.sh` to ensure seamless `docker` vs `podman` aliasing.
- **Secrets & Security:** 
    - Update `bootstrap.sh` to generate RSA 4096 and ECC P-256 key pairs using OpenSSL.
    - Generate Argon2id salts and TOTP master secrets.
    - Inject these into `.env` as Base64 encoded strings.
- **Live Spin-Up:** 
    - Ensure `bootstrap.sh` correctly parses `.env` to start `postgres`, `redis`, and `pgbouncer`.
    - Implement a strict `pg_isready` polling loop to halt execution until the DB is 100% responsive.
- **API Gateway:** 
    - Replace/Wrap Node.js `app-gateway` with **Envoy** for SSL termination, gRPC-Web support, and traffic routing.
    - Configure Envoy for mutual TLS (mTLS) in internal service mesh.

### Phase 1: Rust Integration, Math Kernels & GPU Acceleration
- **Rust Core Refactor:**
    - Use `ndarray` and `rayon` in `rust-core` for true vectorization of Black-Scholes and RK4.
    - Implement `mmap_parse_ticks` using the `memmap2` crate for zero-copy parsing.
    - Use `rust-numpy` to expose memory-mapped regions directly to Python as NumPy arrays.
- **GPU Acceleration:**
    - Refine `gpu_math.py` to ensure CuPy handles vectorized Black-Scholes and Greeks calculations.
    - Implement fallback to Rust-core if GPU is unavailable.
- **Math Kernels:**
    - Implement 4th-order Runge-Kutta for Geometric Brownian Motion in `rust-core`.

### Phase 2: Zero-Trust Auth Pipeline, gRPC & Validation
- **Auth Service:**
    - Refactor `services/auth` to use `argon2-cffi` (Python) or the `argon2` crate (Rust/Node) for password hashing.
    - Implement Asymmetric JWT validation (RS256/ES256) using the generated Public Keys.
    - Implement Redis-backed token blocklisting (refresh token rotation).
- **gRPC Integration:**
    - Migrate all internal microservice communication (Auth -> Data -> ML) to gRPC.
    - Define Protobufs in `core/protos/`.
- **Validation:**
    - Enforce **Pydantic V2** across all Python services for hyper-fast validation.

### Phase 3: Hyper-Optimized Live State Architecture
- **TimescaleDB Tuning:**
    - Update `15-runtime-tuning.sql` for NVMe optimization: `random_page_cost = 1.1`, `effective_io_concurrency = 200+`, `max_parallel_workers_per_gather = 4`.
- **Advanced Partitioning:**
    - Establish daily hypertable chunking for high-frequency tick data.
- **Continuous Aggregates:**
    - Implement Hierarchical CAGGs (1m -> 1h -> 1d) in `07-continuous-aggregates.sql`.
    - Enable compression for historical CAGG data.
- **Decoupling:**
    - Ensure RabbitMQ/Kafka decouples scrapers from DB writes with Dead Letter Queues (DLQs).

### Phase 4: MLOps, Distributed Training & Auto-Recovery
- **Distributed Training:**
    - Use Ray Serve 2.x with Deployment Graphs for the inference pipeline.
- **Quantization:**
    - Add an export step to quantize models into **ONNX** format.
- **Auto-Rollback:**
    - Implement a backtesting script that compares new model performance vs baseline and auto-rolls back via MLflow tags.
- **Watchdog:**
    - Refine `mlflow_ray_watchdog.py` to handle Ray OOMs and auto-respawn with adjusted `num_cpus`/`num_gpus`.

### Phase 5: Self-Healing Tests, Cargo & Observability
- **The Gauntlet (`make test-all`):**
    - Ensure `cargo fmt`, `cargo clippy`, `cargo test` and `pytest` are all integrated.
- **Self-Healing Runner:**
    - Create `scripts/self_healing_test_runner.py` to parse failures and suggest/apply code fixes.
- **Observability:**
    - Integrate OpenTelemetry (OTel) for tracing.
    - Configure Prometheus/Grafana to monitor Ray and TimescaleDB metrics.

### Phase 6: DevSecOps, IaC, & Chaos Engineering
- **Infrastructure as Code:**
    - Update Terraform manifests for Blue-Green cloud deployment.
- **Security Scanning:**
    - Integrate `trivy` and `bandit` into the `Makefile` and CI pipelines.
- **Chaos Engineering:**
    - Enhance `scripts/chaos_monkey.py` to randomly kill containers and verify state recovery.

---

## Verification & Testing
1. **Bootstrap:** `make bootstrap` must finish with a healthy database and generated `.env`.
2. **Build:** `make build` must succeed across all Rust and Python services.
3. **Up:** `make up` must result in all containers being `healthy`.
4. **Test:** `make test-all` must pass 100%.
5. **Chaos:** Running `chaos_monkey.py` should not cause permanent data loss or service downtime.
