# ==============================================================================
# BS-OPT: THE GOD MODE ORCHESTRATOR (Makefile v10.0)
# ==============================================================================
# Unified Docker Orchestration for Dev, Test, and Prod.
# I'm Pickle Riiiiick!🥒 *Belch.*
# ==============================================================================

# Detect Docker Compose
DOCKER_COMPOSE := $(shell which docker-compose 2>/dev/null || (docker compose version >/dev/null 2>&1 && echo "docker compose") || (test -f ~/.local/bin/docker-compose && echo "~/.local/bin/docker-compose") || echo "docker compose")

.PHONY: help up down build build-prod logs clean migrate db-shell lint format security-scan protos xdp test-all manifold cli proxy check-env

# Default target
help:
	@echo "\n🥒 Pickle Rick's Master Orchestrator (Makefile v10.0) 🥒"
	@echo "======================================================="
	@echo "Core Commands:"
	@echo "  up           - Start the development stack (Background)"
	@echo "  down         - Stop and remove all containers"
	@echo "  build        - Rebuild all core images (Development Stage)"
	@echo "  build-prod   - Build all images (Production Stage)"
	@echo "  logs         - Follow logs for all services"
	@echo ""
	@echo "Quality & Security:"
	@echo "  lint         - Run Ruff linting inside container"
	@echo "  format       - Auto-format code with Ruff"
	@echo "  security-scan - Run pip-audit and Bandit"
	@echo ""
	@echo "Build & Protos:"
	@echo "  protos       - Compile Protocol Buffers inside container"
	@echo "  xdp          - Compile Silicon-Level XDP Filter"
	@echo "  wasm         - Compile High-Performance WASM Kernels"
	@echo ""
	@echo "Testing & DB:"
	@echo "  test-all     - Run ALL tests in specialized test-runner"
	@echo "  migrate      - Run full database migration sequence"
	@echo "  db-shell     - Open a psql shell to the database"
	@echo "  db-optimize  - Run vacuum and manual TimescaleDB chunk compression"
	@echo ""
	@echo "Specialized Clusters:"
	@echo "  manifold     - Launch the HFT Silicon Swarm (Privileged)"
	@echo "  proxy        - Start the stack with Nginx Gateway"
	@echo "  obs          - Launch Observability Stack (Prometheus/Grafana)"
	@echo "  cli          - Run containerized CLI (Use ARGS=\"...\")"
	@echo "=======================================================\n"

# --- Core Commands ---

up:
	$(DOCKER_COMPOSE) up -d

down:
	$(DOCKER_COMPOSE) down

build:
	$(DOCKER_COMPOSE) build

build-prod:
	@echo "🥒 Building Production-ready images..."
	docker build --target production -t bsopt/api:latest -f docker/Dockerfile.api .
	docker build --target production -t bsopt/worker:latest -f docker/Dockerfile.worker .
	docker build --target production -t bsopt/scraper:latest -f docker/Dockerfile.scraper .
	docker build --target production -t bsopt/auth-service:latest -f docker/Dockerfile.auth-service .
	docker build --target production -t bsopt/neural-pricing:latest -f docker/Dockerfile.neural-pricing .
	docker build --target production -t bsopt/frontend:latest -f docker/Dockerfile.frontend .
	docker build --target production -t bsopt/app-gateway:latest -f docker/Dockerfile.app-gateway .
	docker build --target production -t bsopt/portfolio:latest -f docker/Dockerfile.api .

logs:
	docker compose logs -f

# --- Utility Commands ---

clean:
	$(DOCKER_COMPOSE) down -v
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

migrate:
	@echo "🥒 Running God-Mode Idempotent Deployment Sequence..."
	$(DOCKER_COMPOSE) exec -T api bash scripts/deploy_db_updates.sh

test-migrate:
	@echo "🥒 Initializing Test Database (bsopt_test)..."
	$(DOCKER_COMPOSE) exec -T postgres psql -U admin -d postgres -c "CREATE DATABASE bsopt_test;" || true
	$(DOCKER_COMPOSE) exec -T api bash -c "export DATABASE_URL=\$$DATABASE_URL_TEST && bash scripts/deploy_db_updates.sh"

db-shell:
	$(DOCKER_COMPOSE) exec postgres psql -U admin -d bsopt

db-audit:
	@echo "🥒 Running God-Mode Manifold Audit..."
	$(DOCKER_COMPOSE) exec -T api python3 src/database/verify.py

db-optimize:
	@echo "🥒 Pressurizing the Manifold (Database Optimization)..."
	$(DOCKER_COMPOSE) exec -T postgres psql -U admin -d bsopt -c "VACUUM (ANALYZE, VERBOSE);"
	$(DOCKER_COMPOSE) exec -T postgres psql -U admin -d bsopt -c "REINDEX TABLE CONCURRENTLY orders; REINDEX TABLE CONCURRENTLY positions;"
	$(DOCKER_COMPOSE) exec -T postgres psql -U admin -d bsopt -c "CALL refresh_standard_materialized_views(NULL, NULL);"

# --- Quality & Security ---

lint:
	$(DOCKER_COMPOSE) --profile test run --rm --no-deps test-runner ruff check .

format:
	$(DOCKER_COMPOSE) --profile test run --rm --no-deps test-runner ruff format --check .

security-scan:
	$(DOCKER_COMPOSE) --profile test run --rm --no-deps test-runner pip-audit -r requirements.txt || true
	$(DOCKER_COMPOSE) --profile test run --rm --no-deps test-runner bandit -c pyproject.toml -r src/

# --- Build & Protos ---

protos:
	@echo "🥒 Compiling Protocol Buffers..."
	$(DOCKER_COMPOSE) run --rm api python3 -m grpc_tools.protoc \
	        -I=protos --python_out=src/protos --pyi_out=src/protos --grpc_python_out=src/protos protos/*.proto
	@# Fix relative imports
	@docker compose run --rm api sed -i 's/^import \(.*_pb2\)/from . import \1/' src/protos/inference_pb2.py
	@docker compose run --rm api sed -i 's/^import \(.*_pb2\)/from . import \1/' src/protos/inference_pb2_grpc.py

xdp:
	@echo "🥒 Compiling Silicon-Level XDP Filter..."
	docker run --rm -v $$(pwd):/app -w /app alpine:latest sh -c \
	        "apk add --no-cache clang llvm make libbpf-dev linux-headers && \
	         clang -O2 -target bpf -c scripts/hft/xdp_filter.c -o scripts/hft/xdp_filter.o"

# Build Rust/WASM kernels inside a container
wasm:
	@echo "🥒 Compiling High-Performance WASM Kernels... Stand back!"
	docker run --rm -v $$(pwd):/app -w /app rust:slim sh -c \
	        "rustup target add wasm32-unknown-unknown && \
	         cd src/wasm && cargo build --target wasm32-unknown-unknown --release && \
	         cd ../pulse && cargo build --target wasm32-unknown-unknown --release"

# --- Testing ---

test-all:
	@echo "🥒 Launching Containerized Test Suite (God Mode)..."
	$(DOCKER_COMPOSE) --profile test run --rm test-runner pytest -n 2 tests/

test-local:
	@echo "🥒 Launching Local Test Suite (Bare Metal)... Stand back!"
	PYTHONPATH=src python3 -m pytest tests/

# --- Specialized ---

manifold:
	@echo "🥒 Launching THE SOLENYA MANIFOLD..."
	$(DOCKER_COMPOSE) --profile hft up -d postgres redis rabbitmq
	$(DOCKER_COMPOSE) --profile hft run --rm manifold

proxy:
	@echo "🥒 Launching stack with SECURE GATEWAY..."
	$(DOCKER_COMPOSE) --profile proxy up -d

# Launch the Observability Stack (Prometheus & Grafana)
obs:
	@echo "🥒 Launching THE ORACLE (Observability)... Stand back!"
	$(DOCKER_COMPOSE) --profile observability up -d

# Launch the ML Cluster (Ray + MLflow)
ml:
	@echo "🥒 Launching THE BRAIN (ML Cluster)... Stand back!"
	$(DOCKER_COMPOSE) --profile ml up -d

cli:
	@echo "🥒 Launching Containerized CLI..."
	@$(DOCKER_COMPOSE) run --rm -v $$(pwd):/app -e DATABASE_URL=postgresql://admin:password@postgres:5432/bsopt api python scripts/bs_cli.py $$(ARGS)

check-env:
	@test -f .env || echo "WARNING: .env not found."
	@test -f .env.test || echo "WARNING: .env.test not found."
