# ==============================================================================
# EQUAFLOW: THE INSTITUTIONAL-GRADE MANIFOLD (Makefile v11.0)
# ==============================================================================
# Unified Orchestration for Rust, Python, gRPC, and Envoy.
# ==============================================================================

# Unified Orchestration: Prioritize V2 plugins (podman compose) over V1 (podman-compose)
PODMAN_V2 := $(shell podman compose version >/dev/null 2>&1 && echo "podman compose")
PODMAN_V1 := $(shell which podman-compose 2>/dev/null)
DOCKER_V2 := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose")
DOCKER_V1 := $(shell which docker-compose 2>/dev/null)

ifneq ($(PODMAN_V2),)
  DOCKER_COMPOSE := $(PODMAN_V2)
  CONTAINER_ENGINE := podman
else ifneq ($(PODMAN_V1),)
  DOCKER_COMPOSE := $(PODMAN_V1)
  CONTAINER_ENGINE := podman
else ifneq ($(DOCKER_V2),)
  DOCKER_COMPOSE := $(DOCKER_V2)
  CONTAINER_ENGINE := docker
else ifneq ($(DOCKER_V1),)
  DOCKER_COMPOSE := $(DOCKER_V1)
  CONTAINER_ENGINE := docker
else
  $(error "No container orchestration found (podman compose, podman-compose, or docker compose)")
endif

# Centralized Orchestration Manifest
ENV_FILE := .env
COMPOSE_FILE := infrastructure/orchestration/docker-compose.yml

ifneq ($(wildcard $(ENV_FILE)),)
  DOCKER_COMPOSE := $(DOCKER_COMPOSE) --env-file $(ENV_FILE) -f $(COMPOSE_FILE)
else
  DOCKER_COMPOSE := $(DOCKER_COMPOSE) -f $(COMPOSE_FILE)
endif

.PHONY: help bootstrap up down build logs test-all clean ps protos envoy-up security-scan

help:
	@echo "\n 🚀 EquaFlow Advanced Orchestrator (Makefile v11.0)"
	@echo "======================================================="
	@echo "Automation:"
	@echo "  bootstrap    - Zero-Touch stack & security initialization"
	@echo "  up           - Start the entire stack (Background)"
	@echo "  down         - Stop and remove all containers"
	@echo "  build        - Rebuild all core images"
	@echo "  ps           - Show process status"
	@echo ""
	@echo "Development & Quality:"
	@echo "  protos       - Generate gRPC/Protobuf bindings (Rust & Python)"
	@echo "  lint         - Run high-fidelity linters (Ruff, Cargo Clippy)"
	@echo "  format       - Auto-format codebase (Ruff, Cargo Fmt)"
	@echo ""
	@echo "Testing:"
	@echo "  test-all     - Run THE GAUNTLET (Rust + Python + E2E)"
	@echo "  test-rust    - Execute Rust unit & integration tests"
	@echo "  test-python  - Execute Python test suite"
	@echo "  test-e2e     - Run Playwright Frontend E2E tests"
	@echo "  test-self-heal - Run tests with self-healing retry logic"
	@echo ""
	@echo "Infrastructure:"
	@echo "  envoy-up     - Launch the Envoy API Gateway"
	@echo "  db-shell     - Open psql for TimescaleDB"
	@echo "=======================================================\n"

bootstrap:
	@chmod +x bootstrap.sh
	@./bootstrap.sh

setup-dev:
	@chmod +x scripts/setup_dev.sh
	@./scripts/setup_dev.sh

up:
	$(DOCKER_COMPOSE) up -d

down:
	$(DOCKER_COMPOSE) down

build:
	$(DOCKER_COMPOSE) build

ps:
	$(DOCKER_COMPOSE) ps

logs:
	$(DOCKER_COMPOSE) logs -f

# --- Testing Hub ---

# === [Security] Institutional Hardening ===
# === [Security] Institutional Hardening ===
security-scan:
	@echo "🛡️ Running Trivy Filesystem Scan..."
	@trivy fs --severity HIGH,CRITICAL .
	@echo "🔍 Running Bandit Security Linter..."
	@$(DOCKER_COMPOSE) run --rm api bandit -r core/ services/ -c pyproject.toml
	@echo "🕵️ Running Pip-Audit..."
	@$(DOCKER_COMPOSE) run --rm api pip-audit
	@echo "🐳 Running Trivy Image Scans..."
	@trivy image equaflow-api:latest
	@trivy image equaflow-rust-core:latest

blue-green-deploy:
	@chmod +x scripts/blue_green_deploy.sh
	@./scripts/blue_green_deploy.sh

test-all:
	@echo "🔥 Running The Gauntlet (Institutional Grade)..."
	@echo "--- [Rust Core] ---"
	@$(DOCKER_COMPOSE) run --rm rust-src.shared cargo fmt -- --check
	@$(DOCKER_COMPOSE) run --rm rust-src.shared cargo clippy -- -D warnings
	@$(DOCKER_COMPOSE) run --rm rust-src.shared cargo test
	@echo "--- [Python API] ---"
	@$(DOCKER_COMPOSE) run --rm api pytest tests/unit
	@echo "--- [E2E & Auth] ---"
	@$(DOCKER_COMPOSE) --profile test up e2e-test --abort-on-container-exit
	@echo "✅ Gauntlet Passed."

test-rust:
	$(DOCKER_COMPOSE) run --rm rust-src.shared cargo test

test-python:
	$(DOCKER_COMPOSE) run --rm api pytest tests/unit

test-e2e:
	@echo "🌐 Running Frontend E2E Tests..."
	@pytest tests/e2e/test_frontend.py

test-self-heal:
	@echo "🩹 Running Self-Healing Test Suite..."
	@pytest -m self_heal

# --- Advanced Builds ---

protos:
	@echo "🧬 Generating Cross-Language gRPC Bindings..."
	@# Python Bindings
	@$(DOCKER_COMPOSE) run --rm api python3 -m grpc_tools.protoc \
		-I=core/protos --python_out=core/protos --grpc_python_out=core/protos core/protos/*.proto
	@# Rust Bindings
	$(DOCKER_COMPOSE) run --rm rust-src.shared cargo build

lint:
	ruff check .
	$(DOCKER_COMPOSE) run --rm rust-src.shared cargo clippy -- -D warnings || true

format:
	ruff format .
	$(DOCKER_COMPOSE) run --rm rust-src.shared cargo fmt || true

envoy-up:
	@echo "🕸️  Launching Envoy Edge Proxy..."
	$(DOCKER_COMPOSE) up -d envoy

db-shell:
	$(DOCKER_COMPOSE) exec postgres psql -U admin -d bsopt

alembic:
	$(DOCKER_COMPOSE) run --rm api alembic $(ARGS)

institutional-ready:
	@echo "🏢 Initializing Institutional Verification Flow..."
	@$(MAKE) up
	@echo "🩺 Running Containerized Readiness Checks..."
	@$(DOCKER_COMPOSE) run --rm api python3 scripts/verify_readiness.py
	@echo "🚀 Executing Institutional Smoke Test..."
	@$(DOCKER_COMPOSE) run --rm api python3 scripts/institutional_smoke_test.py
	@echo "📊 Benchmarking System Latency..."
	@$(DOCKER_COMPOSE) run --rm api python3 scripts/benchmark_latency.py
	@echo "✅ INSTITUTIONAL STATUS: GREEN"

clean:
	$(DOCKER_COMPOSE) down -v
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf venv/ .venv/ target/
	rm -rf services/quant/rust-core/target/
