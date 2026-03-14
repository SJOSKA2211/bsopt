# ==============================================================================
# EQUAFLOW: THE INSTITUTIONAL-GRADE MANIFOLD (Makefile v11.0)
# ==============================================================================
# Unified Orchestration for Rust, Python, gRPC, and Envoy.
# ==============================================================================

# Detect Docker Compose
DOCKER_COMPOSE := $(shell which docker-compose 2>/dev/null || echo "docker compose")

.PHONY: help bootstrap up down build logs test-all clean ps protos envoy-up

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
	@echo ""
	@echo "Infrastructure:"
	@echo "  envoy-up     - Launch the Envoy API Gateway"
	@echo "  db-shell     - Open psql for TimescaleDB"
	@echo "=======================================================\n"

bootstrap:
	@chmod +x bootstrap.sh
	@./bootstrap.sh

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
security-scan:
	@echo "🛡️ Running Trivy Vulnerability Scan..."
	@trivy fs --severity HIGH,CRITICAL .
	@echo "🔍 Running Bandit Security Linter..."
	@$(DOCKER_COMPOSE) run --rm api bandit -r src/
	@echo "🕵️ Running Pip-Audit..."
	@$(DOCKER_COMPOSE) run --rm api pip-audit

test-all:
	@echo "🔥 Running The Gauntlet (Institutional Grade)..."
	@echo "--- [Rust Core] ---"
	@$(DOCKER_COMPOSE) run --rm rust-core cargo fmt -- --check
	@$(DOCKER_COMPOSE) run --rm rust-core cargo clippy -- -D warnings
	@$(DOCKER_COMPOSE) run --rm rust-core cargo test
	@echo "--- [Python API] ---"
	@$(DOCKER_COMPOSE) run --rm api ruff check .
	@$(DOCKER_COMPOSE) run --rm api ruff format --check .
	@$(DOCKER_COMPOSE) run --rm api pytest tests/unit
	@echo "--- [E2E & Auth] ---"
	@$(DOCKER_COMPOSE) --profile test up e2e-test --abort-on-container-exit
	@echo "✅ Gauntlet Passed."

test-rust:
	$(DOCKER_COMPOSE) run --rm rust-core cargo test

test-python:
	$(DOCKER_COMPOSE) run --rm api pytest tests/unit

# --- Advanced Builds ---

protos:
	@echo "🧬 Generating Cross-Language gRPC Bindings..."
	@# Python Bindings
	@$(DOCKER_COMPOSE) run --rm api python3 -m grpc_tools.protoc \
		-I=src/protos --python_out=src/protos --grpc_python_out=src/protos src/protos/*.proto
	@# Rust Bindings
	@$(DOCKER_COMPOSE) run --rm rust-core cargo build

lint:
	$(DOCKER_COMPOSE) run --rm api ruff check .
	$(DOCKER_COMPOSE) run --rm rust-core cargo clippy -- -D warnings

format:
	$(DOCKER_COMPOSE) run --rm api ruff format .
	$(DOCKER_COMPOSE) run --rm rust-core cargo fmt

envoy-up:
	@echo "🕸️  Launching Envoy Edge Proxy..."
	$(DOCKER_COMPOSE) up -d envoy

db-shell:
	$(DOCKER_COMPOSE) exec postgres psql -U admin -d bsopt

clean:
	$(DOCKER_COMPOSE) down -v
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf target/
	rm -rf src/rust-core/target/
