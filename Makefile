# Makefile

# 1. Dynamic Engine Detection
CONTAINER_ENGINE ?= $(shell command -v podman 2> /dev/null || echo docker)
COMPOSE_CMD ?= $(shell if [ "$(CONTAINER_ENGINE)" = "podman" ]; then \
                    if podman compose version > /dev/null 2>&1; then echo "podman compose"; else echo "podman-compose"; fi; \
                else \
                    if docker compose version > /dev/null 2>&1; then echo "docker compose"; else echo "docker-compose"; fi; \
                fi)

# 2. Paths
COMPOSE_FILE = infrastructure/orchestration/docker-compose.yml

all: setup

# --- Orchestration & Initialization ---
setup:
	@echo "🚀 Initiating Zero-Touch Setup..."
	@bash scripts/bootstrap.sh
	@$(MAKE) gen-all
	@echo "🎯 System Fully Initialized."

# bootstrap: Fully Autonomous "Zero-Touch" Initialization
# It enforces executable permissions on all system scripts and initializes PKI, Secrets, and core Infra.
bootstrap: setup

revamp: clean setup
	@echo "🔄 Full System Revamp Complete."

# --- Lifecycle Management ---
infra:
	@bash scripts/start_infra.sh

up:
	@bash scripts/start_all_dev.sh

down:
	$(COMPOSE_CMD) -f $(COMPOSE_FILE) down

clean: down
	@echo "🧹 Deep cleaning Manifold stack..."
	$(CONTAINER_ENGINE) volume rm $$( $(CONTAINER_ENGINE) volume ls -q | grep bsopt ) 2>/dev/null || true
	rm -rf .pki .env bootstrap.log logs/*.log
	@echo "✨ System Sanitized."

# --- Observability & Debugging ---
logs:
	$(COMPOSE_CMD) -f $(COMPOSE_FILE) logs -f

ps:
	$(COMPOSE_CMD) -f $(COMPOSE_FILE) ps

shell-%:
	$(CONTAINER_ENGINE) exec -it $(shell $(COMPOSE_CMD) -f $(COMPOSE_FILE) ps -q $*) /bin/bash

# --- Data & DB ---
db-init:
	@bash scripts/deploy_full_db.sh

db-sync:
	@bash scripts/deploy_db_updates.sh

ingest:
	@bash scripts/run_ingestion.sh

# --- Build & Generation ---
gen-all:
	@bash scripts/build_protos.sh
	@$(MAKE) verify-proto

clean-gen:
	@echo "🧹 Removing generated code..."
	rm -rf src/shared/protos/*_pb2*.py src/shared/fbs/*
	@echo "Cleanup complete."

# --- Verification ---
test-unit:
	@echo "🧪 Running Unit Tests..."
	uv run pytest tests/unit

test-integration:
	@echo "🧪 Running Integration Tests (Data-Driven)..."
	uv run pytest tests/integration

test-e2e:
	@echo "🎭 Running Playwright E2E Tests..."
	uv run playwright test tests/e2e

test-all: test-unit test-integration test-e2e
	@echo "✅ Full test suite verified."

lint:
	@echo "🔍 Running codebase linting (ruff)..."
	uv run ruff check .
	uv run ruff format --check .
	@echo "✅ Linting complete."

verify-proto:
	@echo "🧪 Verifying generated proto integrity..."
	@uv run python -c "from src.shared.protos import market_data_pb2; print('Proto Data Engine: Synchronized')"
	@echo "✅ Proto verification complete."

build:
	$(COMPOSE_CMD) -f $(COMPOSE_FILE) build --parallel

help:
	@echo "Manifold Production Build Factory (v2026):"
	@echo "  setup        - Production Zero-Touch initialize (Bootstrap + Gen)"
	@echo "  infra        - Start Core Infrastructure substrate (DB, Cache, MQ)"
	@echo "  up           - Start Full Application Ecosystem (Build + Up)"
	@echo "  down         - Graceful shutdown of all services"
	@echo "  build        - Production parallel image construction"
	@echo "  db-init      - Full schema deployment & migration factory"
	@echo "  db-sync      - Synchronize analytical views and optimizations"
	@echo "  gen-all      - Multi-language protocol generation (gRPC/FBS)"
	@echo "  test-all     - Comprehensive verification (Unit, Integration, E2E)"
	@echo "  lint         - Production code quality & format audit"
	@echo "  clean        - Deep purge of volumes, logs, and artifacts"

