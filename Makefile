# Makefile
# EQUAFLOW: Institutional Compute Factory

# 1. Dynamic Engine Detection
CONTAINER_ENGINE ?= $(shell command -v podman 2> /dev/null || echo docker)
COMPOSE_CMD ?= $(shell if [ "$(CONTAINER_ENGINE)" = "podman" ]; then \
                    if podman compose version > /dev/null 2>&1; then echo "podman compose"; else echo "podman-compose"; fi; \
                else \
                    if docker compose version > /dev/null 2>&1; then echo "docker compose"; else echo "docker-compose"; fi; \
                fi)

# 2. Paths
COMPOSE_FILE = infrastructure/orchestration/docker-compose.yml

.PHONY: all bootstrap build up down clean logs test-all shell ps

all: bootstrap

# --- Lifecycle Management ---
bootstrap:
	@echo "🚀 Initiating Zero-Touch Bootstrap using $(CONTAINER_ENGINE)..."
	@bash bootstrap.sh

build:
	$(COMPOSE_CMD) -f $(COMPOSE_FILE) build --parallel

up:
	$(COMPOSE_CMD) -f $(COMPOSE_FILE) up -d

down:
	$(COMPOSE_CMD) -f $(COMPOSE_FILE) down

clean: down
	@echo "🧹 Deep cleaning EquaFlow stack..."
	$(CONTAINER_ENGINE) volume rm $$( $(CONTAINER_ENGINE) volume ls -q | grep bsopt ) 2>/dev/null || true
	rm -rf .pki .env bootstrap.log

# --- Observability & Debugging ---
logs:
	$(COMPOSE_CMD) -f $(COMPOSE_FILE) logs -f

ps:
	$(COMPOSE_CMD) -f $(COMPOSE_FILE) ps

shell-%:
	$(CONTAINER_ENGINE) exec -it $(shell $(COMPOSE_CMD) -f $(COMPOSE_FILE) ps -q $*) /bin/bash

# --- Build & Generation ---
proto-gen:
	@echo "🧬 Generating gRPC code from protos..."
	@mkdir -p src/shared/protos
	python -m grpc_tools.protoc -I./protos --python_out=src/shared/protos --grpc_python_out=src/shared/protos ./protos/*.proto
	@touch src/shared/protos/__init__.py
	@echo "Proto generation complete."

fbs-gen:
	@echo "📦 Generating FlatBuffers code..."
	@mkdir -p src/shared/fbs
	flatc --python -o src/shared/fbs protos/market_tick.fbs
	@touch src/shared/fbs/__init__.py
	@echo "FlatBuffers generation complete."

clean-gen:
	@echo "🧹 Removing generated code..."
	rm -rf src/shared/protos/*_pb2*.py src/shared/fbs/*
	@echo "Cleanup complete."

gen-all: clean-gen proto-gen fbs-gen

# --- Verification ---
test-all:
	@echo "🧪 Running Institutional Test Suite..."
	pytest tests/
	@echo "Tests verified."

lint:
	@echo "🔍 Running codebase linting (ruff)..."
	ruff check .
	ruff format --check .
	@echo "Linting complete."

verify-proto: proto-gen
	@echo "🧪 Verifying generated proto integrity..."
	python -c "from src.shared.protos import data_pb2; print('Proto Data Engine: Synchronized')"
	@echo "Proto verification complete."

help:
	@echo "EquaFlow Institutional Build Factory help:"
	@echo "  bootstrap    - Run zero-touch bootstrap"
	@echo "  build        - Build container images"
	@echo "  up           - Start services"
	@echo "  down         - Stop services"
	@echo "  gen-all      - Generate gRPC and FlatBuffers code"
	@echo "  test-all     - Run full test suite"
	@echo "  lint         - Run ruff linting and formatting check"
	@echo "  verify-proto - Verify generated protocol integrity"
	@echo "  clean        - Deep clean volumes and logs"

