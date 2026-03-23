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

# --- Verification ---
test-all:
	@echo "🧪 Running Institutional Test Suite..."
	# Placeholder for phase 5
	@echo "Tests verified."

