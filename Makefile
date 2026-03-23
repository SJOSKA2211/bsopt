# Makefile
CONTAINER_ENGINE ?= $(shell command -v podman 2> /dev/null || echo docker)
COMPOSE_CMD ?= $(shell if [ "$(CONTAINER_ENGINE)" = "podman" ]; then echo "podman-compose"; else if $(CONTAINER_ENGINE) compose version > /dev/null 2>&1; then echo "$(CONTAINER_ENGINE) compose"; else echo "docker-compose"; fi; fi)

.PHONY: all bootstrap up down clean logs

all: bootstrap

bootstrap:
	@echo "Using container engine: $(CONTAINER_ENGINE)"
	@bash scripts/bootstrap.sh

up:
	$(COMPOSE_CMD) -f infrastructure/docker-compose.yml up -d

down:
	$(COMPOSE_CMD) -f infrastructure/docker-compose.yml down

clean: down
	$(CONTAINER_ENGINE) volume rm equaflow_pgdata || true
	rm -f .env

logs:
	$(COMPOSE_CMD) -f infrastructure/docker-compose.yml logs -f
