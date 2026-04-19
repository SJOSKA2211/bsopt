.PHONY: help bootstrap up down ps logs test lint clean

# --- Core Protocol Commands ---
help:
	@echo "OMARCHY PRIME: Deterministic Execution Engine"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  bootstrap    Initialize the environment (uv, pki)"
	@echo "  up           Deploy all containers via Sequential Build"
	@echo "  down         Teardown all containers"
	@echo "  ps           Show running containers"
	@echo "  logs         Show logs"
	@echo "  test         Run pytest suite"
	@echo "  lint         Run ruff audit (Node 1)"
	@echo "  clean        Cleanup workspace (Node 0)"

bootstrap:
	uv python install 3.12.13
	bash scripts/setup_pki.sh

up:
	docker compose up -d timescaledb redis rabbitmq pgbouncer auth_service api_service

down:
	docker compose down

ps:
	docker compose ps

logs:
	docker compose logs -f

test:
	pytest tests/unit tests/integration

lint:
	ruff check --select ALL .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Workspace sanitized (Node 0)."
