# Black-Scholes Option Pricing Platform - Makefile
# Quick commands for common development tasks

.PHONY: help setup build up down logs test clean install lint format

help: ## Show this help message
	@echo "Black-Scholes Option Pricing Platform - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Run initial setup
	@echo "Running setup script..."
	./setup.sh

build: ## Build Docker images
	@echo "Building Docker images..."
	docker compose build

up: ## Start all services
	@echo "Starting services..."
	docker compose up -d
	@echo "Services started. Access:"
	@echo "  API: http://localhost:8000"
	@echo "  API Docs: http://localhost:8000/docs"
	@echo "  Frontend: http://localhost:3000"
	@echo "  MLflow: http://localhost:5000"
	@echo "  Jupyter: http://localhost:8888"

down: ## Stop all services
	@echo "Stopping services..."
	docker compose down

restart: down up ## Restart all services

logs: ## View logs (use service=<name> for specific service)
	@if [ -z "$(service)" ]; then \
		docker compose logs -f; \
	else \
		docker compose logs -f $(service); \
	fi

ps: ## Show running services
	docker compose ps

shell: ## Open shell in API container
	docker compose exec api /bin/bash

python: ## Open Python REPL in API container
	docker compose exec api python

db-shell: ## Open PostgreSQL shell
	docker compose exec postgres psql -U admin -d bsopt

redis-shell: ## Open Redis CLI
	docker compose exec redis redis-cli -a changeme

install: ## Install Python dependencies locally
	pip install -e ".[dev]"

test: ## Run tests
	docker compose exec api pytest -v

test-cov: ## Run tests with coverage
	docker compose exec api pytest --cov=src --cov-report=html --cov-report=term

test-local: ## Run tests locally (not in Docker)
	pytest -v

lint: ## Run linters
	flake8 src/ tests/
	mypy src/

format: ## Format code
	black src/ tests/
	isort src/ tests/

clean: ## Remove temporary files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

check-all: format lint test ## Run formatters, linters, and tests

clean-all: clean down ## Remove all temporary files and stop services
	docker compose down -v
	rm -rf logs/*

init-db: ## Initialize database
	docker compose exec api python -c "from src.database import init_db; init_db()"

migrate: ## Run database migrations
	docker compose exec api alembic upgrade head

migrate-create: ## Create new migration (use msg='description')
	docker compose exec api alembic revision --autogenerate -m "$(msg)"

seed-db: ## Seed database with sample data
	docker compose exec api python scripts/seed_db.py

price: ## Quick option pricing (interactive)
	@echo "Enter option parameters (or press Ctrl+C to exit):"
	@docker compose exec -T api python -c "\
	from src.pricing.black_scholes import BlackScholesEngine, BSParameters; \
	params = BSParameters(100, 100, 1.0, 0.25, 0.05, 0.02); \
	price = BlackScholesEngine.price_call(params); \
	greeks = BlackScholesEngine.calculate_greeks(params, 'call'); \
	print(f'Call Price: \$$${price:.4f}'); \
	print(f'Delta: {greeks.delta:.4f}'); \
	print(f'Gamma: {greeks.gamma:.4f}')"

benchmark: ## Run performance benchmarks
	docker compose exec api python -m pytest tests/performance/ --benchmark-only

health: ## Check service health
	@echo "Checking service health..."
	@curl -s http://localhost:8000/health | python -m json.tool || echo "API not responding"

notebook: ## Start Jupyter notebook
	@echo "Jupyter available at: http://localhost:8888"
	@open http://localhost:8888 2>/dev/null || xdg-open http://localhost:8888 2>/dev/null || echo "Open manually: http://localhost:8888"

docs: ## Build documentation
	cd docs && make html

docs-serve: ## Serve documentation
	cd docs && make serve

validate-openapi: ## Validate OpenAPI specification
	docker run --rm -v ${PWD}/docs/api:/specs openapitools/openapi-generator-cli validate -i /specs/openapi.yaml

generate-client: ## Generate Python client from OpenAPI (use lang=python|typescript|java)
	@if [ -z "$(lang)" ]; then \
		echo "Usage: make generate-client lang=python"; \
	else \
		docker run --rm -v ${PWD}:/local openapitools/openapi-generator-cli generate \
		-i /local/docs/api/openapi.yaml \
		-g $(lang) \
		-o /local/clients/$(lang); \
	fi

backup-db: ## Backup database
	@mkdir -p backups
	docker compose exec -T postgres pg_dump -U admin bsopt > backups/backup_$$(date +%Y%m%d_%H%M%S).sql
	@echo "Database backed up to backups/"

restore-db: ## Restore database (use file=backups/filename.sql)
	@if [ -z "$(file)" ]; then \
		echo "Usage: make restore-db file=backups/backup.sql"; \
	else \
		docker compose exec -T postgres psql -U admin bsopt < $(file); \
		echo "Database restored from $(file)"; \
	fi

stats: ## Show project statistics
	@echo "=== Project Statistics ==="
	@echo "Total Python files:"
	@find src -name "*.py" | wc -l
	@echo "Total lines of code:"
	@find src -name "*.py" -exec wc -l {} + | tail -1
	@echo "Total tests:"
	@find tests -name "test_*.py" -exec wc -l {} + | tail -1
	@echo "Docker services:"
	@docker compose ps | wc -l

watch-logs: ## Watch logs with automatic refresh
	watch -n 1 'docker compose logs --tail=50'

.DEFAULT_GOAL := help

train: ## Train ML models
	docker compose exec api python src/ml/pipelines/orchestrator.py train --model xgboost
