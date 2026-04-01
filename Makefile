# Centralized Makefile for BSOPT (Vercel-ready)

.PHONY: install test lint build dev clean scraper-up scraper-status scraper-wait health

# --- Operational Substrate ---
install:
	cd src/frontend && npm install
	uv pip install -r requirements.txt

protos:
	scripts/build_protos.sh

# --- Service Orchestration ---
api:
	scripts/start_api.sh

auth:
	scripts/start_auth.sh

mlops:
	scripts/start_mlops.sh

dev:
	cd src/frontend && npm run dev

# --- Quality Assurance ---
test:
	uv run pytest tests/unit tests/integration

test-integration:
	uv run pytest tests/integration

test-e2e:
	cd src/frontend && npx playwright test

lint:
	scripts/lint_all.sh

# --- Scraper & Health Revamp ---
scraper-up:
	scripts/run_scraper_until_healthy.sh

scraper-status:
	uv run python scripts/engine_health.py

scraper-wait:
	uv run python scripts/engine_health.py --wait

scraper-simulate:
	uv run python scripts/engine_health.py --simulate

health: scraper-status

# --- Cleanup ---
clean:
	rm -rf src/frontend/dist
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

