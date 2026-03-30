# Centralized Makefile for BSOPT (Vercel-ready)

.PHONY: install test lint build dev clean scraper-up scraper-status scraper-wait health

install:
	cd src/frontend && npm install
	uv pip install -r requirements.txt

dev:
	cd src/frontend && npm run dev

test:
	uv run pytest tests/unit tests/integration

test-e2e:
	cd src/frontend && npx playwright test

lint:
	uv run ruff check .
	cd src/frontend && npm run lint

build:
	cd src/frontend && npm run build

clean:
	rm -rf src/frontend/dist
	find . -type d -name "__pycache__" -exec rm -rf {} +

# --- Scraper & Health Revamp ---
scraper-up:
	scripts/run_scraper_until_healthy.sh

scraper-status:
	uv run python scripts/engine_health.py

scraper-wait:
	uv run python scripts/engine_health.py --wait

health: scraper-status
