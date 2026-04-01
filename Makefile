# BS-OPT Unified Makefile

.PHONY: help build test lint run-api health-check clean

help:
	@echo "BS-OPT Unified Build System"
	@echo "Usage:"
	@echo "  make build         Build the project"
	@echo "  make test          Run all pytest suites"
	@echo "  make lint          Run linting checks"
	@echo "  make run-api       Start the FastAPI engine"
	@echo "  make health-check  Report engine health"
	@echo "  make clean         Cleanup temporary files"

build:
	@echo "Building BS-OPT..."
	pip install -r requirements.txt

test:
	@echo "Running tests..."
	pytest tests/unit

lint:
	@echo "Linting codebase..."
	ruff check .

run-api:
	@echo "Starting API..."
	uvicorn api.index:app --host 0.0.0.0 --port 8000

health-check:
	@echo "Checking health..."
	python3 scripts/engine_health.py --simulate

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
