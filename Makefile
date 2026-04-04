# BS-OPT Unified Makefile

PYTHON := ./.venv/bin/python
PIP := ./.venv/bin/pip
UVICORN := ./.venv/bin/uvicorn
PYTEST := ./.venv/bin/pytest
RUFF := ./.venv/bin/ruff
MATURIN := $(shell pwd)/.venv/bin/maturin

.PHONY: help build rust-build test rust-test rust-bench lint format run-api ml-train ml-serve health-check clean bootstrap setup-pki

help:
	@echo "BS-OPT Unified Build System"
	@echo "Usage:"
	@echo "  make build         Build the Python project"
	@echo "  make rust-build    Build the Rust math core"
	@echo "  make bootstrap     Full system bootstrap (mTLS, Secrets, Containers)"
	@echo "  make setup-pki     Initialize mTLS certificates"
	@echo "  make test          Run Python unit tests"
	@echo "  make rust-test     Run Rust unit tests"
	@echo "  make rust-bench    Run Rust micro-benchmarks"
	@echo "  make lint          Run linting (Ruff)"
	@echo "  make format        Run formatting (Ruff)"
	@echo "  make run-api       Start the FastAPI engine locally"
	@echo "  make ml-train      Trigger ML model training"
	@echo "  make ml-serve      Start the ML inference engine"
	@echo "  make health-check  Report engine health"
	@echo "  make clean         Cleanup temporary files"

build:
	@echo "Building BS-OPT Python..."
	$(PIP) install -e .

# build:
# 	@echo "Building Rust Core..."
# 	cd src/math_kernel/rust-core && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 $(MATURIN) develop --release

setup-pki:
	@echo "Setting up PKI..."
	./scripts/setup_pki.sh

bootstrap:
	@echo "Starting Full Bootstrap..."
	./bootstrap.sh

test:
	@echo "Running Python tests..."
	$(PYTEST) tests/unit

rust-test:
	@echo "Running Rust tests..."
	cd src/math_kernel/rust-core && cargo test --release

rust-bench:
	@echo "Running Rust benchmarks..."
	cd src/math_kernel/rust-core && cargo bench

lint:
	@echo "Linting codebase..."
	$(RUFF) check .

format:
	@echo "Formatting codebase..."
	$(RUFF) format .

run-api:
	@echo "Starting API..."
	bash start_api.sh

ml-train:
	@echo "Training models..."
	$(PYTHON) src/ml/training/train_all.py

ml-serve:
	@echo "Starting ML Inference Service..."
	$(UVICORN) src.ml.serving.serve:app --host 0.0.0.0 --port 8001

health-check:
	@echo "Checking health..."
	curl -s http://localhost:8000/health | jq .

rmq-clean:
	@echo "Cleaning RabbitMQ queues..."
	docker exec bsopt-rabbitmq-1 rabbitmqctl delete_vhost /
	docker exec bsopt-rabbitmq-1 rabbitmqctl add_vhost /
	docker exec bsopt-rabbitmq-1 rabbitmqctl set_permissions -p / bsopt_admin ".*" ".*" ".*"

docker-clean:
	docker system prune -a --volumes -f

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	cd src/math_kernel/rust-core && cargo clean
