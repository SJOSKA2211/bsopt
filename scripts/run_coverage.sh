#!/bin/bash

# Initialize coverage
.venv/bin/coverage erase

echo "🥒 Running API Tests..."
.venv/bin/pytest tests/api/ --cov=src/api --cov-append

echo "🥒 Running Auth/Security Tests..."
.venv/bin/pytest tests/auth/ tests/security/ --cov=src/auth --cov=src/security --cov-append

echo "🥒 Running Database Tests..."
.venv/bin/pytest tests/database/ --cov=src/database --cov-append

echo "🥒 Running ML Tests..."
.venv/bin/pytest tests/ml/ --cov=src/ml --cov-append

echo "🥒 Running Pricing Tests..."
.venv/bin/pytest tests/pricing/ --cov=src/pricing --cov-append

echo "🥒 Running Infra/Docker Tests..."
.venv/bin/pytest tests/infra/ tests/docker/ --cov=src/infra --cov-append || true # Allow failures for now

echo "🥒 Running Streaming Tests..."
.venv/bin/pytest tests/streaming/ --cov=src/streaming --cov-append || true

echo "🥒 Running Scraper Tests..."
.venv/bin/pytest tests/scrapers/ --cov=src/scrapers --cov-append || true

echo "🥒 Running Worker Tests..."
.venv/bin/pytest tests/workers/ --cov=src/workers --cov-append || true

echo "🥒 Running Shared/Utils Tests..."
.venv/bin/pytest tests/shared/ tests/utils/ --cov=src/shared --cov=src/utils --cov-append || true

echo "🥒 Generating Report..."
.venv/bin/coverage report
.venv/bin/coverage xml
echo "🥒 Done!"
