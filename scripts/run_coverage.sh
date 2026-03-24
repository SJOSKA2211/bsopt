#!/bin/bash

# Initialize coverage
coverage erase

echo " Running API Tests..."
pytest tests/api/ --cov=src/api --cov-append

echo " Running Auth/Security Tests..."
pytest tests/auth/ tests/security/ --cov=src/auth --cov=src/security --cov-append

echo " Running Database Tests..."
pytest tests/database/ --cov=src/database --cov-append

echo " Running ML Tests..."
pytest tests/ml/ --cov=src/ml --cov-append

echo " Running Pricing Tests..."
pytest tests/pricing/ --cov=src/pricing --cov-append

echo " Running Infra/Docker Tests..."
pytest tests/infra/ tests/docker/ --cov=src/infra --cov-append || true # Allow failures for now

echo " Running Streaming Tests..."
pytest tests/streaming/ --cov=src/streaming --cov-append || true

echo " Running Scraper Tests..."
pytest tests/scrapers/ --cov=src/scrapers --cov-append || true

echo " Running Worker Tests..."
pytest tests/workers/ --cov=src/workers --cov-append || true

echo " Running Shared/Utils Tests..."
pytest tests/shared/ tests/utils/ --cov=src/shared --cov=src/utils --cov-append || true

echo " Generating Report..."
coverage report
coverage xml
echo " Done!"
