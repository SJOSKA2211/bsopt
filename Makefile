.PHONY: up down build test clean

up:
	docker-compose up

down:
	docker-compose down

build:
	docker-compose build

test:
	docker-compose run --rm test-runner pytest

clean:
	docker-compose down -v
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
