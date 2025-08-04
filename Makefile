# Makefile for Explainable MoE Debugger

.PHONY: help install install-dev test lint format type-check pre-commit build clean docs serve docker-build docker-run

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install package in development mode"
	@echo "  install-dev  - Install with development dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  type-check   - Run type checking"
	@echo "  pre-commit   - Install pre-commit hooks"
	@echo "  build        - Build package"
	@echo "  clean        - Clean build artifacts"
	@echo "  docs         - Build documentation"
	@echo "  serve        - Run development server"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e .[dev,database,cache,huggingface]

install-all:
	pip install -e .[all]

# Development targets
test:
	pytest tests/ -v --cov=src/moe_debugger --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v -x --ff

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

type-check:
	mypy src/moe_debugger --ignore-missing-imports

pre-commit:
	pre-commit install
	pre-commit run --all-files

# Build targets
build:
	python -m build

build-frontend:
	cd frontend && npm install && npm run build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs && make livehtml

# Development server
serve:
	moe-debugger --port 8080 --host 0.0.0.0

serve-dev:
	uvicorn moe_debugger.server:app --reload --host 0.0.0.0 --port 8080

# Database operations
db-upgrade:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-migration:
	alembic revision --autogenerate -m "$(message)"

db-reset:
	rm -f moe_debugger.db
	alembic upgrade head

# Docker targets
docker-build:
	docker build -t moe-debugger .

docker-run:
	docker run -p 8080:8080 moe-debugger

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

# Testing with different backends
test-memory:
	CACHE_TYPE=memory pytest tests/

test-redis:
	CACHE_TYPE=redis REDIS_URL=redis://localhost:6379 pytest tests/

# Performance testing
benchmark:
	python -m pytest tests/performance/ -v --benchmark-only

# Security scanning
security:
	bandit -r src/
	safety check

# Release targets
version-patch:
	bump2version patch

version-minor:
	bump2version minor

version-major:
	bump2version major

release: test lint type-check build
	@echo "Ready for release. Run 'make publish' to upload to PyPI."

publish:
	twine upload dist/*

# Development utilities
jupyter:
	jupyter lab --port 8888 --no-browser --allow-root

shell:
	python -c "from moe_debugger import *; import IPython; IPython.embed()"

profile:
	python -m cProfile -o profile.stats -m moe_debugger.cli --help
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Example runs
example-mixtral:
	moe-debugger --model "mistralai/Mixtral-8x7B-Instruct-v0.1" --port 8080

example-local:
	moe-debugger --model ./models/my_moe_model.pt --port 8080

# CI/CD helpers
ci-install:
	pip install -e .[dev,database,cache]

ci-test:
	pytest tests/ --cov=src/moe_debugger --cov-report=xml --junitxml=test-results.xml

ci-lint:
	ruff check src/ tests/ --output-format=github