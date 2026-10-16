.PHONY: help setup install lint format typecheck test test-cov train optimize serve clean docker-build docker-up

PYTHON := python3
UV := uv
RUN := $(UV) run

help:  ## Display available Makefile targets
	@echo "Available commands for Favorita Demand Forecasting & Inventory Optimization:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create virtual environment using uv and install all dependencies
	@command -v $(UV) >/dev/null 2>&1 || (echo "uv not found. Please install uv (curl -LsSf https://astral.sh/uv/install.sh) or use pip." && exit 1)
	$(UV) venv
	$(UV) pip install -e ".[dev]"

install: ## Install package in editable mode with development dependencies
	$(UV) pip install -e ".[dev]" || $(PYTHON) -m pip install -e ".[dev]"

lint: ## Check source code using ruff
	$(RUN) ruff check src tests configs

format: ## Format source code using ruff
	$(RUN) ruff format src tests configs
	$(RUN) ruff check --fix src tests configs

typecheck: ## Run strict static type checking with mypy
	$(RUN) mypy src tests

test: ## Run unit tests with pytest
	$(RUN) pytest tests

test-cov: ## Run unit tests with coverage reporting
	$(RUN) pytest --cov=src --cov-report=term-missing --cov-report=html:tests/coverage tests

train: ## Train hierarchical LightGBM & CatBoost models on processed data
	$(RUN) python -m src.models.train --config configs/model_config.yaml

optimize: ## Run Monte Carlo stochastic (s, S) inventory optimization simulation
	$(RUN) python -m src.optimization.simulation --config configs/base_config.yaml

notebooks: ## Execute analytical notebooks and embed static PNG charts for GitHub viewer
	$(UV) run --with nbclient --with nbformat --with ipykernel --with kaleido python scripts/execute_notebooks.py

serve: ## Launch the FastAPI production service for real-time inference
	$(RUN) uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build: ## Build docker container for inference API
	docker build -f docker/Dockerfile.api -t favorita-forecast-api:latest .

docker-up: ## Start docker-compose stack
	docker compose -f docker/docker-compose.yml up --build -d

clean: ## Remove caches, bytecode, and coverage artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov tests/coverage .coverage dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
