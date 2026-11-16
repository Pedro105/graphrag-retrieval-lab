.DEFAULT_GOAL := help
.PHONY: help install up down api lint typecheck test test-integration check populate eval

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

install:  ## Install dependencies (incl. dev) with uv
	uv sync

up:  ## Start FalkorDB in Docker (browser on :3000)
	docker compose up -d --wait falkordb

down:  ## Stop the Docker stack
	docker compose down

api:  ## Run the API locally with reload (needs `make up` and a .env)
	uv run uvicorn graphrag_lab.api.app:app --reload

lint:  ## Ruff lint + format check
	uv run ruff check src tests scripts eval
	uv run ruff format --check src tests scripts eval

typecheck:  ## mypy --strict on src
	uv run mypy src

test:  ## Unit tests (offline, no services needed)
	uv run pytest tests/unit

test-integration:  ## Integration tests (needs `make up`; LLM tests also need API keys)
	uv run pytest tests/integration

check: lint typecheck test  ## Everything CI runs without services

populate:  ## Ingest the synthetic eval graph through the running API
	uv run python scripts/populate_graph.py

eval:  ## Run the recall smoke eval against the running API
	uv run python eval/run_eval.py
