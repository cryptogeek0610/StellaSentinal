ifneq (,$(wildcard .env))
include .env
export
endif

FRONTEND_PORT ?= 3000

.PHONY: help build up down restart logs shell test verify clean status

# ============================================================================
# Core Workflow Commands
# ============================================================================

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Core Commands:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -v '(debug)' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ''
	@echo 'Debug Commands (service-specific):'
	@grep -E '^[a-zA-Z_-]+:.*?## \(debug\).*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## \\(debug\\) "}; {printf "  \033[33m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build all Docker images
	docker compose build

up: ## Start all services
	docker compose up -d
	@echo "Services starting... API: http://localhost:8000 | Frontend: http://localhost:$(FRONTEND_PORT)"

down: ## Stop all services
	docker compose down

restart: down up ## Restart all services

logs: ## Show logs from all services (Ctrl+C to exit)
	docker compose logs -f

shell: ## Open shell in app container
	docker compose exec app /bin/bash

test: ## Run pytest test suite
	docker compose run --rm app python -m pytest tests/ -v

verify: ## Run full verification (backend + frontend)
	python3 scripts/verify.py

clean: ## Remove containers, volumes, and images
	docker compose down -v --rmi local

status: ## Show status of all services with health
	@echo "Service Status:"
	@docker compose ps
	@echo ""
	@echo "Endpoints:"
	@echo "  API:        http://localhost:8000"
	@echo "  API Docs:   http://localhost:8000/docs"
	@echo "  Frontend:   http://localhost:$(FRONTEND_PORT)"
	@echo "  PostgreSQL: localhost:5432"
	@echo "  Redis:      localhost:6379"

reset: ## Full reset: stop, remove volumes, rebuild, start
	docker compose down -v
	docker compose build --no-cache
	docker compose up -d
	@echo "Services reset and ready!"

# ============================================================================
# Development Commands
# ============================================================================

run-api: ## Run FastAPI server locally (without Docker)
	uvicorn device_anomaly.api.main:app --reload --port 8000 --host 0.0.0.0

run-frontend: ## Run frontend dev server locally
	cd frontend && npm run dev

# ============================================================================
# Debug Commands (Service-Specific)
# ============================================================================

logs-app: ## (debug) Show app container logs only
	docker compose logs -f app

logs-ollama: ## (debug) Show Ollama container logs
	docker compose logs -f ollama

pull-model: ## (debug) Pull llama3.2 model for Ollama
	docker compose exec ollama ollama pull llama3.2

test-api: ## (debug) Test API health endpoint
	@curl -sf http://localhost:8000/health && echo " API OK" || echo " API not responding"

test-ollama: ## (debug) Test Ollama service
	@curl -sf http://localhost:11434/api/tags > /dev/null && echo "Ollama OK" || echo "Ollama not responding"

test-qdrant: ## (debug) Test Qdrant service
	@curl -sf http://localhost:6333/health > /dev/null && echo "Qdrant OK" || echo "Qdrant not responding"
