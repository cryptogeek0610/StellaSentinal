ifneq (,$(wildcard .env))
include .env
export
endif

FRONTEND_PORT ?= 3000
FRONTEND_DEV_PORT ?= 5173

.PHONY: help build up down restart logs shell test-synthetic test-dw reset clean

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build Docker images
	docker-compose build

up: ## Start all services
	docker-compose up -d
	@echo "Waiting for SQL Server to be ready..."
	@timeout 60 bash -c 'until docker-compose exec -T sqlserver /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P "$${DW_DB_PASS:?set in .env}" -Q "SELECT 1" &>/dev/null; do sleep 2; done' || true
	@echo "Services are up!"

down: ## Stop all services
	docker-compose down

restart: down up ## Restart all services

logs: ## Show logs from all services
	docker-compose logs -f

logs-app: ## Show logs from app service only
	docker-compose logs -f app

logs-db: ## Show logs from SQL Server only
	docker-compose logs -f sqlserver

shell: ## Open shell in app container
	docker-compose exec app /bin/bash

shell-db: ## Open SQL Server command line
	docker-compose exec sqlserver /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P "$${DW_DB_PASS:?set in .env}"

test-synthetic: ## Run synthetic experiment (persists to DW tables)
	docker-compose run --rm app python -m device_anomaly.cli.synthetic_experiment

test-dw: ## Run DW experiment (requires DB with data)
	docker-compose run --rm app python -m device_anomaly.cli.dw_experiment

test-main: ## Run main entry point
	docker-compose run --rm app python -m device_anomaly.cli.main

init-db: ## Initialize database (create if not exists)
	./scripts/init_db.sh

test-llm: ## Test local LLM connection
	./scripts/test_llm_connection.sh

run-api: ## Run FastAPI server (local development)
	uvicorn device_anomaly.api.main:app --reload --port 8000 --host 0.0.0.0

run-frontend: ## Run frontend dev server (requires: cd frontend && npm install)
	cd frontend && npm run dev

build-frontend: ## Build frontend Docker image
	docker-compose build frontend

rebuild-frontend: ## Rebuild frontend Docker image (no cache)
	docker-compose build --no-cache frontend

reset: ## Reset everything: stop, remove volumes, rebuild, and start
	docker-compose down -v
	docker-compose build --no-cache
	docker-compose up -d
	@echo "Waiting for SQL Server to be ready..."
	@timeout 60 bash -c 'until docker-compose exec -T sqlserver /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P "$${DW_DB_PASS:?set in .env}" -Q "SELECT 1" &>/dev/null; do sleep 2; done' || true
	@echo "Services reset and ready!"

clean: ## Remove containers, volumes, and images
	docker-compose down -v --rmi local

# ============================================================================
# Database Commands
# ============================================================================

init-backend-db: ## Initialize backend database schema (SOTI_AnomalyDetection)
	@echo "Creating backend database and schema..."
	docker-compose exec -T sqlserver /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P "$${DW_DB_PASS:?set in .env}" -i /app/scripts/init_backend_schema.sql
	@echo "Backend database schema initialized!"

create-dw-db: ## Create XSight DW database (empty)
	@echo "Creating SOTI_XSight_dw database..."
	docker-compose exec -T sqlserver /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P "$${DW_DB_PASS:?set in .env}" -Q "IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = 'SOTI_XSight_dw') CREATE DATABASE SOTI_XSight_dw;"
	@echo "DW database created!"

show-dbs: ## Show all databases in SQL Server
	docker-compose exec -T sqlserver /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P "$${DW_DB_PASS:?set in .env}" -Q "SELECT name FROM sys.databases;"

show-tables: ## Show tables in backend database
	docker-compose exec -T sqlserver /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P "$${DW_DB_PASS:?set in .env}" -d SOTI_AnomalyDetection -Q "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE';"

# ============================================================================
# Ollama (LLM) Commands
# ============================================================================

logs-ollama: ## Show Ollama logs
	docker-compose logs -f ollama

pull-model: ## Pull llama3.2 model for Ollama
	@echo "Pulling llama3.2 model (this may take a while)..."
	docker-compose exec ollama ollama pull llama3.2

list-models: ## List available Ollama models
	docker-compose exec ollama ollama list

test-ollama: ## Test Ollama service health
	@echo "Testing Ollama service..."
	@curl -sf http://localhost:11434/api/tags > /dev/null && echo "Ollama is healthy!" || echo "Ollama is not responding"

run-ollama: ## Run a test prompt through Ollama
	@echo "Testing Ollama with a simple prompt..."
	curl -X POST http://localhost:11434/api/generate -d '{"model": "llama3.2", "prompt": "Say hello in one sentence.", "stream": false}' | python3 -c "import sys, json; print(json.load(sys.stdin).get('response', 'No response'))"

# ============================================================================
# Qdrant (Vector DB) Commands
# ============================================================================

logs-qdrant: ## Show Qdrant logs
	docker-compose logs -f qdrant

test-qdrant: ## Test Qdrant service health
	@echo "Testing Qdrant service..."
	@curl -sf http://localhost:6333/health > /dev/null && echo "Qdrant is healthy!" || echo "Qdrant is not responding"

qdrant-collections: ## List Qdrant collections
	@curl -s http://localhost:6333/collections | python3 -c "import sys, json; data = json.load(sys.stdin); print('Collections:', [c['name'] for c in data.get('result', {}).get('collections', [])] or 'None')"

qdrant-info: ## Get Qdrant cluster info
	curl -s http://localhost:6333/cluster | python3 -m json.tool

# ============================================================================
# API Commands
# ============================================================================

run-api-docker: ## Run FastAPI server in Docker container
	docker-compose run --rm -p 8000:8000 app uvicorn device_anomaly.api.main:app --host 0.0.0.0 --port 8000 --reload

test-api: ## Test API health endpoint
	@echo "Testing API health..."
	@curl -sf http://localhost:8000/health && echo "" || echo "API is not responding"

api-docs: ## Open API documentation in browser
	@echo "Opening API docs at http://localhost:8000/docs"
	@open http://localhost:8000/docs 2>/dev/null || xdg-open http://localhost:8000/docs 2>/dev/null || echo "Open http://localhost:8000/docs in your browser"

# ============================================================================
# Full Test Suite
# ============================================================================

test-all: ## Run all service tests
	@echo "============================================"
	@echo "Running full test suite..."
	@echo "============================================"
	@echo ""
	@echo "1. Testing PostgreSQL..."
	@docker-compose exec -T postgres pg_isready -U postgres > /dev/null && echo "   ✓ PostgreSQL is healthy" || echo "   ✗ PostgreSQL failed"
	@echo ""
	@echo "2. Testing Qdrant..."
	@curl -sf http://localhost:6333/health > /dev/null && echo "   ✓ Qdrant is healthy" || echo "   ✗ Qdrant failed"
	@echo ""
	@echo "3. Testing API..."
	@curl -sf http://localhost:8000/api/dashboard/stats > /dev/null && echo "   ✓ API is healthy" || echo "   ✗ API failed"
	@echo ""
	@echo "4. Testing Frontend..."
	@curl -sf http://localhost:$(FRONTEND_PORT)/health > /dev/null && echo "   ✓ Frontend is healthy" || echo "   ✗ Frontend failed"
	@echo ""
	@echo "============================================"
	@echo "Test suite complete!"
	@echo "============================================"

status: ## Show status of all services
	@echo "Service Status:"
	@echo "============================================"
	docker-compose ps
	@echo ""
	@echo "Port Mappings:"
	@echo "  SQL Server: localhost:1433"
	@echo "  Ollama:     localhost:11434"
	@echo "  Qdrant:     localhost:6333"
	@echo "  API:        localhost:8000"
	@echo "  Frontend:   localhost:$(FRONTEND_PORT)"

# ============================================================================
# Distribution Build
# ============================================================================

dist: ## Build distribution package for deployment
	@echo "Building distribution package..."
	@./scripts/build_dist.sh

dist-clean: ## Clean distribution build artifacts
	rm -rf dist/
