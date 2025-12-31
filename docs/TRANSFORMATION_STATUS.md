# SOTI Anomaly Detection - Production Transformation Status

**Last Updated**: 2025-12-24
**Phase**: Fast MVP (Week 1-2)
**Priority**: Docker Compose + Basic Infrastructure

---

## ✅ Completed (Phase 0 - Infrastructure)

### 1. Docker Infrastructure
- ✅ **Dockerfile** - Python 3.11, ODBC Driver 18 for SQL Server
- ✅ **docker-compose.yml** - Multi-service orchestration
  - SQL Server 2022 (with health checks)
  - **Ollama** (local LLM service) - ADDED TODAY
  - **Qdrant** (vector database for RAG) - ADDED TODAY
  - Python application container
- ✅ **Makefile** - 15+ commands for development workflow
- ✅ **Environment Configuration**
  - `env.template` with all required variables
  - `.env` created with secure defaults
  - Added Qdrant and API configuration

### 2. Configuration Management
- ✅ **Pydantic Settings** ([src/device_anomaly/config/settings.py](src/device_anomaly/config/settings.py))
  - `DWSettings` - SQL Server connection
  - `LLMSettings` - LLM configuration
  - `MobiControlSettings` - MobiControl API credentials
  - Validation: prevents default passwords
  - `.env` file loading support

### 3. FastAPI Application
- ✅ **API Skeleton** ([src/device_anomaly/api/main.py](src/device_anomaly/api/main.py))
  - Basic FastAPI app with CORS
  - Health check endpoint (`/health`)
  - Root endpoint with version info
- ✅ **API Routes** (stub implementations exist):
  - `/api/anomalies` - Anomaly management
  - `/api/devices` - Device information
  - `/api/dashboard` - Dashboard data
- ✅ **Dependencies** - FastAPI + Uvicorn in pyproject.toml

### 4. Existing Anomaly Detection Pipeline
- ✅ **Synthetic Data Generator** - Creates realistic telemetry with injected anomalies
- ✅ **Feature Engineering** - Rolling windows (12h), deltas
- ✅ **Isolation Forest Detector** - 300 estimators, 3% contamination
- ✅ **Evaluation Metrics** - Precision, recall on synthetic ground truth
- ✅ **CLI Experiments**:
  - `make test-synthetic` - Works without database
  - `make test-dw` - Requires database with schema

### 5. Documentation
- ✅ **README.md** - Comprehensive Docker setup instructions (in root)
- ✅ **Skills.md** - Multi-agent architecture definition (9 agents)
- ✅ **Transformation Plan** - Detailed implementation plan approved

---

## 🔧 In Progress

### Backend Database Schema (SQLAlchemy Models)
**Status**: Need to create
**Priority**: HIGH
**Files to Create**:
- `src/device_anomaly/db/models.py` - Canonical entities
- `src/device_anomaly/db/repositories/` - Data access layer

**Required Tables** (per plan):
- `tenants` - Multi-tenant isolation
- `devices` - Device registry (XSight + MobiControl)
- `metric_definitions` - Dynamic metric catalog
- `telemetry_points` - Time-series data (partitioned)
- `baselines` - Baseline profiles
- `anomalies` - Detected anomalies
- `change_log` - Environment changes
- `explanations` - LLM-generated explanations

---

## 📋 Next Steps (Ordered by Priority)

### Phase 0 - Complete MVP Infrastructure (This Week)

#### 1. Backend Database Models (NEXT)
**Effort**: 2-3 hours
**Blockers**: None

Create SQLAlchemy models for the unified data model supporting both XSight and MobiControl:

```bash
# Create files
touch src/device_anomaly/db/__init__.py
touch src/device_anomaly/db/models.py
mkdir -p src/device_anomaly/db/repositories
touch src/device_anomaly/db/repositories/__init__.py
touch src/device_anomaly/db/repositories/anomaly_repo.py
```

**Key Models**:
- `Tenant` - tenant_id (PK), name, tier, created_at, metadata (JSON)
- `Device` - device_id (PK), tenant_id (FK), source ('xsight'|'mobicontrol'), external_id
- `MetricDefinition` - metric_id (PK), name, category, unit, data_type, is_standard
- `TelemetryPoint` - id (PK), device_id (FK), timestamp, metric_id (FK), value
- `Anomaly` - anomaly_id (PK), tenant_id (FK), device_id (FK), timestamp, detector_name, severity, score
- `Baseline` - baseline_id (PK), tenant_id (FK), scope, stats (JSON), valid_from/to

#### 2. Base Connector Interface
**Effort**: 1-2 hours
**Files**:
- `src/device_anomaly/connectors/base.py` - Abstract base class
- `src/device_anomaly/connectors/registry.py` - Connector factory

**Actions**:
- Define `BaseConnector` ABC with `connect()`, `load_telemetry()`, `validate_schema()`
- Implement `ConnectorRegistry` for dynamic connector loading
- Refactor existing `dw_loader.py` to implement `BaseConnector`

#### 3. Base Anomaly Detector Interface
**Effort**: 1-2 hours
**Files**:
- `src/device_anomaly/models/base.py` - Abstract detector interface
- `src/device_anomaly/models/isolation_forest.py` - Rename from `anomaly_detector.py`

**Actions**:
- Define `BaseAnomalyDetector` ABC with `fit()`, `score()`, `predict()`, `explain()`
- Refactor `AnomalyDetectorIsolationForest` to extend base class
- Add `DetectorConfig` dataclass for configuration

#### 4. Update Makefile for New Services
**Effort**: 30 minutes
**Add Commands**:
```makefile
pull-ollama-model: ## Pull llama3.2 model for Ollama
	docker-compose exec ollama ollama pull llama3.2

logs-ollama: ## Show Ollama logs
	docker-compose logs -f ollama

logs-qdrant: ## Show Qdrant logs
	docker-compose logs -f qdrant

test-ollama: ## Test Ollama connection and model
	docker-compose exec ollama ollama list

test-qdrant: ## Test Qdrant health
	curl http://localhost:6333/health
```

#### 5. Integration Test with Docker
**Effort**: 1 hour (when Docker is running)
**Test Sequence**:
```bash
# 1. Start all services
make up

# 2. Verify all services healthy
docker-compose ps

# 3. Run synthetic experiment
make test-synthetic

# 4. Test API health
curl http://localhost:8000/health

# 5. Pull LLM model (optional)
make pull-ollama-model

# 6. Test Qdrant
make test-qdrant
```

---

## 🚀 Phase 1 - Stabilize (Week 3-4)

### Connector Expansion
- MobiControl DB connector (similar to XSight)
- MobiControl REST API client (with retry, rate limiting)
- Connector factory and registry pattern
- Unit tests for each connector

### Enhanced Anomaly Detection
- Z-Score detector (statistical)
- Seasonal detector (STL decomposition)
- Ensemble detector (multi-detector voting)
- Baseline builder pipeline

### REST API Implementation
- `/api/v1/anomalies` - List, get, provide feedback
- `/api/v1/baselines` - CRUD for baseline profiles
- `/api/v1/metrics` - Catalog of all metrics (standard + custom)
- `/api/v1/devices/{id}/telemetry` - Raw time-series data
- Authentication middleware (JWT)

---

## 📊 Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Services                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ SQL Server │  │   Ollama   │  │   Qdrant   │            │
│  │   (DW +    │  │  (Local    │  │  (Vector   │            │
│  │  Backend)  │  │   LLM)     │  │    DB)     │            │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘            │
│        │               │               │                     │
│        └───────────────┼───────────────┘                     │
│                        │                                      │
│              ┌─────────▼──────────┐                          │
│              │   Python App       │                          │
│              │  - CLI experiments │                          │
│              │  - FastAPI server  │                          │
│              │  - Anomaly engine  │                          │
│              └────────────────────┘                          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🗂️ File Structure (Current)

```
AnomalyDetection/
├── .env                          ✅ Created with secure defaults
├── env.template                  ✅ Updated with Qdrant + API vars
├── docker-compose.yml            ✅ 4 services (SQL, Ollama, Qdrant, App)
├── Dockerfile                    ✅ Python 3.11 + ODBC Driver 18
├── Makefile                      ✅ 15+ commands
├── pyproject.toml                ✅ All dependencies including FastAPI
├── README.md                     ✅ Docker setup guide (in root)
├── docs/Skills.md                ✅ 9-agent architecture
│
├── src/device_anomaly/
│   ├── api/                      ✅ FastAPI skeleton
│   │   ├── main.py               ✅ App with /health endpoint
│   │   ├── routes/               ✅ Anomalies, devices, dashboard
│   │   └── models/               📁 Created (empty)
│   │
│   ├── cli/                      ✅ CLI experiments
│   │   ├── main.py               ✅ Main entry point
│   │   ├── synthetic_experiment.py  ✅ Works end-to-end
│   │   └── dw_experiment.py      ✅ Requires DB
│   │
│   ├── config/                   ✅ Settings management
│   │   ├── settings.py           ✅ Pydantic + MobiControl
│   │   └── logging_config.py     ✅ Basic logging
│   │
│   ├── data_access/              ✅ Data loading (to be refactored → connectors/)
│   │   ├── dw_connection.py      ✅ SQL Server engine
│   │   ├── dw_loader.py          ✅ XSight query builder
│   │   └── synthetic_generator.py ✅ Test data generator
│   │
│   ├── features/                 ✅ Feature engineering
│   │   └── device_features.py    ✅ Rolling windows + deltas
│   │
│   ├── models/                   ✅ Anomaly detection
│   │   └── anomaly_detector.py   ✅ Isolation Forest wrapper
│   │
│   └── db/                       ❌ TO CREATE - Backend schema
│
├── scripts/                      ✅ Helper scripts
│   ├── init_db.sh                ✅ DB initialization
│   ├── smoke_test.sh             ✅ Smoke tests
│   └── test_llm_connection.sh    ✅ LLM connectivity test
│
└── tests/                        ⚠️  Minimal (only test_imports.py)
```

---

## 🎯 Quick Start (When Docker is Running)

```bash
# 1. Ensure .env is configured
cat .env

# 2. Start all services
make up

# 3. Verify services are healthy
make logs

# 4. Run synthetic experiment (no DB required)
make test-synthetic

# 5. Test API
curl http://localhost:8000/health
curl http://localhost:8000/

# 6. Optional: Pull Ollama model
docker-compose exec ollama ollama pull llama3.2

# 7. Verify Qdrant
curl http://localhost:6333/health

# 8. Open shell in app container
make shell
```

---

## 📝 Notes

### Security
- ✅ `.env` file is gitignored
- ✅ No default passwords allowed (validation in settings.py)
- ✅ SQL Server password meets complexity requirements
- ⚠️  MobiControl credentials in env.template (should be redacted for commits)

### Performance
- SQL Server uses persistent volume (data survives restarts)
- Ollama models stored in volume (downloaded once)
- Qdrant data persisted in volume

### Dependencies
All Python dependencies are defined in [pyproject.toml](pyproject.toml):
- Core: pandas, numpy, scikit-learn
- Database: SQLAlchemy, pyodbc
- API: FastAPI, uvicorn
- LLM: (to be added: qdrant-client, sentence-transformers, anthropic/openai)

---

## 🔗 Key References

- **Plan**: `/Users/yannickweijenberg/.claude/plans/scalable-leaping-hinton.md`
- **generative_ai_project**: https://github.com/HeyNina101/generative_ai_project
- **README**: [README.md](../README.md)
- **Skills**: [Skills.md](Skills.md)

---

## ✅ Success Criteria for Phase 0

- [x] Docker Compose with 4 services (SQL Server, Ollama, Qdrant, App)
- [x] Environment configuration secured
- [x] Synthetic experiment runs successfully
- [ ] Backend database schema defined (SQLAlchemy models)
- [ ] Base connector interface implemented
- [ ] Base detector interface implemented
- [ ] Full integration test passes
- [ ] API serves /health and basic /anomalies stub

**Current Status**: 4/8 complete (50%)
**Estimated completion**: When Docker is available for testing
