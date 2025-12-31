# 🚀 Immediate Next Steps

**Goal**: Complete Fast MVP infrastructure so you can start testing end-to-end.

---

## ⚡ Quick Wins (Do These First)

### 1. Start Docker and Verify Services (5 minutes)

```bash
# Start Docker Desktop (if on Mac)
open -a Docker

# Wait for Docker to start, then:
cd /Users/yannickweijenberg/Documents/GitHub/AnomalyDetection

# Build and start all services
make up

# Expected output: 4 containers starting (sqlserver, ollama, qdrant, app)
# Wait ~60 seconds for SQL Server health check

# Verify all services are running
docker-compose ps

# Should show:
# - device-anomaly-db (healthy)
# - device-anomaly-ollama (healthy)
# - device-anomaly-qdrant (healthy)
# - device-anomaly-app (running)
```

### 2. Test Synthetic Experiment (2 minutes)

```bash
# Run the synthetic anomaly detection pipeline
make test-synthetic

# Expected: Logs showing:
# - "Synthetic raw data shape: (840, 11)"
# - "Feature data shape: (840, 35)"
# - "Evaluation: TP=X, FP=Y, FN=Z"
# - "Top 20 most anomalous rows computed: 20"
```

### 3. Test API (1 minute)

```bash
# Test health endpoint
curl http://localhost:8000/health
# Expected: {"status":"healthy"}

# Test root endpoint
curl http://localhost:8000/
# Expected: {"message":"Device Anomaly Detection API","version":"0.1.0"}

# View API docs
open http://localhost:8000/docs
```

**If all 3 succeed** ✅ → Your infrastructure is working! Proceed to development tasks below.

**If something fails** ❌ → Check logs:
```bash
make logs           # All services
make logs-db        # SQL Server only
make logs-app       # Application only
make logs-ollama    # Ollama only
make logs-qdrant    # Qdrant only
```

---

## 🛠️ Development Tasks (After Infrastructure Works)

### Task 1: Create Backend Database Schema (2-3 hours)

**Why**: We need persistent storage for anomalies, baselines, and user data (not just in-memory DataFrames).

**What to create**:
```bash
# 1. Create database module
mkdir -p src/device_anomaly/db/repositories

# 2. Create empty files
touch src/device_anomaly/db/__init__.py
touch src/device_anomaly/db/models.py
touch src/device_anomaly/db/repositories/__init__.py
touch src/device_anomaly/db/repositories/anomaly_repo.py
touch src/device_anomaly/db/repositories/baseline_repo.py
```

**In `src/device_anomaly/db/models.py`**, create these SQLAlchemy models:

```python
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Tenant(Base):
    __tablename__ = 'tenants'
    tenant_id = Column(String(50), primary_key=True)
    name = Column(String(255), nullable=False)
    tier = Column(String(20), default='standard')
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)  # NVARCHAR(MAX) in SQL Server

class Device(Base):
    __tablename__ = 'devices'
    device_id = Column(String(50), primary_key=True)
    tenant_id = Column(String(50), ForeignKey('tenants.tenant_id'), nullable=False)
    source = Column(String(20), nullable=False)  # 'xsight' or 'mobicontrol'
    external_id = Column(String(100), nullable=False)
    name = Column(String(255))
    device_type = Column(String(50))
    os_version = Column(String(50))
    last_seen = Column(DateTime)
    metadata = Column(JSON)

    __table_args__ = (
        Index('idx_tenant_source', 'tenant_id', 'source'),
        Index('idx_external', 'external_id', 'source'),
    )

class MetricDefinition(Base):
    __tablename__ = 'metric_definitions'
    metric_id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    category = Column(String(50))
    unit = Column(String(20))
    data_type = Column(String(20))
    source = Column(String(50))
    is_standard = Column(Integer, default=0)  # BIT in SQL Server
    validation_rules = Column(JSON)

    __table_args__ = (
        Index('idx_source', 'source'),
        Index('idx_category', 'category'),
    )

class Anomaly(Base):
    __tablename__ = 'anomalies'
    anomaly_id = Column(String(50), primary_key=True)
    tenant_id = Column(String(50), ForeignKey('tenants.tenant_id'), nullable=False)
    device_id = Column(String(50), ForeignKey('devices.device_id'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    detector_name = Column(String(100), nullable=False)
    severity = Column(String(20), nullable=False)
    score = Column(Float, nullable=False)
    metrics_involved = Column(JSON)
    explanation = Column(String)  # NVARCHAR(MAX)
    explanation_cache_key = Column(String(100))
    user_feedback = Column(String(20))
    status = Column(String(20), default='new')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_tenant_time', 'tenant_id', 'timestamp'),
        Index('idx_device_time', 'device_id', 'timestamp'),
        Index('idx_severity', 'severity', 'timestamp'),
        Index('idx_status', 'status', 'timestamp'),
    )

# Add more models as needed (Baseline, ChangeLog, Explanation, etc.)
```

**Then create migration script** `scripts/init_backend_schema.sql`:
```sql
USE SOTI_AnomalyDetection;
GO

-- Drop if exists (for development)
DROP TABLE IF EXISTS anomalies;
DROP TABLE IF EXISTS metric_definitions;
DROP TABLE IF EXISTS devices;
DROP TABLE IF EXISTS tenants;

-- Create tables (run the DDL from TRANSFORMATION_STATUS.md)
-- ...
```

**Test**:
```bash
# Initialize backend database
docker-compose exec sqlserver /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P "DevAnomalyStrong!Pass123" -Q "CREATE DATABASE SOTI_AnomalyDetection;"

# Run schema creation
docker-compose exec sqlserver /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P "DevAnomalyStrong!Pass123" -i /app/scripts/init_backend_schema.sql
```

---

### Task 2: Implement Base Connector Interface (1-2 hours)

**Why**: Pluggable data sources (XSight, MobiControl, synthetic) with consistent API.

**What to do**:
```bash
# 1. Create connector module structure
mkdir -p src/device_anomaly/connectors/xsight
mkdir -p src/device_anomaly/connectors/mobicontrol

# 2. Move existing files
mv src/device_anomaly/data_access/dw_connection.py src/device_anomaly/connectors/xsight/
mv src/device_anomaly/data_access/dw_loader.py src/device_anomaly/connectors/xsight/
mv src/device_anomaly/data_access/synthetic_generator.py src/device_anomaly/connectors/
```

**Create `src/device_anomaly/connectors/base.py`**:
```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime

class BaseConnector(ABC):
    """Base interface for data connectors"""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to data source"""
        pass

    @abstractmethod
    def load_telemetry(
        self,
        start_date: datetime,
        end_date: datetime,
        device_ids: Optional[list[int]] = None,
        tenant_id: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load telemetry data in canonical format

        Returns DataFrame with columns:
        - DeviceId, Timestamp, <metric_columns>, tenant_id (if multi-tenant)
        """
        pass

    @abstractmethod
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate DataFrame matches canonical schema"""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return connector metadata (source, version, etc.)"""
        pass
```

**Create `src/device_anomaly/connectors/registry.py`**:
```python
from typing import Dict, Type
from .base import BaseConnector

class ConnectorRegistry:
    """Registry for data connectors"""
    _connectors: Dict[str, Type[BaseConnector]] = {}

    @classmethod
    def register(cls, name: str, connector_class: Type[BaseConnector]):
        cls._connectors[name] = connector_class

    @classmethod
    def get(cls, name: str) -> Type[BaseConnector]:
        if name not in cls._connectors:
            raise ValueError(f"Connector '{name}' not registered")
        return cls._connectors[name]

    @classmethod
    def list_connectors(cls) -> list[str]:
        return list(cls._connectors.keys())
```

**Refactor XSight loader** to implement `BaseConnector`:
```python
# In src/device_anomaly/connectors/xsight/dw_loader.py
from device_anomaly.connectors.base import BaseConnector

class XSightConnector(BaseConnector):
    def connect(self) -> None:
        self.engine = create_dw_engine()

    def load_telemetry(self, start_date, end_date, device_ids=None, tenant_id=None, **kwargs):
        # Use existing load_device_hourly_telemetry logic
        pass

    # ... implement other methods
```

**Register connectors**:
```python
# In src/device_anomaly/connectors/__init__.py
from .registry import ConnectorRegistry
from .xsight.dw_loader import XSightConnector
from .synthetic_generator import SyntheticConnector  # Create this wrapper

ConnectorRegistry.register('xsight_dw', XSightConnector)
ConnectorRegistry.register('synthetic', SyntheticConnector)
```

---

### Task 3: Implement Base Anomaly Detector Interface (1-2 hours)

**Why**: Support multiple detection algorithms (Isolation Forest, Z-Score, Seasonal, Ensemble).

**What to do**:
```bash
# 1. Create base detector
touch src/device_anomaly/models/base.py

# 2. Rename existing detector
mv src/device_anomaly/models/anomaly_detector.py src/device_anomaly/models/isolation_forest.py
```

**Create `src/device_anomaly/models/base.py`**:
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

@dataclass
class DetectorConfig:
    """Base configuration for anomaly detectors"""
    name: str
    enabled: bool = True
    contamination: float = 0.03
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseAnomalyDetector(ABC):
    """Base interface for anomaly detectors"""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.is_fitted = False

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Train detector on normal data"""
        pass

    @abstractmethod
    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores (lower = more anomalous)"""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary predictions (1=normal, -1=anomaly)"""
        pass

    @abstractmethod
    def explain(self, df: pd.DataFrame, indices: list[int]) -> Dict[str, Any]:
        """Provide explanation for flagged anomalies"""
        pass
```

**Refactor Isolation Forest**:
```python
# In src/device_anomaly/models/isolation_forest.py
from device_anomaly.models.base import BaseAnomalyDetector, DetectorConfig

class AnomalyDetectorIsolationForest(BaseAnomalyDetector):
    def __init__(self, config: Optional[DetectorConfig] = None):
        if config is None:
            config = DetectorConfig(name='isolation_forest')
        super().__init__(config)
        # ... rest of implementation
```

---

### Task 4: Update Makefile (30 minutes)

**Add these commands** to `Makefile`:

```makefile
# Ollama commands
pull-ollama-model: ## Pull llama3.2 model for Ollama
	docker-compose exec ollama ollama pull llama3.2

list-ollama-models: ## List available Ollama models
	docker-compose exec ollama ollama list

logs-ollama: ## Show Ollama logs
	docker-compose logs -f ollama

test-ollama: ## Test Ollama service
	@echo "Testing Ollama connection..."
	@docker-compose exec ollama ollama list || echo "Ollama not ready"

# Qdrant commands
logs-qdrant: ## Show Qdrant logs
	docker-compose logs -f qdrant

test-qdrant: ## Test Qdrant service
	@echo "Testing Qdrant health..."
	@curl -f http://localhost:6333/health || echo "Qdrant not ready"

qdrant-info: ## Get Qdrant collections info
	@curl http://localhost:6333/collections

# API commands
run-api: ## Run FastAPI server
	docker-compose run --rm -p 8000:8000 app uvicorn device_anomaly.api.main:app --host 0.0.0.0 --port 8000 --reload

test-api: ## Test API endpoints
	@echo "Testing API health..."
	@curl -f http://localhost:8000/health || echo "API not ready"
	@echo "\nTesting API root..."
	@curl http://localhost:8000/

# Database commands
init-backend-db: ## Initialize backend database schema
	docker-compose exec sqlserver /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P "$${DW_DB_PASS}" -Q "IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = 'SOTI_AnomalyDetection') CREATE DATABASE SOTI_AnomalyDetection;"
	@echo "Backend database created. Run schema script next."

# Full test suite
test-all: test-synthetic test-api test-qdrant test-ollama ## Run all tests
```

---

## 📋 Task Checklist

Use this to track your progress:

### Infrastructure ✅
- [x] Docker Compose with 4 services
- [x] Environment configuration (.env)
- [x] Makefile commands
- [x] FastAPI skeleton

### To Do Next 🔲
- [ ] Start Docker and verify services (5 min)
- [ ] Run `make test-synthetic` successfully
- [ ] Test API health endpoint
- [ ] Create backend database schema (2-3 hrs)
- [ ] Implement base connector interface (1-2 hrs)
- [ ] Implement base detector interface (1-2 hrs)
- [ ] Update Makefile with new commands (30 min)
- [ ] Pull Ollama model: `make pull-ollama-model`
- [ ] Full integration test: `make test-all`

---

## 🐛 Common Issues & Solutions

### Docker Not Starting
```bash
# Mac: Open Docker Desktop app
open -a Docker

# Check if running
docker --version
docker-compose --version
```

### SQL Server Won't Start
```bash
# Check logs
make logs-db

# Common issue: Password complexity
# Must have: uppercase, lowercase, numbers, special chars, min 8 chars
# Fix in .env: DW_DB_PASS=DevAnomalyStrong!Pass123
```

### Ollama Health Check Fails
```bash
# Ollama needs curl installed in container
# If health check fails but container runs, that's OK
# Test manually:
docker-compose exec ollama ollama list
```

### Port Already in Use
```bash
# Check what's using ports
lsof -i :1433  # SQL Server
lsof -i :8000  # API
lsof -i :11434 # Ollama
lsof -i :6333  # Qdrant

# Kill process or change port in .env
```

---

## 📚 Reference Documents

1. **Transformation Plan**: `.claude/plans/scalable-leaping-hinton.md` - Full implementation plan
2. **Status Tracking**: `TRANSFORMATION_STATUS.md` - What's done, what's next
3. **README**: `../README.md` - Docker setup guide
4. **Skills**: `Skills.md` - Multi-agent architecture

---

## 🎯 Success Criteria

You'll know you're ready to move to Phase 1 when:

1. ✅ All 4 Docker services start and show healthy
2. ✅ `make test-synthetic` completes successfully
3. ✅ `curl http://localhost:8000/health` returns `{"status":"healthy"}`
4. ✅ Backend database schema is created
5. ✅ Base connector interface exists and XSight implements it
6. ✅ Base detector interface exists and Isolation Forest implements it
7. ✅ `make test-all` passes

**After that**, you'll be ready for Phase 1 (Week 3-4):
- MobiControl connector
- Statistical detectors
- Full REST API with authentication
- Ollama LLM integration
- Qdrant RAG setup

---

**Questions?** Check the plan at `.claude/plans/scalable-leaping-hinton.md` or ask me!
