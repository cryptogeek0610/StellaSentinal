# Smoke Test Checklist

This document describes the smoke tests to validate the local setup is working correctly.

## Prerequisites

- Docker and Docker Compose installed
- Services started: `make up` or `docker-compose up -d`
- `.env` file configured with your settings

## Automated Smoke Tests

Run the automated smoke test script:

```bash
chmod +x scripts/smoke_test.sh
./scripts/smoke_test.sh
```

## Manual Smoke Test Checklist

### 1. Container Health

- [ ] PostgreSQL container is running
  ```bash
  docker-compose ps postgres
  # Should show "Up" status
  ```

- [ ] Redis container is running
  ```bash
  docker-compose ps redis
  # Should show "Up" status
  ```

- [ ] App container exists (may be stopped if not running a command)
  ```bash
  docker-compose ps app
  ```

### 2. Database Connectivity

- [ ] PostgreSQL accepts connections
  ```bash
  docker-compose exec postgres pg_isready -U postgres
  # Expected: "accepting connections"
  ```

- [ ] External SQL Server connectivity (if configured)
  ```bash
  # This tests the external XSight/MobiControl SQL Server
  docker-compose exec app python -c \
    "from device_anomaly.data_access.db_connection import create_dw_engine; \
     e = create_dw_engine(); \
     print('DW connection OK')"
  ```

### 3. Python Environment

- [ ] Python version is 3.10+
  ```bash
  docker-compose exec app python --version
  # Expected: Python 3.11.x or higher
  ```

- [ ] All required packages are installed
  ```bash
  docker-compose exec app python -c \
    "import pandas, numpy, sklearn, sqlalchemy; print('OK')"
  # Expected: "OK"
  ```

- [ ] Application package is importable
  ```bash
  docker-compose exec app python -c \
    "from device_anomaly.config.settings import get_settings; print('OK')"
  # Expected: "OK"
  ```

### 4. Application Entry Points

- [ ] Main entry point runs without errors
  ```bash
  docker-compose run --rm app python -m device_anomaly.cli.main
  # Expected: Logs "Device Anomaly Service starting up..."
  ```

- [ ] Synthetic experiment runs successfully
  ```bash
  docker-compose run --rm app python -m device_anomaly.cli.synthetic_experiment
  # Expected:
  # - Logs "Running synthetic experiment"
  # - Shows synthetic data shape and feature engineering
  # - Shows anomaly detection results
  # - Shows evaluation metrics (precision, recall)
  ```

### 5. Configuration

- [ ] Settings load from environment variables
  ```bash
  docker-compose exec app python -c \
    "from device_anomaly.config.settings import get_settings; \
     s = get_settings(); \
     print(f'DW Host: {s.dw.host}, Backend DB: {s.backend_db.host}')"
  # Expected: Shows configured hosts
  ```

### 6. Expected Outputs

#### Synthetic Experiment Output

When running `make test-synthetic`, you should see:

1. **Data Generation:**
   ```
   Running synthetic experiment: n_devices=5, n_days=7, window=12, anomaly_rate=0.030
   Synthetic raw data shape: (840, X)
   ```

2. **Feature Engineering:**
   ```
   Feature data shape: (828, Y)
   ```

3. **Anomaly Detection:**
   ```
   Evaluation on synthetic ground truth:
     TP=X, FP=Y, FN=Z
     Precision=0.XXX, Recall=0.XXX
   ```

4. **Top Anomalies Summary:**
   - Logs how many top anomalies were computed

#### Main Entry Point Output

When running `make test-main`, you should see:
```
Device Anomaly Service starting up...
Version: 0.1.0
No experiments defined yet. This is just a skeleton.
```

## Common Issues and Solutions

### Issue: PostgreSQL container keeps restarting

**Solution:**
- Check logs: `docker-compose logs postgres`
- Ensure volume permissions are correct

### Issue: "Cannot connect to external SQL Server"

**Solution:**
- Verify `DW_DB_HOST`, `DW_DB_USER`, `DW_DB_PASS` in `.env`
- Ensure the external SQL Server is reachable from Docker (check firewall, network)
- For Docker Desktop, use `host.docker.internal` to reach host machine

### Issue: Import errors for device_anomaly package

**Solution:**
- Rebuild image: `docker-compose build`
- Check PYTHONPATH is set correctly in Dockerfile
- Verify source code is mounted: `docker-compose exec app ls -la /app/src`

### Issue: DW experiment returns empty data

**Expected behavior if:**
- Database exists but has no tables
- Tables exist but have no data in the date range
- This is normal for a fresh database setup

**Solution:**
- Create tables and seed data manually
- Or use synthetic experiment which doesn't require DB

## Success Criteria

All smoke tests pass when:
- ✅ All containers start successfully
- ✅ PostgreSQL accepts connections
- ✅ Python environment has all dependencies
- ✅ Main entry point runs
- ✅ Synthetic experiment completes with results
- ✅ Configuration loads from environment
