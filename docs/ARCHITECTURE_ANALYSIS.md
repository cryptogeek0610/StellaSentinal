# Device Anomaly Detection - Architecture Analysis

## 1. What This Repo Does (One-Page Summary)

**Purpose**: A Python-based anomaly detection system for IoT device telemetry that identifies unusual patterns in device behavior using machine learning (Isolation Forest) and prepares for future LLM-based explanation generation.

**Current State**: Phase 1-2 implementation (synthetic data testing + real DW data integration). Phase 3 (LLM explanations) is planned but not implemented.

**Execution Modes**:
- **CLI batch mode**: Two experiment scripts (`synthetic_experiment.py`, `dw_experiment.py`) that run end-to-end pipelines
- **No API/streaming**: Pure batch processing, no real-time capabilities

**Core Flow**:
1. **Data Loading**: Either synthetic generation or SQL Server DW query
2. **Feature Engineering**: Rolling statistics (mean/std) and deltas over time windows per device
3. **Anomaly Detection**: Isolation Forest (unsupervised ML) trained on feature vectors
4. **Scoring**: All data points scored, top anomalies logged to console
5. **Output**: Console logs only (no persistence, no API, no files)

**Key Assumptions**:
- Hourly telemetry data with 8 metrics (battery, storage, network, WiFi, connectivity)
- Devices identified by `DeviceId` integer
- Time-series data sorted by device and timestamp
- Anomaly rate ~3% (configurable via `contamination` parameter)

---

## 2. Detailed Architecture Diagram (Text Form)

```
┌─────────────────────────────────────────────────────────────────┐
│                         ENTRY POINTS                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────┐      ┌──────────────────────┐         │
│  │ synthetic_experiment  │      │   dw_experiment      │         │
│  │    .py (main)         │      │    .py (main)        │         │
│  └──────────┬───────────┘      └──────────┬───────────┘         │
│             │                             │                      │
│             │                             │                      │
└─────────────┼─────────────────────────────┼──────────────────────┘
              │                             │
              ▼                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA ACCESS LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────┐      ┌──────────────────────┐         │
│  │ synthetic_generator  │      │     dw_loader        │         │
│  │  .generate_*()       │      │  .load_device_*()    │         │
│  │                      │      │                      │         │
│  │ - Generates hourly   │      │ - SQL query to DW    │         │
│  │   telemetry with     │      │ - Joins 5 tables     │         │
│  │   injected anomalies  │      │ - Returns DataFrame  │         │
│  │ - 8 metrics + flag   │      │ - Uses SQLAlchemy    │         │
│  └──────────┬───────────┘      └──────────┬───────────┘         │
│             │                             │                      │
│             │                             │                      │
│             └─────────────┬───────────────┘                      │
│                           │                                      │
│                           ▼                                      │
│              ┌────────────────────────┐                         │
│              │   DataFrame (pandas)   │                         │
│              │   Columns:             │                         │
│              │   - DeviceId           │                         │
│              │   - Timestamp          │                         │
│              │   - 8 metric columns   │                         │
│              └──────────┬─────────────┘                         │
└─────────────────────────┼───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         DeviceFeatureBuilder.transform()                  │   │
│  │                                                           │   │
│  │  Per DeviceId group:                                      │   │
│  │  1. Sort by Timestamp                                     │   │
│  │  2. Rolling window (default 12 hours)                     │   │
│  │     - mean_{window} for each metric                       │   │
│  │     - std_{window} for each metric                        │   │
│  │  3. First-order delta (diff with previous row)            │   │
│  │                                                           │   │
│  │  Output: DataFrame with ~24 feature columns               │   │
│  │  (8 metrics × 3 features each = 24)                       │   │
│  └────────────────────┬──────────────────────────────────────┘   │
└───────────────────────┼──────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ANOMALY DETECTION                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │      AnomalyDetectorIsolationForest                       │   │
│  │                                                           │   │
│  │  1. fit(df_features)                                     │   │
│  │     - Auto-selects columns with "_mean_", "_std_",       │   │
│  │       "_delta" in name                                    │   │
│  │     - Trains sklearn IsolationForest                      │   │
│  │                                                           │   │
│  │  2. score_dataframe(df_features)                        │   │
│  │     - decision_function() → anomaly_score                │   │
│  │     - predict() → anomaly_label (-1 or 1)                │   │
│  │     - Returns DataFrame with scores appended              │   │
│  └────────────────────┬──────────────────────────────────────┘   │
└───────────────────────┼──────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Console Logging Only                                    │   │
│  │                                                           │   │
│  │  - Top N anomalies (sorted by score)                     │   │
│  │  - For synthetic: precision/recall vs ground truth        │   │
│  │  - No file output, no DB writes, no API                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────┐      ┌──────────────────────┐         │
│  │   settings.py        │      │  logging_config.py   │         │
│  │                      │      │                      │         │
│  │  - DWSettings        │      │  - setup_logging()   │         │
│  │    (env vars)        │      │  - StreamHandler     │         │
│  │  - LLMSettings       │      │  - INFO level        │         │
│  │    (not used yet)    │      │                      │         │
│  └──────────────────────┘      └──────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Module-by-Module Breakdown

### Entry Points (`src/device_anomaly/cli/`)

#### `main.py`
- **Purpose**: Skeleton entrypoint, currently just logs startup message
- **Status**: Not functional, placeholder
- **Assumption**: Intended as future unified CLI, not used

#### `synthetic_experiment.py`
- **Purpose**: End-to-end pipeline on synthetic data
- **Flow**:
  1. Calls `generate_synthetic_device_telemetry()` with defaults (5 devices, 7 days, 3% anomaly rate)
  2. Builds features via `DeviceFeatureBuilder(window=12)`
  3. Trains `AnomalyDetectorIsolationForest(contamination=0.03)`
  4. Scores all data
  5. Evaluates precision/recall against `is_injected_anomaly` flag
  6. Logs top 20 anomalies
- **Hardcoded**: All parameters in `main()` function
- **Output**: Console logs only

#### `dw_experiment.py`
- **Purpose**: Same pipeline but loads real data from SQL Server DW
- **Flow**:
  1. Calls `load_device_hourly_telemetry(start_date, end_date, device_ids, limit)`
  2. Same feature building and detection as synthetic
  3. Logs top 50 anomalies
- **Hardcoded**: Date range `2025-01-01` to `2025-11-20`, `row_limit=50_000`
- **Output**: Console logs only

### Data Access (`src/device_anomaly/data_access/`)

#### `synthetic_generator.py`
- **Function**: `generate_synthetic_device_telemetry()`
- **Schema**: Returns DataFrame with:
  - `DeviceId` (int, 1..n_devices)
  - `Timestamp` (datetime, hourly)
  - 8 metrics: `TotalBatteryLevelDrop`, `TotalFreeStorageKb`, `Download`, `Upload`, `OfflineTime`, `DisconnectCount`, `WiFiSignalStrength`, `ConnectionTime`
  - `is_injected_anomaly` (bool, ground truth)
- **Anomaly Types**: 5 types injected randomly (battery spike, storage low, traffic burst, massive offline, WiFi drop)
- **Patterns**: Diurnal usage patterns (business hours = higher usage)
- **Assumptions**: Hourly frequency, integer metrics, no timezone handling

#### `dw_loader.py`
- **Function**: `load_device_hourly_telemetry()`
- **SQL Query**: Complex CTE joining 5 tables:
  - `Device` (driving table)
  - `cs_DataUsageByHour` (hourly aggregation)
  - `cs_BatteryStat` (LEFT JOIN on date)
  - `cs_Offline` (LEFT JOIN on date + hour)
  - `cs_WifiHour` (LEFT JOIN on date + hour, note: `Deviceid` lowercase)
  - `cs_DataUsageByHour` (duplicate join for Download/Upload)
- **SQL Injection Risk**: `device_ids` parameter uses string interpolation (`f"IN ({in_clause})"`) - **SECURITY ISSUE**
- **Parameters**: Uses SQLAlchemy `text()` with `params` dict for dates (safe), but device_ids are interpolated
- **Schema Assumption**: Matches synthetic schema exactly (8 columns)
- **Time Handling**: Converts `Timestamp` to pandas datetime, no timezone conversion
- **Limit**: Optional `TOP (N)` clause for safety

#### `dw_connection.py`
- **Function**: `create_dw_engine()`
- **Connection String**: `mssql+pyodbc://user:password@host/db?driver=...`
- **Credentials**: Loaded from `settings.dw` (env vars with defaults)
- **Default Credentials**: Hardcoded defaults in `settings.py` (`sa/sa`) - **SECURITY ISSUE**
- **Engine Options**: `fast_executemany=True` for performance

### Feature Engineering (`src/device_anomaly/features/`)

#### `device_features.py`
- **Class**: `DeviceFeatureBuilder`
- **Method**: `transform(df)`
- **Process**:
  1. Validates `Timestamp` column exists
  2. Sorts by `[DeviceId, Timestamp]`
  3. Groups by `DeviceId`
  4. Per device: sets `Timestamp` as index, computes rolling stats, resets index
  5. Returns combined DataFrame
- **Features Generated** (per metric):
  - `{metric}_mean_{window}`: Rolling mean (default 12 hours)
  - `{metric}_std_{window}`: Rolling std (filled with 0 for NaN)
  - `{metric}_delta`: First-order difference (filled with 0)
- **Window**: Configurable, default 12 (hours/rows)
- **Metrics**: Hardcoded list of 8 metrics
- **Assumptions**: Data is pre-sorted, no missing timestamps, no timezone issues

### Models (`src/device_anomaly/models/`)

#### `anomaly_detector.py`
- **Class**: `AnomalyDetectorIsolationForest`
- **Config**: `AnomalyDetectorConfig` (contamination, n_estimators, random_state)
- **Model**: sklearn `IsolationForest`
- **Feature Selection**: Heuristic - selects columns containing `"_mean_"`, `"_std_"`, or `"_delta"`
- **Methods**:
  - `fit(df)`: Trains on selected feature columns
  - `score(df)`: Returns decision function scores (higher = more normal)
  - `predict(df)`: Returns labels (-1 = anomaly, 1 = normal)
  - `score_dataframe(df)`: Combines scores + labels into DataFrame
- **Assumptions**: Feature columns follow naming convention, no missing values in features

### Configuration (`src/device_anomaly/config/`)

#### `settings.py`
- **Classes**: `DWSettings`, `LLMSettings`, `AppSettings`
- **Loading**: Environment variables with hardcoded defaults
- **Default Credentials**: `host="CAD469\\SQLEXPRESS"`, `user="sa"`, `password="sa"` - **SECURITY ISSUE**
- **Singleton Pattern**: Global `_settings` variable, `get_settings()` returns cached instance
- **LLM Settings**: Defined but not used (Phase 3)

#### `logging_config.py`
- **Function**: `setup_logging(level=INFO)`
- **Handler**: `StreamHandler` to stdout
- **Formatter**: Timestamp + level + name + message
- **Idempotency**: Checks for existing handlers to avoid duplicates

---

## 4. Data Flow and Assumptions

### Data Sources

**Synthetic Data**:
- Generated in-memory, no external dependencies
- Schema: 8 metrics + DeviceId + Timestamp + ground truth flag
- Frequency: Hourly (configurable)
- Time Range: Configurable days from fixed start date (`2025-01-01`)

**Real DW Data**:
- SQL Server database (connection via pyodbc)
- Tables: `Device`, `cs_DataUsageByHour`, `cs_BatteryStat`, `cs_Offline`, `cs_WifiHour`
- Schema: Assumed to match synthetic exactly
- Time Range: Query-based (`start_date` to `end_date` strings)
- **Assumption**: All tables have matching `DeviceId` and date/hour keys

### Time Window Handling

- **Window Size**: Configurable integer (default 12), treated as row count (not time-aware)
- **Assumption**: Data is hourly, so window=12 = 12 hours
- **No Timezone Handling**: Timestamps converted to pandas datetime but no TZ conversion
- **No Missing Time Handling**: Rolling window uses `min_periods=1`, so first rows have NaN filled with 0

### Tenancy/Multi-Device Handling

- **Device Isolation**: Features computed per `DeviceId` group
- **Training**: Model trained on all devices together (no per-device models)
- **Assumption**: All devices have similar normal behavior patterns
- **No Device Metadata**: No filtering by device type, location, etc.

### Anomaly Definition

- **Type**: Unsupervised ML (Isolation Forest)
- **Contamination**: Configurable (default 0.03 = 3% expected anomalies)
- **Algorithm**: Isolation Forest (tree-based, works well for high-dimensional data)
- **Output**: 
  - `anomaly_score`: Decision function (higher = more normal, lower = more anomalous)
  - `anomaly_label`: -1 (anomaly) or 1 (normal)
- **No Rule-Based**: Pure statistical/ML approach, no business rules

---

## 5. Output Artifacts

### Current Outputs

1. **Console Logs**:
   - Data shape information
   - Sample rows
   - Top N anomalies (sorted by score, ascending = most anomalous)
   - For synthetic: precision/recall metrics

2. **No Persistence**:
   - No files written
   - No database writes
   - No API responses
   - No dashboards

### Output Schema (in-memory DataFrame)

After scoring, DataFrame contains:
- Original columns (DeviceId, Timestamp, 8 metrics)
- Feature columns (24 columns: 8 metrics × 3 features)
- `anomaly_score` (float, lower = more anomalous)
- `anomaly_label` (int, -1 or 1)
- For synthetic: `is_injected_anomaly` (bool)

---

## 6. Quality and Risk Review

### Security Risks 🔴

1. **SQL Injection in `dw_loader.py` (Line 100)**:
   ```python
   in_clause = ", ".join(str(int(x)) for x in device_ids)
   base_sql += f"\n  AND DeviceId IN ({in_clause})"
   ```
   - **Risk**: Medium (only integers, but still unsafe pattern)
   - **Fix**: Use parameterized query with SQLAlchemy `in_()` clause

2. **Hardcoded Default Credentials in `settings.py` (Lines 22-26)**:
   ```python
   user=os.getenv("DW_DB_USER", "sa"),
   password=os.getenv("DW_DB_PASS", "sa"),
   ```
   - **Risk**: High (defaults to `sa/sa`, exposed in code)
   - **Fix**: Remove defaults, require env vars, fail fast if missing

3. **Connection String in Logs**:
   - **Risk**: Low (connection string not logged, but password in memory)
   - **Fix**: Ensure no logging of connection strings

4. **No Input Validation**:
   - Date strings not validated
   - Device IDs not validated (could be negative, zero, etc.)
   - **Fix**: Add Pydantic validators or explicit checks

### Reliability Risks 🟡

1. **No Error Handling**:
   - Database connection failures not caught
   - SQL query failures not caught
   - Empty DataFrame handling: Only warning logged, continues
   - **Fix**: Add try/except blocks, proper error messages

2. **No Retries**:
   - Database operations have no retry logic
   - Network timeouts not handled
   - **Fix**: Add retry decorator or exponential backoff

3. **No Idempotency**:
   - Running same experiment twice produces same results (good for ML)
   - But no deduplication if results were persisted
   - **Fix**: N/A (not needed for current design)

4. **Memory Risks**:
   - Loads entire dataset into memory (pandas DataFrame)
   - `row_limit=50_000` is a safety valve but not enforced in synthetic
   - **Fix**: Add chunking for large datasets, streaming processing

5. **No Data Validation**:
   - No schema validation on loaded data
   - No check for required columns
   - **Fix**: Add Pydantic models or explicit validation

### Performance Risks 🟡

1. **N+1 Query Pattern**: 
   - SQL query uses multiple LEFT JOINs (acceptable, single query)
   - But feature building groups by DeviceId (in-memory, acceptable)
   - **Status**: OK for current scale

2. **No Pagination**:
   - Loads all data at once
   - `row_limit` exists but is optional
   - **Fix**: Implement chunked processing for large date ranges

3. **Inefficient Feature Building**:
   - Groups by DeviceId, then applies rolling window (acceptable)
   - But copies DataFrame multiple times (`df.copy()` in multiple places)
   - **Fix**: Use views or in-place operations where safe

4. **Model Training**:
   - Trains on entire dataset each run (no incremental learning)
   - Isolation Forest is fast, but could be slow for millions of rows
   - **Fix**: Sample training data, or use pre-trained models

5. **No Caching**:
   - No caching of feature computations
   - No caching of model artifacts
   - **Fix**: Add joblib/pickle caching for features and models

### Test Coverage 🟡

1. **Current Tests**:
   - `tests/test_imports.py`: Only checks package can be imported
   - **Coverage**: ~0%

2. **Missing Tests**:
   - No unit tests for feature builder
   - No unit tests for anomaly detector
   - No integration tests for data loading
   - No tests for synthetic generator
   - No tests for SQL query construction

3. **How to Add Tests**:
   - Add pytest fixtures for synthetic data
   - Mock database connections for DW tests
   - Test feature builder with known inputs/outputs
   - Test anomaly detector with labeled synthetic data
   - Add CI/CD pipeline (GitHub Actions, etc.)

---

## 7. Must Fix First Issues

### Critical (Fix Immediately)

1. **Remove Hardcoded Credentials** (`settings.py`):
   - Remove default `sa/sa` credentials
   - Fail fast if env vars missing
   - Add validation

2. **Fix SQL Injection** (`dw_loader.py`):
   - Use parameterized query for `device_ids`
   - Use SQLAlchemy `in_()` or bind parameters

3. **Add Error Handling**:
   - Wrap database operations in try/except
   - Add meaningful error messages
   - Handle empty DataFrame gracefully (fail or skip)

### High Priority (Fix Soon)

4. **Add Input Validation**:
   - Validate date strings format
   - Validate device IDs (positive integers)
   - Validate contamination parameter (0 < x < 1)

5. **Add Data Schema Validation**:
   - Check required columns exist after loading
   - Validate data types
   - Check for nulls in critical columns

6. **Add Logging for Errors**:
   - Log database connection failures
   - Log query execution errors
   - Log model training failures

### Quick Wins (Low Effort, High Value)

7. **Add Command-Line Arguments**:
   - Use `argparse` or `click` for experiment scripts
   - Remove hardcoded parameters

8. **Add Configuration File**:
   - YAML/TOML config for experiment parameters
   - Separate configs for dev/prod

9. **Add Basic Tests**:
   - Test feature builder with known data
   - Test anomaly detector with synthetic labeled data
   - Test synthetic generator output schema

10. **Add Output Persistence**:
    - Write results to CSV/Parquet
    - Or write to database table
    - Add `--output-dir` parameter

---

## 8. Open Questions to Validate with Repo Owner

### Data & Schema

1. **DW Schema Accuracy**: Does the SQL query in `dw_loader.py` match the actual DW schema? (Especially `cs_WifiHour.Deviceid` lowercase)

2. **Time Zone**: What timezone are DW timestamps in? Should we convert to UTC?

3. **Data Completeness**: Are there missing hours in the DW data? How should gaps be handled?

4. **Device Filtering**: Should we filter by device type, location, or other metadata before training?

5. **Historical Data**: How far back should we look for training? Is there a concept of "recent" vs "historical" normal behavior?

### Anomaly Detection

6. **Per-Device Models**: Should we train separate models per device type, or is a global model acceptable?

7. **Contamination Parameter**: Is 3% the expected anomaly rate, or should this be tuned per device type?

8. **Feature Selection**: Are the 8 metrics the right ones? Are there other signals we should include?

9. **Window Size**: Is 12 hours the right window? Should it vary by metric?

10. **Model Persistence**: Should trained models be saved and reused, or retrain each run?

### Output & Integration

11. **Output Format**: Where should results go? Database table? File? API? Dashboard?

12. **Alerting**: Should anomalies trigger alerts? Email? Slack? Webhook?

13. **Explanation**: When Phase 3 (LLM) is added, what format should explanations be in? Natural language? Structured JSON?

14. **Scheduling**: Should this run on a schedule? Cron? Airflow? Kubernetes job?

15. **Scalability**: Expected data volume? How many devices? How many hours of history?

### Configuration & Environment

16. **Environment Variables**: What's the production environment? Where are secrets stored? (Vault? AWS Secrets Manager?)

17. **Database Connection Pooling**: Should we use connection pooling? What's the expected concurrency?

18. **Logging**: Where should logs go? File? CloudWatch? ELK stack?

---

## 9. Architecture Recommendations

### Short-Term Improvements

1. **Add CLI Framework**: Use `click` or `argparse` for proper CLI interface
2. **Add Configuration Management**: Use `pydantic-settings` or `python-dotenv` properly
3. **Add Output Module**: Create `output/` module for writing results (CSV, DB, JSON)
4. **Add Validation Module**: Create `validation/` for input and data schema validation

### Medium-Term Improvements

5. **Add Model Persistence**: Save/load trained models using `joblib` or `pickle`
6. **Add Feature Store**: Cache computed features to avoid recomputation
7. **Add Monitoring**: Add metrics (precision, recall, data quality) to monitoring system
8. **Add Tests**: Comprehensive test suite with >80% coverage

### Long-Term Improvements

9. **Add API Layer**: FastAPI or Flask REST API for real-time scoring
10. **Add Streaming Support**: Kafka/Pulsar integration for real-time anomaly detection
11. **Add LLM Integration**: Phase 3 - explanation generation
12. **Add Dashboard**: Grafana or custom dashboard for visualization
13. **Add A/B Testing**: Compare different models or parameters

---

## 10. File Reference Map

| File Path | Purpose | Key Functions/Classes |
|-----------|---------|----------------------|
| `cli/main.py` | Skeleton entrypoint | `main()` |
| `cli/synthetic_experiment.py` | Synthetic data pipeline | `run_synthetic_experiment()`, `main()` |
| `cli/dw_experiment.py` | Real DW data pipeline | `run_dw_experiment()`, `main()` |
| `data_access/synthetic_generator.py` | Generate test data | `generate_synthetic_device_telemetry()` |
| `data_access/dw_loader.py` | Load DW data | `load_device_hourly_telemetry()` |
| `data_access/dw_connection.py` | DB connection | `create_dw_engine()` |
| `features/device_features.py` | Feature engineering | `DeviceFeatureBuilder.transform()` |
| `models/anomaly_detector.py` | ML model | `AnomalyDetectorIsolationForest` |
| `config/settings.py` | Configuration | `get_settings()`, `AppSettings` |
| `config/logging_config.py` | Logging setup | `setup_logging()` |

---

**Analysis Date**: 2025-01-XX  
**Analyzer**: Software Archaeologist Agent  
**Repo Version**: 0.1.0 (from `__init__.py`)

