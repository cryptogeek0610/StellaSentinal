# Data Discovery & Extended Ingestion

This document describes the enhanced data discovery and ingestion capabilities for Stella Sentinel.

## Overview

The system now supports:
- **Runtime schema discovery** for both SOTI_XSight_dw and MobiControlDB
- **Incremental extraction** with watermark-based change detection
- **Extended table support** - from 5 tables to 43+ in XSight, plus 7 new MobiControl time-series tables
- **Canonical event normalization** - unified format for ML pipelines

## Database Inventory

### SOTI_XSight_dw (242 objects)

**Original Tables (5):**
- `cs_BatteryStat`
- `cs_AppUsage`
- `cs_DataUsage`
- `cs_BatteryAppDrain`
- `cs_Heatmap`

**Extended Tables (13 total):**
- `cs_DataUsageByHour` - 104M rows, hourly data usage per device/app
- `cs_BatteryLevelDrop` - 14.8M rows, battery drain patterns
- `cs_AppUsageListed` - 8.5M rows, app foreground time
- `cs_WifiHour` - 755K rows, WiFi connectivity patterns
- `cs_WiFiLocation` - 790K rows, WiFi + GPS location
- `cs_LastKnown` - 674K rows, last known location + signal
- `cs_DeviceInstalledApp` - 372K rows, app install events
- `cs_PresetApps` - 19M rows, preset app usage

### MobiControlDB (809 objects)

**New Time-Series Tables:**
- `DeviceStatInt` - 764K rows, device integer metrics over time
- `DeviceStatLocation` - 619K rows, GPS tracking history
- `DeviceStatString` - 349K rows, device string metrics
- `DeviceStatNetTraffic` - 244K rows, network traffic per app
- `MainLog` - 1M rows, activity log
- `Alert` - 1.3K rows, system alerts
- `DeviceInstalledApp` - 1.5M rows, app inventory

## Environment Variables

### Feature Flags (Safe Deployment)

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_MC_TIMESERIES` | `false` | Enable MobiControl time-series tables |
| `ENABLE_XSIGHT_HOURLY` | `false` | Enable XSight hourly tables (high volume) |
| `ENABLE_XSIGHT_EXTENDED` | `true` | Enable extended XSight tables |
| `ENABLE_SCHEMA_DISCOVERY` | `true` | Enable runtime schema discovery caching |

### Ingestion Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `INGEST_LOOKBACK_HOURS` | `24` | Default lookback for new tables |
| `INGEST_BATCH_SIZE` | `50000` | Max rows per batch |
| `INGEST_MAX_TABLES_PARALLEL` | `3` | Max concurrent table loads |

## Module Reference

### schema_discovery.py

Runtime table/view discovery with caching.

```python
from device_anomaly.data_access.schema_discovery import (
    discover_xsight_schema,
    discover_mobicontrol_schema,
    get_high_value_tables,
    SourceDatabase,
)

# Discover all tables in XSight
schema = discover_xsight_schema()
print(f"Found {len(schema.tables)} tables, {len(schema.views)} views")

# Get high-value tables for ML
tables = get_high_value_tables(
    source_db=SourceDatabase.MOBICONTROL,
    min_rows=1000,
    require_time_series=True,
)
```

### watermark_store.py

Incremental extraction with watermarks.

```python
from device_anomaly.data_access.watermark_store import get_watermark_store

store = get_watermark_store()

# Get last extraction timestamp
watermark = store.get_watermark("mobicontrol", "DeviceStatInt")

# Update watermark after extraction
store.set_watermark(
    source_db="mobicontrol",
    table_name="DeviceStatInt",
    watermark_value=new_timestamp,
    rows_extracted=1000,
)

# Reset watermark for backfill
store.reset_watermark("xsight", "cs_DataUsageByHour", lookback_hours=168)
```

### mc_timeseries_loader.py

Load MobiControl time-series data.

```python
from device_anomaly.data_access.mc_timeseries_loader import (
    load_and_update_watermark,
    load_all_mc_timeseries,
    get_mc_timeseries_stats,
)

# Load incremental data with automatic watermark update
df = load_and_update_watermark("DeviceStatInt", batch_size=50000)

# Load all time-series tables
results = load_all_mc_timeseries(batch_size=50000)
for table, df in results.items():
    print(f"{table}: {len(df)} rows")

# Get table statistics
stats = get_mc_timeseries_stats()
```

### xsight_loader_extended.py

Load extended XSight tables.

```python
from device_anomaly.data_access.xsight_loader_extended import (
    load_and_update_watermark,
    load_all_xsight_tables,
    get_xsight_table_stats,
)

# Load priority 1 tables (original 5)
results = load_all_xsight_tables(priority_filter=1)

# Load all extended tables
results = load_all_xsight_tables(priority_filter=3)

# Stream large tables
for batch in stream_xsight_table("cs_DataUsageByHour"):
    process(batch)
```

### canonical_events.py

Normalize data to unified format.

```python
from device_anomaly.data_access.canonical_events import (
    dataframe_to_canonical_events,
    SourceDatabase,
    pivot_events_to_features,
)

# Convert DataFrame to canonical events
events = dataframe_to_canonical_events(
    df=df,
    source_db=SourceDatabase.MOBICONTROL,
    source_table="DeviceStatInt",
)

# Pivot to feature matrix
features = pivot_events_to_features(events, time_bucket="1h")
```

## Deployment Guide

### Phase 1: Enable Extended XSight (Low Risk)

Extended XSight tables are enabled by default. They use the same schema as existing tables.

```bash
# Verify extended tables are enabled (default)
ENABLE_XSIGHT_EXTENDED=true
```

### Phase 2: Enable Schema Discovery

Schema discovery caches table metadata to improve startup time.

```bash
ENABLE_SCHEMA_DISCOVERY=true
```

### Phase 3: Enable MobiControl Time-Series (Higher Volume)

Enable time-series tables after validating capacity.

```bash
# Enable MC time-series
ENABLE_MC_TIMESERIES=true

# Start with smaller batch size
INGEST_BATCH_SIZE=25000
```

### Phase 4: Enable XSight Hourly Tables (High Volume)

Hourly tables have 100M+ rows. Enable only after capacity testing.

```bash
# Enable hourly tables
ENABLE_XSIGHT_HOURLY=true

# Use streaming for backfill
INGEST_BATCH_SIZE=100000
INGEST_LOOKBACK_HOURS=168  # 7 days
```

## Watermark Storage

Watermarks are stored in order of priority:
1. PostgreSQL (`ingestion_watermarks` table) - persistent, multi-instance safe
2. Redis - fast cache, shared
3. File (`data/watermarks.json`) - fallback for development

The PostgreSQL table is created automatically:

```sql
CREATE TABLE IF NOT EXISTS ingestion_watermarks (
    id SERIAL PRIMARY KEY,
    source_db VARCHAR(50) NOT NULL,
    table_name VARCHAR(255) NOT NULL,
    watermark_column VARCHAR(100) NOT NULL,
    watermark_value TIMESTAMP WITH TIME ZONE NOT NULL,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    rows_extracted INTEGER DEFAULT 0,
    metadata_json TEXT,
    UNIQUE(source_db, table_name)
);
```

## Canonical Event Schema

All data is normalized to this format:

```json
{
    "tenant_id": "default",
    "source_db": "mobicontrol",
    "source_table": "DeviceStatInt",
    "device_id": 12345,
    "event_time": "2024-01-15T10:30:00Z",
    "metric_name": "battery_level",
    "metric_type": "int",
    "metric_value": 85,
    "dimensions": {
        "stat_type_code": 1
    },
    "raw_hash": "a1b2c3d4e5f67890"
}
```

## MobiControl StatType Mappings

| StatType | Metric Name |
|----------|-------------|
| 1 | battery_level |
| 2 | battery_temperature |
| 3 | battery_voltage |
| 10 | available_ram |
| 11 | total_ram |
| 12 | available_storage |
| 20 | wifi_signal_strength |
| 21 | cell_signal_strength |

Unknown StatTypes are preserved as `stat_type_<code>`.

## Testing

Run the test suite:

```bash
# Unit tests
pytest tests/test_schema_discovery.py -v

# Integration test (requires SQL Server)
pytest tests/test_schema_discovery.py::TestIntegration -v --run-integration
```

## Monitoring

### Structured Logging

All modules use structured logging with the following fields:
- `table_name`: Source table
- `rows_extracted`: Number of rows in batch
- `watermark`: Current watermark value
- `duration_ms`: Operation duration

### Metrics

When observability is enabled, the following metrics are available:
- `ingestion_rows_total` - Total rows ingested per table
- `ingestion_batch_duration_seconds` - Batch processing time
- `watermark_lag_seconds` - Time since last watermark update

## Troubleshooting

### Table Not Found

If a table doesn't exist in the database:
- Discovery logs a warning and continues
- Loading returns empty DataFrame
- No watermark is updated

### Permission Errors

If user lacks SELECT permission:
- Discovery logs structured warning with table name
- Table is excluded from curated list
- Other tables continue processing

### Large Table Backfill

For tables with 100M+ rows:
1. Use streaming: `stream_xsight_table("cs_DataUsageByHour")`
2. Reduce batch size: `INGEST_BATCH_SIZE=25000`
3. Consider date partitioning in queries

### Reset Watermark

To trigger a full backfill:
```python
store = get_watermark_store()
store.reset_watermark("xsight", "cs_DataUsageByHour", lookback_hours=720)  # 30 days
```
