# Data Ingestion Architecture for SOTI Anomaly Detection Platform

**Version:** 1.0  
**Last Updated:** December 2024  
**Status:** Design Document

---

## Table of Contents

1. [Overview](#overview)
2. [Current State](#current-state)
3. [Target Architecture](#target-architecture)
4. [Data Sources](#data-sources)
5. [Ingestion Patterns](#ingestion-patterns)
6. [Canonical Telemetry Model](#canonical-telemetry-model)
7. [Implementation Phases](#implementation-phases)
8. [Scalability & Performance](#scalability--performance)
9. [Data Quality & Validation](#data-quality--validation)
10. [Multi-Tenancy](#multi-tenancy)

---

## Overview

This document describes the data ingestion architecture for the SOTI Anomaly Detection Platform, which integrates telemetry from SOTI XSight and SOTI MobiControl into a unified data pipeline.

### Goals

1. **Unified Data Model**: Normalize data from multiple sources into a canonical telemetry model
2. **Real-Time & Batch**: Support both real-time streaming and batch historical data ingestion
3. **Scalability**: Handle thousands of devices with high-frequency updates
4. **Reliability**: Ensure data completeness, handle failures gracefully
5. **Multi-Tenancy**: Maintain tenant isolation throughout the pipeline
6. **Extensibility**: Support future data sources and new datapoints

### Architecture Principles

- **Separation of Concerns**: Ingest, normalize, store, analyze as distinct stages
- **Idempotency**: Ingest operations should be safe to retry
- **Schema Evolution**: Support schema changes without breaking existing pipelines
- **Observability**: Comprehensive logging, metrics, and alerting
- **Privacy by Design**: Minimize sensitive data, support data retention policies

---

## Current State

### Existing Implementation

The current system uses a **Data Warehouse (DW) approach**:

```
SOTI XSight DW (SQL Server)
    ↓
SQLAlchemy + pyodbc
    ↓
pandas DataFrame
    ↓
Feature Engineering
    ↓
Anomaly Detection Models
```

**Current Data Access:**
- Direct SQL queries to XSight data warehouse
- Tables: `Device`, `cs_DataUsageByHour`, `cs_BatteryStat`, `cs_Offline`, `cs_WifiHour`
- Batch processing: Load data for date ranges
- No real-time ingestion

**Limitations:**
- Single data source (XSight DW only)
- No MobiControl integration
- No real-time capabilities
- Limited to what's already in DW
- No unified data model

---

## Target Architecture

### High-Level Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  SOTI MobiControl│     │   SOTI XSight   │
│   REST API       │     │   API / DW      │
└────────┬─────────┘     └────────┬────────┘
         │                        │
         │                        │
    ┌────▼────────────────────────▼────┐
    │     Data Ingestion Layer          │
    │  ┌──────────┐  ┌──────────┐      │
    │  │MobiControl│  │  XSight  │      │
    │  │ Connector │  │Connector │      │
    │  └────┬─────┘  └────┬──────┘      │
    │       │             │             │
    │       └──────┬──────┘             │
    │              │                     │
    │       ┌──────▼──────┐              │
    │       │ Normalizer  │              │
    │       │  (Canonical │              │
    │       │   Model)    │              │
    │       └──────┬──────┘              │
    └──────────────┼─────────────────────┘
                   │
         ┌─────────▼─────────┐
         │  Message Queue    │
         │  (Kafka/RabbitMQ) │
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │  Data Processor   │
         │  (Stream/Batch)   │
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │  Storage Layer    │
         │  ┌──────┐┌──────┐ │
         │  │Time  ││Meta  │ │
         │  │Series││Data  │ │
         │  │DB    ││Store │ │
         │  └──────┘└──────┘ │
         └───────────────────┘
                   │
         ┌─────────▼─────────┐
         │ Anomaly Detection │
         │     Pipeline      │
         └───────────────────┘
```

### Component Responsibilities

#### 1. Data Ingestion Layer

**MobiControl Connector**
- Polls MobiControl REST API for device inventory and events
- Handles authentication (OAuth/API key)
- Implements rate limiting and retry logic
- Supports incremental sync (query since last update)

**XSight Connector**
- Connects to XSight API (if available) or DW
- Streams real-time metrics or queries historical data
- Handles different data formats (time-series vs inventory)

**Normalizer**
- Transforms source-specific data into canonical telemetry model
- Handles schema mapping and data type conversion
- Enriches data with metadata (tenant, site, device context)
- Validates data quality

#### 2. Message Queue

**Purpose:**
- Decouple ingestion from processing
- Buffer data during high load
- Enable multiple consumers (real-time and batch)

**Options:**
- **Apache Kafka**: High throughput, distributed, good for streaming
- **RabbitMQ**: Simpler, good for moderate scale
- **Redis Streams**: Lightweight, good for real-time use cases

#### 3. Data Processor

**Stream Processor** (Real-time)
- Processes events as they arrive
- Updates device state in real-time
- Triggers immediate anomaly detection for critical metrics

**Batch Processor** (Historical)
- Processes historical data for training
- Handles backfills and corrections
- Generates aggregated features

#### 4. Storage Layer

**Time-Series Database**
- Stores telemetry metrics (battery, CPU, memory, etc.)
- Optimized for time-range queries
- Options: InfluxDB, TimescaleDB, ClickHouse

**Metadata Store**
- Device inventory, compliance status, policies
- Relational database (PostgreSQL) or document store
- Updated less frequently than telemetry

#### 5. Anomaly Detection Pipeline

- Consumes normalized telemetry
- Applies feature engineering
- Runs anomaly detection models
- Generates alerts and insights

---

## Data Sources

### SOTI MobiControl

**Data Types:**
- Device inventory (identity, hardware, OS)
- Compliance status and policies
- Application inventory
- Security posture
- Location and geofencing
- Management events (reboots, app installs, etc.)
- Custom attributes

**Update Frequency:**
- Periodic (device check-ins)
- Event-driven (policy changes, app installs)

**Ingestion Method:**
- REST API polling
- Incremental sync (query since last update)
- Webhooks (if available)

### SOTI XSight

**Data Types:**
- Real-time performance metrics (CPU, memory, storage)
- Battery health and charging status
- Network signal strength
- Device temperature
- Running applications
- Alerts and incidents

**Update Frequency:**
- Near real-time (seconds to minutes)
- Continuous streaming

**Ingestion Method:**
- REST API (if available)
- Direct DW access (current approach)
- Real-time streaming (if supported)

---

## Ingestion Patterns

### Pattern 1: Incremental Sync (MobiControl)

**Use Case:** Device inventory, compliance status, app inventory

**Flow:**
```
1. Poll MobiControl API for devices updated since last sync
2. For each device:
   - Fetch current state (inventory, compliance, apps)
   - Compare with last known state
   - Emit changes as events
3. Update last sync timestamp
4. Repeat at configured interval (e.g., every 15 minutes)
```

**Implementation:**
```python
class MobiControlIncrementalSync:
    def sync_devices_since(self, last_sync_time: datetime):
        """Sync devices updated since last sync."""
        # Query devices modified since last sync
        devices = self.client.get_devices(
            filter=f"LastModifiedDate gt {last_sync_time.isoformat()}"
        )
        
        for device in devices:
            # Fetch full device details
            device_details = self.client.get_device(device["deviceId"])
            
            # Normalize to canonical model
            normalized = self.normalizer.normalize_device(device_details)
            
            # Emit to message queue
            self.queue.publish("device.inventory", normalized)
            
            # Update last known state
            self.state_store.update_device_state(device["deviceId"], normalized)
        
        # Update sync timestamp
        self.state_store.update_sync_time("mobicontrol", datetime.now())
```

### Pattern 2: Real-Time Streaming (XSight)

**Use Case:** Performance metrics, battery, network

**Flow:**
```
1. Subscribe to XSight real-time metrics stream
2. For each metric update:
   - Normalize to canonical model
   - Enrich with device metadata
   - Emit to message queue
3. Process in real-time for anomaly detection
```

**Implementation:**
```python
class XSightStreamingConnector:
    def stream_metrics(self):
        """Stream real-time metrics from XSight."""
        # Subscribe to metrics stream (WebSocket or SSE)
        for metric_update in self.xsight_client.stream_metrics():
            # Normalize
            normalized = self.normalizer.normalize_metric(metric_update)
            
            # Enrich with device metadata
            device_meta = self.metadata_store.get_device(normalized["deviceId"])
            normalized["deviceModel"] = device_meta["model"]
            normalized["tenantId"] = device_meta["tenantId"]
            
            # Emit to queue
            self.queue.publish("device.metrics", normalized)
```

### Pattern 3: Batch Historical Load

**Use Case:** Training data, backfills, corrections

**Flow:**
```
1. Query historical data for date range
2. Process in batches (e.g., 1000 devices at a time)
3. Normalize and validate
4. Load into time-series database
5. Trigger feature engineering and model training
```

**Implementation:**
```python
class BatchHistoricalLoader:
    def load_historical_data(
        self,
        start_date: datetime,
        end_date: datetime,
        device_ids: Optional[List[str]] = None
    ):
        """Load historical data in batches."""
        # Get device list
        devices = device_ids or self.get_all_device_ids()
        
        # Process in batches
        batch_size = 1000
        for i in range(0, len(devices), batch_size):
            batch = devices[i:i+batch_size]
            
            # Query historical data for batch
            data = self.xsight_client.get_historical_metrics(
                device_ids=batch,
                start_date=start_date,
                end_date=end_date
            )
            
            # Normalize
            normalized = [self.normalizer.normalize_metric(d) for d in data]
            
            # Load into time-series DB
            self.tsdb.batch_insert("device_metrics", normalized)
```

### Pattern 4: Event-Driven (Webhooks)

**Use Case:** Immediate notification of critical events

**Flow:**
```
1. Register webhook endpoints with SOTI
2. Receive events (device alerts, policy violations, etc.)
3. Process immediately
4. Trigger real-time anomaly detection
```

**Implementation:**
```python
@app.post("/webhooks/mobicontrol")
async def handle_mobicontrol_webhook(event: dict):
    """Handle webhook from MobiControl."""
    # Validate webhook signature
    if not validate_webhook_signature(event):
        raise HTTPException(401, "Invalid signature")
    
    # Normalize event
    normalized = normalizer.normalize_event(event)
    
    # Emit to queue for processing
    queue.publish("device.events", normalized)
    
    # If critical, trigger immediate anomaly check
    if normalized["severity"] == "Critical":
        anomaly_detector.check_device(normalized["deviceId"])
```

---

## Canonical Telemetry Model

### Core Schema

```python
@dataclass
class DeviceTelemetry:
    """Canonical telemetry model for device data."""
    
    # Identity
    device_id: str
    serial_number: Optional[str]
    imei: Optional[str]
    
    # Multi-tenancy
    tenant_id: str
    site_id: Optional[str]
    
    # Timestamp
    timestamp: datetime
    
    # Device State
    device_name: Optional[str]
    device_model: Optional[str]
    manufacturer: Optional[str]
    os_version: Optional[str]
    security_patch_level: Optional[str]
    
    # Performance Metrics
    cpu_usage: Optional[float]  # Percentage
    memory_usage: Optional[int]  # Bytes
    memory_available: Optional[int]  # Bytes
    storage_usage: Optional[int]  # Bytes
    storage_available: Optional[int]  # Bytes
    device_temperature: Optional[float]  # Celsius
    cpu_temperature: Optional[float]  # Celsius
    
    # Battery
    battery_level: Optional[int]  # Percentage
    battery_health: Optional[str]  # Good, Fair, Poor
    charging_status: Optional[str]  # Charging, Not Charging
    battery_temperature: Optional[float]  # Celsius
    
    # Network
    wifi_ssid: Optional[str]
    wifi_signal_strength: Optional[int]  # dBm
    cellular_carrier: Optional[str]
    cellular_signal_strength: Optional[int]  # dBm
    cellular_network_type: Optional[str]  # LTE, 5G, etc.
    ip_address: Optional[str]
    vpn_status: Optional[str]  # Connected, Disconnected
    
    # Connectivity
    uptime: Optional[int]  # Seconds
    last_connected_time: Optional[datetime]
    
    # Security
    compliance_status: Optional[str]  # Compliant, Non-Compliant
    encryption_status: Optional[str]  # Encrypted, Not Encrypted
    passcode_status: Optional[str]  # Set, Not Set
    jailbreak_status: Optional[str]  # Not Jailbroken, Jailbroken
    
    # Location
    latitude: Optional[float]
    longitude: Optional[float]
    location_accuracy: Optional[float]  # Meters
    geofence_status: Optional[List[Dict]]  # List of geofences
    
    # Applications
    installed_apps: Optional[List[Dict]]  # List of apps
    running_apps: Optional[List[Dict]]  # List of running apps
    
    # Source metadata
    source_system: str  # "mobicontrol" or "xsight"
    source_event_type: Optional[str]  # "inventory", "metric", "event"
    raw_data: Optional[Dict]  # Original data for debugging
```

### Normalization Rules

**Device ID Mapping:**
- Primary: Use MobiControl Device ID as canonical
- Fallback: Serial Number or IMEI if Device ID unavailable
- Cross-reference: Maintain mapping table for XSight device IDs

**Timestamp Normalization:**
- Convert all timestamps to UTC
- Store as ISO 8601 format
- Preserve timezone information in metadata

**Data Type Conversion:**
- Percentages: Convert to 0-100 range
- Signal strength: Convert to dBm (negative values)
- Storage: Convert to bytes (standardize units)
- Temperature: Convert to Celsius

**Enrichment:**
- Join device metadata (model, manufacturer) from inventory
- Add tenant/site context from device assignment
- Include data quality flags (missing, estimated, actual)

---

## Implementation Phases

### Phase 1: MVP - Basic Ingestion (Weeks 1-4)

**Goals:**
- MobiControl REST API integration
- Basic normalization to canonical model
- Store in existing DW or simple database
- Support batch processing

**Components:**
- MobiControl REST API client
- Basic normalizer
- Simple storage (extend existing DW or add PostgreSQL)
- Batch ingestion script

**Deliverables:**
- MobiControl connector library
- Normalization service
- Data validation
- Basic monitoring

### Phase 2: Real-Time Capabilities (Weeks 5-8)

**Goals:**
- XSight API integration (or streaming from DW)
- Message queue integration
- Real-time processing pipeline
- Stream processing for critical metrics

**Components:**
- XSight connector
- Message queue (Kafka or RabbitMQ)
- Stream processor
- Real-time anomaly detection triggers

**Deliverables:**
- Streaming pipeline
- Real-time metrics ingestion
- Event-driven anomaly detection

### Phase 3: Scale & Optimize (Weeks 9-12)

**Goals:**
- Time-series database for metrics
- Optimized storage and queries
- Improved error handling and retries
- Performance monitoring

**Components:**
- Time-series database (InfluxDB or TimescaleDB)
- Query optimization
- Caching layer
- Performance metrics

**Deliverables:**
- Optimized storage
- Query performance improvements
- Comprehensive monitoring

### Phase 4: Advanced Features (Weeks 13-16)

**Goals:**
- Webhook support
- Advanced data quality checks
- Schema evolution handling
- Multi-tenant optimizations

**Components:**
- Webhook handlers
- Data quality service
- Schema registry
- Tenant isolation enhancements

**Deliverables:**
- Event-driven ingestion
- Data quality dashboard
- Schema management

---

## Scalability & Performance

### Scaling Strategies

**Horizontal Scaling:**
- Multiple ingestion workers (process different device sets)
- Partition message queue by tenant or device
- Distribute processing across multiple nodes

**Vertical Scaling:**
- Optimize database queries
- Use connection pooling
- Implement caching for frequently accessed data

**Data Partitioning:**
- Partition by tenant ID
- Partition by time (monthly/quarterly)
- Partition by device ID hash

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Ingestion Rate | 10,000 events/sec | Peak capacity |
| Latency (P50) | < 100ms | End-to-end ingestion |
| Latency (P99) | < 1s | End-to-end ingestion |
| Data Freshness | < 5 minutes | Time from source to available |
| Storage Query | < 100ms | Time-series queries |

### Optimization Techniques

1. **Batch Processing**
   - Group multiple updates per device
   - Batch database inserts
   - Use bulk operations

2. **Caching**
   - Cache device metadata (changes infrequently)
   - Cache authentication tokens
   - Cache frequently queried data

3. **Async Processing**
   - Use async I/O for API calls
   - Process in background threads/workers
   - Non-blocking database operations

4. **Data Compression**
   - Compress historical data
   - Use efficient serialization (Avro, Parquet)
   - Archive old data

---

## Data Quality & Validation

### Validation Rules

**Required Fields:**
- `device_id`, `tenant_id`, `timestamp`, `source_system`

**Data Type Validation:**
- Numeric ranges (e.g., battery 0-100, temperature -50 to 100)
- String formats (e.g., ISO 8601 timestamps, valid IP addresses)
- Enum values (e.g., compliance status, charging status)

**Business Rules:**
- Device ID must exist in device inventory
- Timestamp must be within reasonable range (not future, not too old)
- Tenant ID must match authenticated tenant

### Data Quality Metrics

- **Completeness**: Percentage of expected fields present
- **Accuracy**: Validation against known good data
- **Timeliness**: Time from source event to ingestion
- **Consistency**: Cross-field validation (e.g., battery + charging status)

### Error Handling

**Validation Errors:**
- Log error with full context
- Store in dead letter queue for manual review
- Continue processing other records

**API Errors:**
- Retry with exponential backoff
- Alert on persistent failures
- Fallback to cached data if available

**Data Corruption:**
- Detect anomalies in data patterns
- Flag suspicious records
- Maintain audit trail

---

## Multi-Tenancy

### Tenant Isolation

**Data Isolation:**
- All queries filtered by tenant ID
- Separate database schemas or tables per tenant (optional)
- Row-level security in database

**API Access:**
- Tenant context in all API requests
- Validate tenant access before processing
- Audit tenant access logs

**Processing:**
- Process tenants in separate workers (optional)
- Queue partitioning by tenant
- Tenant-specific rate limiting

### Tenant Context Flow

```
API Request
    ↓
Extract Tenant ID (from auth token or header)
    ↓
Validate Tenant Access
    ↓
Filter Data by Tenant
    ↓
Process with Tenant Context
    ↓
Store with Tenant ID
    ↓
Return Tenant-Scoped Results
```

### Implementation

```python
class TenantAwareIngestion:
    def ingest_device_data(self, data: dict, tenant_id: str):
        """Ingest data with tenant context."""
        # Validate tenant access
        if not self.tenant_service.has_access(tenant_id, data["device_id"]):
            raise UnauthorizedError("Device not accessible by tenant")
        
        # Add tenant context
        data["tenant_id"] = tenant_id
        
        # Normalize
        normalized = self.normalizer.normalize(data)
        
        # Store with tenant isolation
        self.storage.store(normalized, tenant_id=tenant_id)
```

---

## Next Steps

1. **Validate API Access**
   - Obtain SOTI API credentials
   - Test actual endpoints
   - Confirm authentication methods

2. **Prototype Connectors**
   - Build MVP MobiControl connector
   - Test with sample data
   - Validate normalization logic

3. **Design Storage Schema**
   - Design time-series schema
   - Design metadata schema
   - Plan migration from current DW

4. **Set Up Infrastructure**
   - Choose message queue solution
   - Set up time-series database
   - Configure monitoring

5. **Implement MVP**
   - Build Phase 1 components
   - Test end-to-end flow
   - Validate data quality

---

## Production Hardening (January 2025)

This section documents the production-hardened data ingestion implementation with safety guarantees.

### Feature Flags and Safe Defaults

All extended data ingestion features are **OFF by default** to prevent accidental production impact.

```bash
# Core ingestion (safe, always available)
ENABLE_SCHEMA_DISCOVERY=true         # Metadata discovery only, no table scans

# Extended ingestion (OFF by default - enable explicitly)
ENABLE_MC_TIMESERIES=false           # MobiControl DeviceStatInt, DeviceStatLocation, etc.
ENABLE_XSIGHT_HOURLY=false           # XSight hourly tables (HIGH VOLUME - 6M+ rows)
ENABLE_XSIGHT_EXTENDED=false         # XSight extended tables (cs_WiFiLocation, etc.)

# Watermark configuration
ENABLE_FILE_WATERMARK_FALLBACK=false # File fallback disabled in production

# Observability (ON by default)
ENABLE_INGESTION_METRICS=true        # Per-table metrics to PostgreSQL
ENABLE_DAILY_COVERAGE_REPORT=true    # Daily aggregated report
```

### Table Allowlists

For fine-grained control, use allowlists to enable specific tables only:

```bash
# Only ingest these specific tables (comma-separated)
XSIGHT_TABLE_ALLOWLIST="cs_DataUsageByHour,cs_BatteryStat,cs_Offline"
MC_TABLE_ALLOWLIST="DeviceStatInt,DeviceStatLocation"
```

When an allowlist is set, only tables in that list will be ingested, even if broader feature flags are enabled.

### Monotonic Watermarks

Watermarks are **monotonic** - they can only move forward, never backward.

**Key Behaviors:**
- PostgreSQL is the **single source of truth** for watermarks
- Redis is an optional write-through cache only
- `set_watermark()` rejects any watermark < current watermark
- To move backward (e.g., for backfill), use `reset_watermark()` explicitly

```python
from device_anomaly.data_access.watermark_store import get_watermark_store

store = get_watermark_store()

# Normal operation - only moves forward
success, error = store.set_watermark("xsight", "cs_DataUsageByHour", new_watermark)
if not success:
    logger.warning(f"Watermark rejected: {error}")

# Explicit backfill - resets watermark
new_wm, ok = store.reset_watermark("xsight", "cs_DataUsageByHour", lookback_hours=72)
```

### Idempotent Event IDs

All canonical events have a stable `event_id` computed via SHA256 from:
- source_db, source_table, device_id, event_time, metric_name, metric_value, dimensions

This ensures:
- Re-ingesting the same data produces identical event IDs
- Deduplication on write prevents duplicate records
- Backfills don't create duplicates

```python
from device_anomaly.data_access.canonical_events import dedupe_events

# Dedupe events before writing
unique_events, seen_ids = dedupe_events(events)
```

### Keyset Pagination (No OFFSET)

Large table loads use **keyset pagination** instead of OFFSET for stable, efficient pagination:

```sql
-- Instead of: SELECT ... OFFSET 50000 LIMIT 50000 (slow, unstable)
-- We use keyset pagination:
SELECT TOP 50000 * FROM cs_DataUsageByHour
WHERE CollectedDate > @last_date
  OR (CollectedDate = @last_date AND Hour > @last_hour)
ORDER BY CollectedDate, Hour
```

Benefits:
- O(1) pagination regardless of offset
- Stable results even if data changes during pagination
- Works efficiently with SQL Server clustered indexes

### Weight-Based Parallelism Throttling

Concurrent table loads are throttled via a **weighted semaphore**:

| Table Category | Weight | Effect |
|----------------|--------|--------|
| XSight hourly huge (cs_DataUsageByHour, cs_WiFiLocation) | 5 | Only 1 at a time |
| XSight extended tables | 2 | Max 2 concurrent |
| MobiControl time-series | 2 | Max 2 concurrent |
| Small tables (Alert, Events) | 1 | Max 5 concurrent |

Default `MAX_INGEST_WEIGHT=5` ensures SQL Server isn't overwhelmed.

```python
from device_anomaly.services import IngestionOrchestrator

orchestrator = IngestionOrchestrator(max_weight=5)
result = await orchestrator.run_batch(tables, loader_func)
```

### Observability

**Per-Table Metrics** (stored in `ingestion_metrics` table):
- rows_fetched, rows_inserted, rows_deduped
- watermark_start, watermark_end
- lag_seconds (how far behind current time)
- duration_ms, query_time_ms, transform_time_ms
- success/error status

**Daily Coverage Report** (stored in `telemetry_coverage_report` table):
- Tables loaded/failed per source database
- Total rows processed
- Average and max lag
- Tables with errors, high lag (>1hr), or high dedupe ratio (>20%)

```python
from device_anomaly.services import generate_daily_coverage_report

# Run daily (e.g., via scheduler)
report = generate_daily_coverage_report()
```

### Recommended SQL Server Indexes

For optimal keyset pagination performance, ensure these indexes exist:

```sql
-- XSight hourly tables
CREATE CLUSTERED INDEX IX_cs_DataUsageByHour_CollectedDate_Hour
ON cs_DataUsageByHour (CollectedDate, Hour);

CREATE CLUSTERED INDEX IX_cs_WiFiLocation_CollectedDate_Hour
ON cs_WiFiLocation (CollectedDate, Hour);

-- MobiControl time-series tables
CREATE CLUSTERED INDEX IX_DeviceStatInt_ServerDateTime
ON DeviceStatInt (ServerDateTime);

CREATE CLUSTERED INDEX IX_DeviceStatLocation_ServerDateTime
ON DeviceStatLocation (ServerDateTime);

-- For device filtering
CREATE NONCLUSTERED INDEX IX_DeviceStatInt_DeviceId_ServerDateTime
ON DeviceStatInt (DeviceId, ServerDateTime);
```

### Graceful Degradation

The system gracefully handles missing or unavailable tables:

- Missing tables: Logged as warning, skipped without error
- Missing columns: Query uses only available columns
- Schema changes: Cache invalidated on table modification date change
- Database unavailable: Returns empty results with structured warning

```python
# Schema discovery caches valid tables
from device_anomaly.data_access.schema_discovery import get_curated_table_list

tables = get_curated_table_list(engine)  # Only returns existing, valid tables
```

### StatType Mapping

MobiControl `DeviceStatInt` uses integer StatType codes. Use the discovery function to get human-readable names:

```python
from device_anomaly.data_access.stat_type_mapper import (
    get_stat_type_name,
    discover_stat_types,
)

# Get name for known type
name = get_stat_type_name(3)  # Returns "AvailableStorage"

# Discover all types from database
types = discover_stat_types(engine)
```

### Configuration Reference

Full environment variable reference:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_MC_TIMESERIES` | `false` | Enable MobiControl time-series ingestion |
| `ENABLE_XSIGHT_HOURLY` | `false` | Enable XSight hourly tables (HIGH VOLUME) |
| `ENABLE_XSIGHT_EXTENDED` | `false` | Enable XSight extended tables |
| `ENABLE_SCHEMA_DISCOVERY` | `true` | Enable runtime schema discovery caching |
| `XSIGHT_TABLE_ALLOWLIST` | `` | Comma-separated list of allowed XSight tables |
| `MC_TABLE_ALLOWLIST` | `` | Comma-separated list of allowed MC tables |
| `INGEST_LOOKBACK_HOURS` | `24` | Default lookback for new tables |
| `INGEST_BATCH_SIZE` | `50000` | Max rows per batch |
| `INGEST_MAX_TABLES_PARALLEL` | `3` | Max concurrent table loads (weight) |
| `MAX_BACKFILL_DAYS_HOURLY` | `2` | Max days to backfill for hourly tables |
| `ENABLE_FILE_WATERMARK_FALLBACK` | `false` | Enable file-based watermark fallback |
| `ENABLE_INGESTION_METRICS` | `true` | Enable per-table metrics |
| `ENABLE_DAILY_COVERAGE_REPORT` | `true` | Enable daily coverage report |

---

**Document Status:** Implementation Complete - Production Ready
**Last Updated:** January 2025

