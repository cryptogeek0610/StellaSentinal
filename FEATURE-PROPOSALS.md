# StellaSentinal -- Architecture & Feature Strategy

> **Document Version:** 1.0
> **Date:** 2026-02-06
> **Status:** Draft for review
> **Audience:** Engineering leads, product stakeholders

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture](#2-current-architecture)
   - [System Diagram](#21-system-diagram)
   - [Component Inventory](#22-component-inventory)
   - [Data Flow](#23-data-flow)
3. [TODO Audit -- 14 Unfinished Items](#3-todo-audit--14-unfinished-items)
4. [Feature Proposals](#4-feature-proposals)
   - [FP-01: Alert Delivery (Email + Slack)](#fp-01-alert-delivery-email--slack)
   - [FP-02: Auto-Remediation Execution](#fp-02-auto-remediation-execution)
   - [FP-03: User Management & Authentication](#fp-03-user-management--authentication)
   - [FP-04: General Audit Logging](#fp-04-general-audit-logging)
   - [FP-05: Data Retention Policies](#fp-05-data-retention-policies)
   - [FP-06: Security Module Completion](#fp-06-security-module-completion)
   - [FP-07: Cost Analytics Completion](#fp-07-cost-analytics-completion)
   - [FP-08: Dashboard Custom Attribute Aggregation](#fp-08-dashboard-custom-attribute-aggregation)
   - [FP-09: Worker Health Monitoring](#fp-09-worker-health-monitoring)
   - [FP-10: Cohort Trend Computation](#fp-10-cohort-trend-computation)
5. [Scalability Roadmap](#5-scalability-roadmap)
   - [Phase 1: Connection Pool & Database](#phase-1-connection-pool--database-q2-2026)
   - [Phase 2: Job Queue Migration](#phase-2-job-queue-migration-q2q3-2026)
   - [Phase 3: ML Pipeline Evolution](#phase-3-ml-pipeline-evolution-q3-2026)
   - [Phase 4: Horizontal API Scaling](#phase-4-horizontal-api-scaling-q4-2026)
6. [Priority Matrix](#6-priority-matrix)
7. [Appendix: File Reference](#7-appendix-file-reference)

---

## 1. Executive Summary

StellaSentinal is an AI-powered anomaly detection platform for enterprise device fleets. The system ingests telemetry from MobiControl-managed devices, trains Isolation Forest and VAE models to detect anomalies, and surfaces findings through a React dashboard with LLM-powered explanations.

The platform is functional for single-tenant pilot deployments but has significant gaps that block production readiness: alerts only go to logs, the "Execute Fix" button is a no-op, authentication is header-based with no session management, and the job queue is a hand-rolled Redis list with no retry or dead-letter semantics.

This document catalogs every known gap, proposes concrete solutions, and lays out a phased scalability roadmap.

---

## 2. Current Architecture

### 2.1 System Diagram

```
                          Clients (Browser)
                               |
                         [ Nginx :80 ]
                          /          \
                   React SPA        API Proxy
                   (Vite build)       |
                                [ FastAPI :8000 ]
                                (24 route modules)
                                (15+ services)
                               /    |    \      \
                              /     |     \      \
                    [ PostgreSQL ] [ Redis ] [ Ollama ]  [ Qdrant ]
                       :5432       :6379     :11434       :6333
                                     |
                          +----------+----------+
                          |                     |
                  [ scheduler worker ]   [ ml-worker ]
                   (cron loops)          (blpop queue consumer)
                          |
                  +-----------------+
                  | training queue  |  ml:training:queue (Redis list)
                  | scoring queue   |  scheduler:scoring:queue
                  | insights queue  |  scheduler:insights:queue
                  | device sync q   |  scheduler:device_sync:queue
                  +-----------------+

    External Data Sources:
    +-------------------+     +-------------------+
    | XSight DW         |     | MobiControl DB    |
    | (SQL Server)      |     | (SQL Server)      |
    | :1433             |     | :1433             |
    +-------------------+     +-------------------+
```

### 2.2 Component Inventory

| Layer | Technology | Role |
|-------|-----------|------|
| **API** | FastAPI + Uvicorn | 24 route modules, request context middleware, rate limiting |
| **ORM** | SQLAlchemy 2.x (sync) | 20+ models across `models.py` and `models_cost.py` |
| **Database** | PostgreSQL 16 | Anomalies, baselines, devices, users, audit logs, cost data |
| **Cache / Queue** | Redis 7 | Rate limiting, job queues (4 queues), training status, alert buffer |
| **ML Models** | scikit-learn, PyTorch, ONNX | Isolation Forest, VAE, ensemble detector, calibration |
| **LLM** | Ollama (llama3.2) | Anomaly explanations, investigation assist; rule-based fallback |
| **Vector DB** | Qdrant | Similarity search for anomaly patterns (RAG) |
| **Frontend** | React 18 + TypeScript 5.2 + Vite 5 | SPA with route-level code splitting, Tailwind CSS |
| **Legacy UI** | Streamlit | Power-user dashboard (parallel deployment) |
| **Workers** | 2 Python processes | `scheduler` (cron loops), `ml_worker` (queue consumer) |
| **Infra** | Docker Compose | 8 services, bridge network, volume mounts |

### 2.3 Data Flow

1. **Ingestion**: `ingestion_orchestrator` and `ingestion_pipeline` pull telemetry from XSight DW and MobiControl DB (SQL Server) via SQLAlchemy connectors.
2. **Feature Computation**: `streaming/feature_computer.py` and `features/system_health_features.py` transform raw telemetry into ML-ready features.
3. **Training**: `pipeline/training.py` trains Isolation Forest models on 90 days of data. `pipeline/auto_retrain.py` automates retraining. Models exported to ONNX via `models/onnx_exporter.py`.
4. **Scoring**: Scheduler triggers scoring runs; results written to PostgreSQL `anomaly_events` and `anomalies` tables.
5. **Insights**: `correlation_service`, `cohort_detector`, `security_grouper`, and LLM generate higher-order insights.
6. **Presentation**: React frontend calls FastAPI endpoints; SSE streaming via `streaming/websocket_manager.py`.

---

## 3. TODO Audit -- 14 Unfinished Items

The following TODO comments were found across the Python backend. Each represents incomplete functionality shipping in the current build.

| # | File | Line | TODO Comment | Severity |
|---|------|------|-------------|----------|
| 1 | `workers/scheduler.py` | 1120 | `Implement actual alerting (email, Slack, webhook)` | **Critical** -- alerts go to logs only |
| 2 | `api/routes/action_center.py` | 433 | `Actually execute the fix via resolver` | **Critical** -- "Fix" button is a no-op |
| 3 | `api/routes/training.py` | 898 | `Add worker health check` | High -- `worker_available` hardcoded to `True` |
| 4 | `api/routes/dashboard.py` | 1627 | `Implement actual data aggregation when custom attributes are available` | High -- location dashboard returns empty data in live mode |
| 5 | `models/cohort_detector.py` | 360 | `Compute trend from historical data` | Medium -- trend hardcoded to `"stable"` |
| 6 | `api/routes/security.py` | 846 | `Query devices sorted by security score ascending` | High -- at-risk devices returns empty in live mode |
| 7 | `api/routes/security.py` | 874 | `Enrich with real security metrics` | High -- path hierarchy lacks actual metrics |
| 8 | `api/routes/security.py` | 945 | `Implement real PATH grouping` | High -- security-by-path returns empty in live mode |
| 9 | `api/routes/security.py` | 975 | `Use SecurityGrouper to compute real clusters` | High -- risk clusters endpoint is a stub |
| 10 | `api/routes/security.py` | 1009 | `Use SecurityGrouper to compute real comparison` | High -- path comparison endpoint is a stub |
| 11 | `api/routes/security.py` | 1036 | `Use SecurityGrouper to find real temporal correlations` | High -- temporal clusters endpoint is a stub |
| 12 | `api/routes/costs/summary.py` | 137 | `Calculate from cache` (anomaly impact MTD) | Medium -- anomaly cost impact always returns 0 |
| 13 | `api/routes/costs/summary.py` | 138 | `Calculate from cache` (anomaly impact YTD) | Medium -- anomaly cost impact always returns 0 |
| 14 | `services/proactive_resolver.py` | 384 | `Real implementation` (data query stubs) | **Critical** -- all proactive resolver queries return empty |

---

## 4. Feature Proposals

### FP-01: Alert Delivery (Email + Slack)

| Field | Value |
|-------|-------|
| **Priority** | P0 -- Critical |
| **Effort** | Medium (3-5 days) |
| **Impact** | High -- without this, anomalies detected overnight are invisible until someone checks the dashboard |

**Problem Statement**

`scheduler.py:1117-1138` defines `_send_alert()` which logs a warning and pushes to a Redis list capped at 100 entries. There is no email, Slack, webhook, or any external notification channel. Alerts are effectively silent.

**Proposed Solution**

1. Create `src/device_anomaly/notifications/` package with pluggable backends:
   - `email_backend.py` -- SendGrid or SMTP integration with HTML templates for anomaly alerts.
   - `slack_backend.py` -- Incoming webhook integration with Block Kit formatted messages including anomaly severity, affected device count, and a deep-link to the dashboard.
   - `webhook_backend.py` -- Generic HTTP POST with configurable URL, headers, and JSON payload template.
2. Add `NotificationSettings` to `config/settings.py` with env-var-driven configuration (`SENDGRID_API_KEY`, `SLACK_WEBHOOK_URL`, etc.).
3. Add an `alert_channels` table in PostgreSQL to persist per-tenant channel configuration (type, destination, severity filter, enabled flag).
4. Replace the `logger.warning` call in `_send_alert()` with a dispatch loop through enabled channels.
5. Add rate limiting per channel to prevent alert storms (e.g., max 10 emails per hour per tenant).

**Files Affected**

- `src/device_anomaly/workers/scheduler.py` (modify `_send_alert`)
- `src/device_anomaly/notifications/` (new package: `__init__.py`, `email_backend.py`, `slack_backend.py`, `webhook_backend.py`, `dispatcher.py`)
- `src/device_anomaly/config/settings.py` (add `NotificationSettings`)
- `src/device_anomaly/db/models.py` (add `AlertChannel` model)
- `frontend/src/pages/Settings/` (UI for channel configuration)

**Dependencies**: SendGrid account or SMTP server; Slack workspace with webhook permissions.

---

### FP-02: Auto-Remediation Execution

| Field | Value |
|-------|-------|
| **Priority** | P0 -- Critical |
| **Effort** | Large (8-12 days) |
| **Impact** | High -- the Action Center is the primary value proposition for operations teams; it currently does nothing |

**Problem Statement**

`action_center.py:431-442` iterates over automatable issues but never executes any fix. It returns `success=True` with a message "Initiated: ..." without calling any external API. Similarly, `proactive_resolver.py:382-405` contains data query stubs that all return empty results.

**Proposed Solution**

1. Implement the MobiControl REST API client in `src/device_anomaly/connectors/mobicontrol_api.py`:
   - OAuth2 client-credentials flow using existing `MOBICONTROL_CLIENT_ID` / `MOBICONTROL_CLIENT_SECRET` env vars.
   - Device action endpoints: send message, lock device, wipe device, push configuration profile, restart service.
   - Rate limiting and retry logic (exponential backoff with jitter).
2. Fill in `proactive_resolver.py` data query stubs to query actual telemetry:
   - `_query_usb_debugging_enabled` -- query DeviceMetadata for `usb_debugging` flag.
   - `_query_unencrypted_devices` -- query DeviceMetadata for `encryption_enabled = false`.
   - `_query_network_dead_zones` -- aggregate WiFi signal data below threshold.
   - `_query_degraded_batteries` -- query battery health predictions from ML baseline.
   - `_query_low_storage_devices` -- query latest storage telemetry below threshold.
3. In `action_center.py`, replace the stub loop with actual MobiControl API calls, executed asynchronously through the job queue to avoid blocking the HTTP response.
4. Add an `RemediationLog` model to track every action taken: device ID, action type, initiated by, result, timestamp.
5. Add a confirmation/approval workflow: critical actions (wipe, lock) require explicit admin approval before execution.

**Files Affected**

- `src/device_anomaly/api/routes/action_center.py` (replace stub logic)
- `src/device_anomaly/services/proactive_resolver.py` (implement query stubs)
- `src/device_anomaly/connectors/mobicontrol_api.py` (new file)
- `src/device_anomaly/db/models.py` (add `RemediationLog`)
- `src/device_anomaly/config/settings.py` (MobiControl API settings already partially present)
- `frontend/src/pages/ActionCenter/` (approval workflow UI)

**Dependencies**: MobiControl API access with device-action permissions; approval workflow depends on FP-03 (user management).

---

### FP-03: User Management & Authentication

| Field | Value |
|-------|-------|
| **Priority** | P0 -- Critical |
| **Effort** | Large (10-15 days) |
| **Impact** | Critical -- header-based auth is not acceptable for production; any client can claim any role |

**Problem Statement**

`api/main.py:244-256` extracts user identity from `X-User-Id` and `X-User-Role` headers when `_trust_client_headers` is enabled. There is no JWT validation, no session management, no login flow, and no user CRUD API. The `User` model exists in `db/models.py` but is not connected to any authentication flow. The `require_role()` dependency in `api/dependencies.py:194-207` checks roles but trusts whatever the client sends.

**Proposed Solution**

1. Add JWT-based authentication:
   - `src/device_anomaly/auth/jwt.py` -- issue and validate RS256 JWTs with configurable expiry.
   - Login endpoint (`POST /auth/login`) accepting email + password, returning access + refresh tokens.
   - Refresh endpoint (`POST /auth/refresh`) for token rotation.
   - Store refresh tokens in Redis with TTL for revocation support.
2. Add user management API:
   - `POST /users` -- create user (admin only).
   - `GET /users` -- list users (admin only).
   - `PATCH /users/{id}` -- update role, deactivate (admin only).
   - `POST /users/{id}/reset-password` -- password reset flow.
3. Add password hashing using `bcrypt` via `passlib`.
4. Update `api/main.py` middleware to validate JWT from `Authorization: Bearer` header, falling back to API key auth for machine-to-machine calls.
5. Update `require_role()` to extract roles from validated JWT claims.
6. Add login page to React frontend with token storage in `httpOnly` cookies or secure localStorage with automatic refresh.
7. Add RBAC matrix: `admin`, `analyst`, `viewer` with endpoint-level permissions.

**Files Affected**

- `src/device_anomaly/auth/` (new package: `jwt.py`, `password.py`, `middleware.py`)
- `src/device_anomaly/api/routes/auth.py` (new route module)
- `src/device_anomaly/api/routes/users.py` (new route module)
- `src/device_anomaly/api/main.py` (update middleware)
- `src/device_anomaly/api/dependencies.py` (update `get_current_user`, `require_role`)
- `src/device_anomaly/db/models.py` (extend `User` model with password hash, last login, active flag)
- `frontend/src/pages/Login/` (new page)
- `frontend/src/pages/UserManagement/` (new page)
- `frontend/src/lib/api.ts` (add auth interceptor)

**Dependencies**: None (self-contained). Should be implemented before FP-02 approval workflows.

---

### FP-04: General Audit Logging

| Field | Value |
|-------|-------|
| **Priority** | P1 -- High |
| **Effort** | Medium (3-5 days) |
| **Impact** | High -- required for SOC2 / GDPR compliance |

**Problem Statement**

The `AuditLog` model exists in `db/models.py:319-346` with proper schema (user_id, tenant_id, action, resource_type, resource_id, ip_address, extra_data) and indexes, but it is never written to. Only `CostAuditLog` (a separate cost-specific model in `models_cost.py:197`) is actively used. All other write operations (anomaly acknowledgment, baseline updates, device actions, configuration changes) go unaudited.

**Proposed Solution**

1. Create `src/device_anomaly/audit/middleware.py` with a FastAPI middleware or dependency that captures write operations:
   - Intercept all POST, PUT, PATCH, DELETE requests.
   - Extract user ID, tenant ID, IP address from request context.
   - Log resource type and ID from the route path.
   - Store request body diff (before/after) in `extra_data` as JSON.
2. Create `src/device_anomaly/audit/service.py` with a `log_audit_event()` helper for explicit audit logging within service layer code.
3. Wire the `AuditLog` model to the existing database session.
4. Add a `GET /audit-logs` admin endpoint with filtering by user, action, resource type, date range.
5. Add audit log viewer to the frontend admin section.
6. Consider async writes (queue audit events to Redis, flush to PostgreSQL in batches) to avoid adding latency to every write request.

**Files Affected**

- `src/device_anomaly/audit/` (new package: `middleware.py`, `service.py`)
- `src/device_anomaly/api/main.py` (register audit middleware)
- `src/device_anomaly/api/routes/audit.py` (new route module for log viewing)
- `src/device_anomaly/db/models.py` (model already exists; no changes needed)
- `frontend/src/pages/Admin/AuditLog.tsx` (new component)

**Dependencies**: FP-03 (user management) for meaningful user attribution.

---

### FP-05: Data Retention Policies

| Field | Value |
|-------|-------|
| **Priority** | P1 -- High |
| **Effort** | Medium (3-5 days) |
| **Impact** | High -- telemetry data grows unbounded; production deployments will hit storage limits |

**Problem Statement**

There are no data retention policies, TTL-based cleanup jobs, or partition strategies. The `telemetry_points` table will grow without bound. The `anomaly_events` table similarly accumulates forever. Redis lists like `scheduler:alerts` are trimmed to 100 entries, but PostgreSQL tables have no equivalent lifecycle management.

**Proposed Solution**

1. Add `RetentionPolicy` model to `db/models.py`:
   - Fields: table_name, retention_days, tenant_id (nullable for global), enabled, last_run_at.
   - Default policies: telemetry_points (90 days), anomaly_events (365 days), audit_logs (730 days), training history (180 days).
2. Add `src/device_anomaly/workers/retention_worker.py`:
   - Runs as a scheduled task within the existing scheduler worker.
   - Deletes rows older than retention threshold in batches (1000 rows per transaction) to avoid long-running locks.
   - Logs deletion counts to audit log.
3. Add PostgreSQL table partitioning for `telemetry_points` by month (using declarative partitioning):
   - Enables `DROP PARTITION` for instant cleanup instead of row-by-row DELETE.
   - Requires Alembic migration.
4. Add retention configuration UI to admin settings.
5. Add `GET /system/storage` endpoint reporting table sizes and projected growth.

**Files Affected**

- `src/device_anomaly/db/models.py` (add `RetentionPolicy`)
- `src/device_anomaly/workers/scheduler.py` (add retention task to cron loop)
- `src/device_anomaly/workers/retention_worker.py` (new file)
- `src/device_anomaly/api/routes/system_health.py` (add storage reporting endpoint)
- `src/device_anomaly/config/settings.py` (default retention values)
- Database migration for table partitioning

**Dependencies**: None.

---

### FP-06: Security Module Completion

| Field | Value |
|-------|-------|
| **Priority** | P1 -- High |
| **Effort** | Large (8-10 days) |
| **Impact** | High -- 6 of 14 TODOs are in security.py; the entire security posture dashboard is mock-only in live mode |

**Problem Statement**

`api/routes/security.py` has 6 TODO comments (lines 846, 874, 945, 975, 1009, 1036). Every endpoint in the security module returns empty results in live mode. The `SecurityGrouper` service exists at `services/security_grouper.py` but is never called from the live-mode code paths. The mock mode works well, indicating the response schemas and frontend are ready.

**Proposed Solution**

1. Wire `SecurityGrouper` into all 6 live-mode code paths:
   - `get_at_risk_devices` -- query devices with security score below threshold, sorted ascending.
   - `get_security_path_hierarchy` -- enrich path nodes with aggregated security scores from device data.
   - `get_security_by_path` -- group device security metrics by PATH hierarchy level.
   - `get_security_risk_clusters` -- use SecurityGrouper clustering to group devices by violation type.
   - `compare_security_by_path` -- compute side-by-side metrics using SecurityGrouper.
   - `get_temporal_security_clusters` -- find correlated security events within time window.
2. Add database queries to compute security scores from actual device metadata and telemetry.
3. Add integration tests using a seeded test database.

**Files Affected**

- `src/device_anomaly/api/routes/security.py` (6 code paths to implement)
- `src/device_anomaly/services/security_grouper.py` (may need additional methods)
- `src/device_anomaly/db/models.py` (may need security score column on Device)
- `tests/test_security_routes.py` (new test file)

**Dependencies**: Requires device metadata with security-relevant fields (encryption status, passcode policy, root detection) to be populated by the ingestion pipeline.

---

### FP-07: Cost Analytics Completion

| Field | Value |
|-------|-------|
| **Priority** | P2 -- Medium |
| **Effort** | Small (2-3 days) |
| **Impact** | Medium -- anomaly cost impact shows $0 in all views |

**Problem Statement**

`api/routes/costs/summary.py:137-138` hardcodes `total_anomaly_impact_mtd` and `total_anomaly_impact_ytd` to `Decimal(0)` with TODO comments to calculate from cache. The cost module has 7 sub-route files and is otherwise functional, but the headline metric (how much anomalies cost) is always zero.

**Proposed Solution**

1. Calculate anomaly cost impact by joining `anomaly_events` with `device_type_costs` and `operational_costs`:
   - MTD: sum of `estimated_impact` for anomalies in current month.
   - YTD: sum of `estimated_impact` for anomalies in current year.
2. Cache results in Redis with 15-minute TTL (keyed by tenant_id).
3. Add `estimated_impact` computation during anomaly scoring based on device type cost and anomaly severity.

**Files Affected**

- `src/device_anomaly/api/routes/costs/summary.py` (implement calculation)
- `src/device_anomaly/workers/scheduler.py` (add impact calculation to scoring pipeline)
- `src/device_anomaly/db/models.py` (add `estimated_impact` to AnomalyEvent if not present)

**Dependencies**: Cost configuration data must be populated (DeviceTypeCost, OperationalCost tables).

---

### FP-08: Dashboard Custom Attribute Aggregation

| Field | Value |
|-------|-------|
| **Priority** | P2 -- Medium |
| **Effort** | Medium (3-5 days) |
| **Impact** | Medium -- the location/utilization dashboard is empty in live mode |

**Problem Statement**

`api/routes/dashboard.py:1627-1638` has a detailed TODO describing the 6-step aggregation needed: query devices, extract custom attributes from `extra_data` JSON, group by attribute value, calculate utilization, calculate baselines, and count anomalies. In live mode the endpoint returns an empty location list.

**Proposed Solution**

1. Implement the 6 steps outlined in the existing TODO comment:
   - Query `DeviceMetadata` table for devices with custom attribute data.
   - Parse `extra_data` JSON field to extract the requested attribute (e.g., "Store", "Region").
   - Group devices by attribute value.
   - Calculate utilization as `active_devices / total_devices` per group.
   - Pull baselines from the `baselines` table or compute rolling 30-day averages.
   - Join with `anomaly_events` to get anomaly counts per group.
2. Add Redis caching with 5-minute TTL for the aggregation result.
3. Add a configuration endpoint to define which custom attributes are available for grouping.

**Files Affected**

- `src/device_anomaly/api/routes/dashboard.py` (implement aggregation)
- `src/device_anomaly/services/` (may add `dashboard_aggregator.py`)
- `src/device_anomaly/db/models.py` (verify DeviceMetadata has `extra_data` field)

**Dependencies**: Requires device metadata with custom attribute data populated from MobiControl sync.

---

### FP-09: Worker Health Monitoring

| Field | Value |
|-------|-------|
| **Priority** | P2 -- Medium |
| **Effort** | Small (1-2 days) |
| **Impact** | Medium -- operators cannot tell if the ML worker is alive or crashed |

**Problem Statement**

`api/routes/training.py:898` hardcodes `worker_available=True` with a TODO to add a worker health check. If the `ml-worker` container crashes, the API continues to report it as available, and submitted jobs silently queue forever.

**Proposed Solution**

1. Add heartbeat mechanism to `ml_worker.py`:
   - Set a Redis key `ml:worker:heartbeat` with current timestamp every 30 seconds.
   - Set TTL of 90 seconds on the key.
2. Update `training.py:898` to check the heartbeat key:
   - `worker_available = redis.exists("ml:worker:heartbeat")`.
3. Extend to all workers: add heartbeat to `scheduler` worker as well.
4. Add `GET /system/workers` endpoint listing all workers with status, last heartbeat, uptime.
5. Trigger an alert (via FP-01) if any worker heartbeat expires.

**Files Affected**

- `src/device_anomaly/workers/ml_worker.py` (add heartbeat loop)
- `src/device_anomaly/workers/scheduler.py` (add heartbeat loop)
- `src/device_anomaly/api/routes/training.py` (check heartbeat)
- `src/device_anomaly/api/routes/system_health.py` (add workers endpoint)

**Dependencies**: FP-01 (alert delivery) for worker-down notifications.

---

### FP-10: Cohort Trend Computation

| Field | Value |
|-------|-------|
| **Priority** | P3 -- Low |
| **Effort** | Small (1-2 days) |
| **Impact** | Low -- cosmetic; trend indicator shows "stable" for all cohort issues |

**Problem Statement**

`models/cohort_detector.py:360` hardcodes `trend="stable"` with a TODO to compute from historical data. The cohort detector correctly identifies device groups with similar anomaly patterns but cannot show whether the issue is worsening, improving, or stable.

**Proposed Solution**

1. Query `anomaly_events` for the cohort's devices over the past 7, 14, and 30 days.
2. Compute a simple linear regression slope on daily anomaly counts.
3. Classify: slope > +0.1 as `"worsening"`, slope < -0.1 as `"improving"`, else `"stable"`.
4. Cache per-cohort trend in Redis with 1-hour TTL.

**Files Affected**

- `src/device_anomaly/models/cohort_detector.py` (replace hardcoded trend)
- `src/device_anomaly/db/models.py` (query helpers if needed)

**Dependencies**: None.

---

## 5. Scalability Roadmap

### Phase 1: Connection Pool & Database (Q2 2026)

| Concern | Current State | Target State |
|---------|--------------|--------------|
| Pool size | Hardcoded `pool_size=5` in `db/session.py:83` and `config/settings.py:61` | Configurable via env var; default 20 for API, 5 for workers |
| Connection pooling | SQLAlchemy built-in pool only | Add PgBouncer as a sidecar container in `docker-compose.yml` for connection multiplexing |
| Read replicas | Single PostgreSQL instance | Add a read replica for analytics queries (dashboards, reports, cost aggregation) |
| Query performance | No query-level monitoring | Add `pg_stat_statements` extension; instrument slow queries via OpenTelemetry |

**Effort**: 3-5 days
**Risk**: Low -- additive changes, no breaking API modifications.

**Implementation Notes**:
- `db/session.py:81-90` -- make `pool_size` and `max_overflow` configurable via `BACKEND_DB_POOL_SIZE` and `BACKEND_DB_MAX_OVERFLOW` env vars.
- `data_access/db_connection.py:105-109` already reads `settings.pool_size` but the settings default is 5; raise defaults.
- Add PgBouncer service to `docker-compose.yml` with `transaction` pooling mode.

---

### Phase 2: Job Queue Migration (Q2/Q3 2026)

| Concern | Current State | Target State |
|---------|--------------|--------------|
| Queue implementation | Raw Redis `rpush`/`blpop` across 4 queues (16 call sites found) | Migrate to a proper task queue with retry, DLQ, priority |
| Retry logic | None -- failed jobs are logged and lost | Configurable retry with exponential backoff (3 retries default) |
| Dead letter queue | None | Failed jobs moved to DLQ after max retries for manual inspection |
| Job priority | FIFO only | Priority queues: critical scoring > scheduled training > manual requests |
| Observability | Status stored in Redis `SET` key | Structured job metadata with timestamps, duration, error traces |

**Recommended Technology**: **RQ (Redis Queue)** or **Celery with Redis broker**.

RQ is recommended over Celery for this project because:
- The project already depends on Redis (no new infrastructure).
- RQ has simpler setup and fewer moving parts (no separate beat scheduler needed -- the existing scheduler worker handles cron).
- The workload is I/O-bound (database queries, API calls) not CPU-bound, so RQ's single-threaded worker model is sufficient.
- RQ provides `rq-dashboard` for free job monitoring.

**Migration Path**:
1. Add `rq` dependency to `pyproject.toml`.
2. Refactor `ml_worker.py` to be an RQ worker class with `process_job` as the task function.
3. Replace all `rpush`/`rpop`/`blpop` calls (16 sites across `scheduler.py`, `ml_worker.py`, `training.py`, `automation.py`) with `rq.Queue.enqueue()`.
4. Add DLQ handler and retry decorator.
5. Add `rq-dashboard` as an optional Docker Compose service.

**Effort**: 5-8 days
**Risk**: Medium -- requires touching all queue producer and consumer code simultaneously. Recommend a feature flag for gradual rollout.

---

### Phase 3: ML Pipeline Evolution (Q3 2026)

| Concern | Current State | Target State |
|---------|--------------|--------------|
| Training strategy | Full retrain on 90 days of data (`auto_retrain.py:40`) | Incremental training with warm-start; full retrain weekly only |
| Model versioning | Models saved to filesystem; `MLModel` + `ModelDeployment` tables exist | Use model registry tables for versioned deployments with rollback |
| Canary deployment | None -- new model immediately replaces old | Score 10% traffic with new model, compare metrics, auto-promote or rollback |
| Feature store | Features computed on-the-fly | Cache computed features in Redis or a feature table for reuse across training and scoring |
| Model monitoring | `drift_monitor.py` exists in `streaming/` | Wire drift detection into alerting (FP-01); auto-trigger retrain on drift |

**Implementation Notes**:
- `models/governance.py` already has `cleanup_old_versions` logic, suggesting model versioning was planned.
- `models/model_registry.py` exists -- extend it with deployment status tracking.
- `pipeline/auto_retrain.py:253` computes `start_date` from `training_lookback_days` (90) -- add incremental mode that only processes new data since last training.
- `streaming/drift_monitor.py` exists but needs integration with the alert system.

**Effort**: 10-15 days
**Risk**: Medium -- ML pipeline changes require careful A/B validation to avoid regression.

---

### Phase 4: Horizontal API Scaling (Q4 2026)

| Concern | Current State | Target State |
|---------|--------------|--------------|
| API instances | Single Uvicorn process with `--reload` | Multiple Uvicorn workers behind load balancer; no `--reload` in production |
| Session state | Request context stored in `contextvars` (thread-local) | Already stateless per-request -- compatible with horizontal scaling |
| Rate limiting | In-memory `InMemoryRateLimiter` in `api/rate_limit.py` | Migrate to Redis-based rate limiting (already has Redis fallback path) |
| Static assets | Served by frontend Nginx container | Add CDN (CloudFront/Cloudflare) for static assets |
| Health checks | Basic health endpoint exists | Add readiness probe (DB + Redis connectivity) and liveness probe (process alive) |

**Implementation Notes**:
- `docker-compose.yml:214` uses `--reload` which is dev-only; production compose override should use `--workers N`.
- Rate limiter in `api/rate_limit.py:52-87` is in-memory with periodic cleanup; needs Redis backend for multi-instance consistency.
- Add Kubernetes manifests or Docker Swarm stack file for orchestrated deployment.

**Effort**: 5-8 days
**Risk**: Low -- the API is already largely stateless.

---

## 6. Priority Matrix

| Proposal | Priority | Effort | Impact | Dependencies | Phase |
|----------|----------|--------|--------|-------------|-------|
| FP-03: User Management & Auth | P0 | L (10-15d) | Critical | None | Immediate |
| FP-01: Alert Delivery | P0 | M (3-5d) | High | None | Immediate |
| FP-02: Auto-Remediation | P0 | L (8-12d) | High | FP-03 | After Auth |
| FP-04: Audit Logging | P1 | M (3-5d) | High | FP-03 | After Auth |
| FP-05: Data Retention | P1 | M (3-5d) | High | None | Q2 2026 |
| FP-06: Security Module | P1 | L (8-10d) | High | Ingestion pipeline | Q2 2026 |
| FP-09: Worker Health | P2 | S (1-2d) | Medium | FP-01 | Q2 2026 |
| FP-07: Cost Analytics | P2 | S (2-3d) | Medium | Cost data populated | Q2 2026 |
| FP-08: Dashboard Aggregation | P2 | M (3-5d) | Medium | MobiControl sync | Q2/Q3 2026 |
| FP-10: Cohort Trends | P3 | S (1-2d) | Low | None | Q3 2026 |
| Scalability Phase 1 | P1 | S (3-5d) | High | None | Q2 2026 |
| Scalability Phase 2 | P1 | M (5-8d) | High | None | Q2/Q3 2026 |
| Scalability Phase 3 | P2 | L (10-15d) | Medium | Phase 2 | Q3 2026 |
| Scalability Phase 4 | P2 | M (5-8d) | Medium | Phase 1 | Q4 2026 |

**Recommended execution order**:
1. FP-03 (Auth) and FP-01 (Alerts) in parallel -- unblock all downstream features.
2. FP-04 (Audit) and FP-09 (Worker Health) -- quick wins after auth is in place.
3. FP-02 (Remediation) -- depends on auth for approval workflows.
4. Scalability Phase 1 + FP-05 (Retention) -- database hardening sprint.
5. FP-06 (Security Module) + FP-07 (Cost) -- complete the stub endpoints.
6. Scalability Phase 2 -- job queue migration.
7. FP-08, FP-10, Phases 3-4 -- polish and scale.

---

## 7. Appendix: File Reference

Key source files referenced in this document:

| File | Path | Relevance |
|------|------|-----------|
| API entrypoint | `src/device_anomaly/api/main.py` | Auth middleware, CORS, request context |
| Dependencies | `src/device_anomaly/api/dependencies.py` | `get_current_user`, `require_role`, `get_tenant_id` |
| DB session | `src/device_anomaly/db/session.py` | Engine creation, pool_size=5 |
| DB models | `src/device_anomaly/db/models.py` | 20+ models including `User`, `AuditLog`, `AnomalyEvent` |
| Cost models | `src/device_anomaly/db/models_cost.py` | `DeviceTypeCost`, `CostAuditLog`, `CostCalculationCache` |
| Settings | `src/device_anomaly/config/settings.py` | All configuration, pool sizes, connection strings |
| Scheduler | `src/device_anomaly/workers/scheduler.py` | Cron loops, `_send_alert()`, queue consumption |
| ML Worker | `src/device_anomaly/workers/ml_worker.py` | Job processing, `blpop` queue consumer |
| Action Center | `src/device_anomaly/api/routes/action_center.py` | Remediation stub |
| Security routes | `src/device_anomaly/api/routes/security.py` | 6 stub endpoints |
| Training routes | `src/device_anomaly/api/routes/training.py` | Worker health check stub |
| Dashboard | `src/device_anomaly/api/routes/dashboard.py` | Custom attribute aggregation stub |
| Cost summary | `src/device_anomaly/api/routes/costs/summary.py` | Anomaly impact calculation stub |
| Proactive resolver | `src/device_anomaly/services/proactive_resolver.py` | Data query stubs |
| Cohort detector | `src/device_anomaly/models/cohort_detector.py` | Trend computation stub |
| Security grouper | `src/device_anomaly/services/security_grouper.py` | Exists but unused in live mode |
| Auto retrain | `src/device_anomaly/pipeline/auto_retrain.py` | 90-day lookback config |
| Rate limiter | `src/device_anomaly/api/rate_limit.py` | In-memory implementation |
| Drift monitor | `src/device_anomaly/streaming/drift_monitor.py` | Exists but not wired to alerts |
| Governance | `src/device_anomaly/models/governance.py` | Model version cleanup |
| Docker Compose | `docker-compose.yml` | 8 services, all infrastructure |
