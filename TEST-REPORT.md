# Test & Reliability Report

**Project**: StellaSentinal — AI-Powered Anomaly Detection
**Date**: 2026-02-06
**Audit Branch**: `claude/repo-audit-baseline-TJLZI`

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Test Inventory](#current-test-inventory)
3. [CI/CD Pipeline](#cicd-pipeline)
4. [Test Coverage Matrix](#test-coverage-matrix)
5. [Coverage Gap Analysis](#coverage-gap-analysis)
6. [Flaky Test Analysis](#flaky-test-analysis)
7. [E2E & Frontend Test Status](#e2e--frontend-test-status)
8. [Recommendations](#recommendations)
9. [Testing Roadmap](#testing-roadmap)

---

## Executive Summary

| Metric | Value | Status |
|--------|-------|--------|
| Total test files | 27 | — |
| Total test functions | 219 | — |
| Tests running in CI | **219** (was 1) | FIXED |
| API route modules tested | 5 of 23 (22%) | NEEDS WORK |
| Service modules tested | 3 of 15 (20%) | NEEDS WORK |
| Frontend component tests | 0 | CRITICAL GAP |
| E2E tests | 2 (no config) | NOT RUNNING |
| Coverage gating in CI | None | RECOMMENDED |
| Flaky test patterns | 3 identified | NEEDS FIX |

**Key win this audit**: Full test suite (`pytest tests/ -v --timeout=120 -x`) now runs in CI. Previously only `test_imports.py` (1 test) executed.

---

## Current Test Inventory

### 27 Test Files — 219 Functions

| Test File | Tests | Category |
|-----------|-------|----------|
| `test_enhanced_anomaly_detection.py` | 30 | ML features, ensemble, explainer, pipeline |
| `test_dl_models.py` | 28 | VAE, autoencoder, DL preprocessor, ONNX export |
| `test_data_discovery_hardening.py` | 24 | Watermarks, event ID, dedup, pagination, throttling |
| `test_schema_discovery.py` | 21 | Schema discovery, canonical events, data profiler |
| `test_streaming_parity.py` | 15 | Batch vs. streaming feature parity |
| `test_onnx_integration.py` | 10 | ONNX export, inference engines, metrics |
| `test_anomaly_drift_fix.py` | 10 | Temporal feature stability, data freshness |
| `test_input_validation.py` | 9 | Attribute name sanitization (SQL injection) |
| `test_dl_sklearn_parity.py` | 9 | DL vs. sklearn detection quality parity |
| `test_request_context.py` | 8 | Context var lifecycle, frozen dataclasses |
| `test_error_responses.py` | 8 | Structured error envelope, status codes |
| `test_rate_limit.py` | 7 | Sliding window counter, burst, expiry, cleanup |
| `test_streaming_state.py` | 6 | Snapshot roundtrip, corruption, size limits |
| `test_response_models.py` | 6 | Pydantic response model serialization |
| `test_pipeline_smoke.py` | 5 | End-to-end pipeline smoke test |
| `test_cost_endpoints.py` | 4 | Cost alert CRUD, battery forecast, NFF |
| `test_baseline_store.py` | 3 | Baseline resolution, production preference |
| `test_api_contracts.py` | 3 | Baseline/anomaly endpoint contracts |
| `test_anomaly_detection.py` | 2 | Hybrid anomaly detection, heuristics |
| `test_cohort_stats.py` | 2 | Cohort z-scores, streaming parity |
| `test_streaming_drift.py` | 2 | Drift monitor metrics, missing features |
| `test_ingestion_pipeline.py` | 2 | Dry run ingestion, disabled MC handling |
| `test_imports.py` | 1 | Module import verification (CI canary) |
| `test_streaming_contracts.py` | 1 | Streaming status endpoint |
| `test_results_schema_migration.py` | 1 | Schema migration columns |
| `test_mc_timeseries_stream.py` | 1 | MC timeseries watermark stop |
| `test_dashboard_stats.py` | 1 | Dashboard stats from trained config |

### Test Categories Breakdown

| Category | Test Count | % of Total |
|----------|-----------|------------|
| ML / Deep Learning | 77 | 35% |
| Data Pipeline / Discovery | 43 | 20% |
| Streaming | 24 | 11% |
| API Infrastructure | 38 | 17% |
| API Endpoints | 8 | 4% |
| Integration / Smoke | 8 | 4% |
| Other | 21 | 9% |

**Observation**: ML and data pipeline tests are well-covered (55% of tests). API endpoint coverage is severely lacking (4% of tests for 75% of the codebase surface area).

---

## CI/CD Pipeline

```
┌─────────────────────────────────────────────────┐
│                 GitHub Actions CI                │
├─────────────────────┬───────────────────────────┤
│    Backend (Python)  │    Frontend (TS/React)    │
├─────────────────────┼───────────────────────────┤
│ 1. checkout          │ 1. checkout              │
│ 2. setup python 3.11 │ 2. setup node 20         │
│ 3. pip install deps  │ 3. npm ci                │
│ 4. ruff check        │ 4. tsc --noEmit          │
│ 5. ruff format check │ 5. eslint (max-warn 10)  │
│ 6. pytest tests/     │ 6. vite build            │
│    -v --timeout=120  │                           │
│    -x (fail fast)    │ ❌ No vitest tests       │
│                      │ ❌ No playwright tests    │
│ ❌ No coverage gate  │                           │
└─────────────────────┴───────────────────────────┘
```

### CI Dependencies Installed

```
pip install ruff pytest pytest-asyncio pytest-timeout
pip install pydantic pydantic-settings fastapi sqlalchemy pyyaml pandas numpy
```

**Missing from CI**: `scikit-learn`, `torch`, `onnx`, `redis` — some tests may be skipped if they import these.

---

## Test Coverage Matrix

### API Route Modules (23 modules, ~18,800 LOC)

| Module | LOC | Tested | Test File | Priority |
|--------|-----|--------|-----------|----------|
| `investigation.py` | 1,472 | No | — | P1 |
| `data_discovery.py` | 1,319 | Partial | `test_data_discovery_hardening.py` | — |
| `automation.py` | 1,128 | No | — | P2 |
| `security.py` | 1,044 | No | — | P1 |
| `baselines.py` | 1,040 | Partial | `test_api_contracts.py` | P2 |
| `training.py` | 972 | No | — | P1 |
| `insights.py` | 969 | No | — | P2 |
| `dashboard.py` | 1,894 | Partial | `test_dashboard_stats.py` | P2 |
| `cross_device.py` | 915 | No | — | P3 |
| `correlations.py` | 875 | No | — | P2 |
| `network.py` | 852 | No | — | P3 |
| `device_actions.py` | 761 | No | — | P2 |
| `setup.py` | 743 | No | — | P3 |
| `location_intelligence.py` | 646 | No | — | P3 |
| `events_alerts.py` | 640 | No | — | P3 |
| `temporal.py` | 593 | No | — | P3 |
| `system_health.py` | 572 | No | — | P3 |
| `llm_settings.py` | 461 | No | — | P3 |
| `action_center.py` | 446 | No | — | P3 |
| `streaming.py` | 442 | Partial | `test_streaming_contracts.py` | — |
| `data_quality.py` | 425 | No | — | P3 |
| `anomalies.py` | 352 | Partial | `test_api_contracts.py` | — |
| `devices.py` | 200 | Partial | `test_input_validation.py` | — |

### Service Modules (15 modules, ~8,100 LOC)

| Module | LOC | Tested | Priority |
|--------|-----|--------|----------|
| `correlation_service.py` | 936 | No | P1 |
| `ml_baseline_service.py` | 753 | No | P1 |
| `location_sync.py` | 661 | No | P1 |
| `anomaly_grouper.py` | 652 | No | P2 |
| `data_reconciliation.py` | 618 | No | P2 |
| `security_grouper.py` | 547 | No | P2 |
| `ingestion_pipeline.py` | 557 | Partial | — |
| `ingestion_orchestrator.py` | 513 | Partial | — |
| `device_metadata_sync.py` | 438 | No | P3 |
| `proactive_resolver.py` | 417 | No | P3 |
| `device_grouper.py` | 334 | No | P3 |
| `battery_status_sync.py` | 327 | No | P3 |
| `path_builder.py` | 270 | No | P3 |
| `device_group_service.py` | 245 | No | P3 |
| `ingestion_metrics.py` | 817 | Partial | — |

---

## Coverage Gap Analysis

### Risk-Weighted Priority

| Area | Untested LOC | Risk | Impact | Priority |
|------|-------------|------|--------|----------|
| Investigation routes | 1,472 | HIGH | Core user workflow | P1 |
| Security routes | 1,044 | HIGH | 6 mock endpoints in prod | P1 |
| Training routes | 972 | HIGH | ML pipeline orchestration | P1 |
| Correlation service | 936 | HIGH | Cross-device analysis | P1 |
| ML baseline service | 753 | HIGH | Anomaly threshold computation | P1 |
| Automation routes | 1,128 | MEDIUM | Scheduled jobs | P2 |
| Insights routes | 969 | MEDIUM | Analytics | P2 |
| Location sync | 661 | MEDIUM | Data integrity | P2 |
| Remaining routes (10) | ~5,800 | LOW-MEDIUM | Various features | P3 |
| Remaining services (8) | ~3,000 | LOW | Supporting services | P3 |

**Total untested backend code**: ~16,700 LOC (64% of total backend)

---

## Flaky Test Analysis

### Pattern 1: `time.sleep()` in Rate Limit Tests
- **File**: `tests/test_rate_limit.py:55-65`
- **Issue**: `time.sleep(1.1)` to test window expiry — unreliable on slow CI runners
- **Fix**: Mock `time.monotonic()` instead of real sleep

### Pattern 2: Time-Window Assertions in Schema Discovery
- **File**: `tests/test_schema_discovery.py:364,416`
- **Issue**: Asserts `23 <= hours <= 25` for 24-hour windows — can fail near hour boundaries
- **Fix**: Use `freezegun` to freeze time, assert exact values

### Pattern 3: `datetime.now()` in Cohort Stats
- **File**: `tests/test_cohort_stats.py:70`
- **Issue**: Uses `datetime.now(UTC)` — results vary by execution time
- **Fix**: Use fixed timestamps via `freezegun` or test fixtures

### Estimated Failure Rate
- Pattern 1: ~5% on slow CI runners
- Pattern 2: ~2% (near hour boundaries)
- Pattern 3: ~1% (rare edge case)

---

## E2E & Frontend Test Status

### Frontend Tests
- `frontend/vitest.config.ts` exists but **0 test files**
- `@testing-library/react` **not installed** (not in `package.json`)
- No component tests, hook tests, or utility tests

### E2E Tests
- `e2e/anomaly-workflow.spec.ts` has **2 Playwright test cases**
- **No Playwright config** (`playwright.config.ts` missing)
- **Not wired to CI** — tests cannot run
- Tests cover: anomaly list loading, investigation drill-down

### Shared Test Fixtures (conftest.py)
Well-structured with:
- `api_client` — FastAPI TestClient with isolated DB
- `sample_telemetry_df` — synthetic telemetry data
- `dummy_watermark_store` — in-memory watermark store
- `baselines_file` — temp baselines JSON
- Custom markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`

---

## Recommendations

### 1. Add Coverage Metrics to CI [HIGH, 1 hour]
```yaml
- name: Run tests with coverage
  run: |
    pip install pytest-cov
    export PYTHONPATH=src
    python -m pytest tests/ -v --timeout=120 -x \
      --cov=device_anomaly --cov-report=term-missing \
      --cov-fail-under=50
```
Start at 50% threshold, increase to 70% over 2 months.

### 2. Fix Flaky Tests [HIGH, 2 hours]
- Add `freezegun` to test dependencies
- Replace `time.sleep()` with mocked time in rate limit tests
- Replace `datetime.now()` with frozen time in cohort/schema tests

### 3. API Route Test Generation [HIGH, 2 weeks]
- Leverage existing `api_client` fixture in conftest.py
- Priority order: investigation → security → training → automation → correlations
- Target: 1 test per endpoint, covering happy path + 1 error case
- Estimated: ~150 new tests

### 4. Frontend Test Infrastructure [MEDIUM, 1 week]
- Install `@testing-library/react`, `@testing-library/jest-dom`
- Add `vitest` test script to `package.json`
- Start with utility functions (formatters, validators)
- Add component tests for high-traffic pages (Dashboard, Fleet, AnomalyList)

### 5. E2E Test Setup [MEDIUM, 1 week]
- Install and configure Playwright
- Add `playwright.config.ts` pointing to dev server
- Wire `npx playwright test` into CI (separate job)
- Expand from 2 existing tests to cover critical user flows

---

## Testing Roadmap

### Phase 1: Foundation (Week 1) — 5 person-days
- [ ] Add `pytest-cov` with 50% threshold to CI
- [ ] Fix 3 flaky test patterns with `freezegun`
- [ ] Install frontend test infrastructure (vitest + testing-library)
- [ ] Add 10 utility function tests for frontend

### Phase 2: Critical API Coverage (Week 2-3) — 10 person-days
- [ ] `investigation.py` — 15 tests (15 endpoints)
- [ ] `security.py` — 12 tests (12 endpoints)
- [ ] `training.py` — 10 tests (10 endpoints)
- [ ] `correlation_service.py` — 8 tests (key methods)
- [ ] `ml_baseline_service.py` — 6 tests (key methods)

### Phase 3: Secondary Coverage (Week 4-5) — 10 person-days
- [ ] `automation.py` — 10 tests
- [ ] `correlations.py` — 8 tests
- [ ] `insights.py` — 8 tests
- [ ] `device_actions.py` — 6 tests
- [ ] Remaining service modules — 15 tests

### Phase 4: Frontend & E2E (Week 6-7) — 10 person-days
- [ ] Dashboard component tests — 8 tests
- [ ] AnomalyList component tests — 5 tests
- [ ] CostManagement component tests — 5 tests
- [ ] Playwright E2E setup + 10 critical flow tests
- [ ] Add Playwright to CI

### Target Metrics
| Phase | Backend Coverage | Frontend Coverage | E2E Flows |
|-------|-----------------|-------------------|-----------|
| Current | ~25% | 0% | 0 |
| After Phase 1 | 30% + gating | 5% | 0 |
| After Phase 2 | 55% | 5% | 0 |
| After Phase 3 | 70% | 5% | 0 |
| After Phase 4 | 70% | 25% | 10 flows |
