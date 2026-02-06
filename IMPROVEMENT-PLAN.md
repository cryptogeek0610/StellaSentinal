# Master Improvement Plan

**Project**: StellaSentinal — AI-Powered Anomaly Detection for Enterprise Device Fleets
**Date**: 2026-02-06
**Branch**: `claude/repo-audit-baseline-TJLZI`

---

## Overview

This plan synthesizes findings from four parallel audits:
- **[AUDIT-REPORT.md](AUDIT-REPORT.md)** — Code quality, security, error handling, performance
- **[FEATURE-PROPOSALS.md](FEATURE-PROPOSALS.md)** — Architecture analysis, missing features, scalability
- **[UI-IMPROVEMENTS.md](UI-IMPROVEMENTS.md)** — Accessibility, responsiveness, component design
- **[TEST-REPORT.md](TEST-REPORT.md)** — Test coverage, CI/CD, reliability

Items are ranked by **Impact x Effort** and grouped into execution phases.

---

## Already Completed (This Audit)

| # | Fix | Impact | Status |
|---|-----|--------|--------|
| 1 | Remove `scranton_bridge` import (app startup blocker) | BLOCKER | DONE |
| 2 | Fix CostManagement.tsx build break (6 TS errors) | CRITICAL | DONE |
| 3 | Enable full test suite in CI (219 tests, was 1) | CRITICAL | DONE |
| 4 | Fix 9 F821 undefined-name errors | HIGH | DONE (prev) |
| 5 | Fix 164 ruff lint errors + format 219 files | HIGH | DONE (prev) |
| 6 | Fix 6 ESLint warnings | MEDIUM | DONE (prev) |
| 7 | SQL injection fixes (3 files) | CRITICAL | DONE (prev) |
| 8 | Input validation for attribute names | HIGH | DONE (prev) |
| 9 | Structured error responses + correlation IDs | HIGH | DONE (prev) |
| 10 | Rate limiting middleware | HIGH | DONE (prev) |
| 11 | N+1 query fixes + composite indexes | HIGH | DONE (prev) |
| 12 | Route-level code splitting (59% smaller bundle) | MEDIUM | DONE (prev) |
| 13 | React.memo on 7 list components | MEDIUM | DONE (prev) |
| 14 | Pre-commit hooks | MEDIUM | DONE (prev) |
| 15 | Loading skeletons + error retry | MEDIUM | DONE (prev) |
| 16 | Breadcrumb navigation | LOW | DONE (prev) |
| 17 | Accessibility aria-labels | MEDIUM | DONE (prev) |
| 18 | Redis graceful degradation | MEDIUM | DONE (prev) |
| 19 | LLM fallback notification banner | MEDIUM | DONE (prev) |
| 20 | Structured API return types | MEDIUM | DONE (prev) |
| 21 | BatteryForecastTab component extraction | LOW | DONE (prev) |

---

## Priority Matrix

```
IMPACT
  ^
  |  ████████ P1: Critical fixes & test CI
  |  ████████ (DONE)
  |
  |  ████   P2: Exception         ████ P3: Test coverage
  |  ████   handling +            ████ for API routes
  |  ████   security mocks        ████
  |
  |  ███ P4: Pagination      ███ P5: Alert delivery
  |  ███ for unbounded       ███ + auto-remediation
  |  ███ queries
  |
  |  ██ P6: Frontend    ██ P7: Component    ██ P8: Accessibility
  |  ██ error UX        ██ splitting        ██ + responsive
  |
  |  █ P9: Coverage   █ P10: Job queue    █ P11: DB replicas
  |  █ gating         █ migration         █ + ML versioning
  |
  +────────────────────────────────────────────> EFFORT
     Low                                   High
```

---

## Phase 1: Reliability & Security Hardening (1-2 weeks)

### 1.1 Standardize Exception Handling [HIGH]
- **Source**: [AUDIT-REPORT.md](AUDIT-REPORT.md) §2.2
- **Problem**: 64 broad `except Exception:` blocks silently swallowing errors
- **Key files**:
  - `database/connection.py:295` — `pass` on connection error
  - `db/session.py:99,143,159,197` — 4 silent failures
  - `services/correlation_service.py:452,585,682,708`
  - `data_access/watermark_store.py:105,274,320,357,530`
- **Fix**: Replace with specific exception types, add structured logging with request_id
- **Impact**: HIGH — silent failures cause phantom data loss
- **Effort**: 2 days

### 1.2 Replace Mock Security Endpoints [HIGH]
- **Source**: [AUDIT-REPORT.md](AUDIT-REPORT.md) §2.1
- **Problem**: 6 endpoints in `security.py:846-1036` return hardcoded mock data
- **Fix**: Either implement real queries or mark as "preview" in OpenAPI + return 501
- **Impact**: HIGH — users see fake security posture data
- **Effort**: 2-3 days (implement) or 2 hours (mark as preview)

### 1.3 Sanitize Error Message Leakage [MEDIUM]
- **Source**: [AUDIT-REPORT.md](AUDIT-REPORT.md) §2.3
- **Problem**: `dashboard.py:51-106` `_parse_connection_error()` leaks hostnames/connection strings
- **Fix**: Return generic message with request_id, log details server-side
- **Impact**: MEDIUM — security information disclosure
- **Effort**: 1 hour

### 1.4 Fix Flaky Tests [HIGH]
- **Source**: [TEST-REPORT.md](TEST-REPORT.md) §6
- **Problem**: 3 patterns using `time.sleep()` and `datetime.now()` fail intermittently
- **Fix**: Add `freezegun`, mock time in rate limit / schema discovery / cohort tests
- **Impact**: HIGH — CI reliability
- **Effort**: 2 hours

---

## Phase 2: Test Coverage (2-3 weeks)

### 2.1 API Route Tests — Priority 1 [HIGH]
- **Source**: [TEST-REPORT.md](TEST-REPORT.md) §5
- **Routes**: `investigation.py` (1,472 LOC), `security.py` (1,044 LOC), `training.py` (972 LOC)
- **Target**: 1 test per endpoint, happy path + 1 error case
- **Impact**: HIGH — core user workflows completely untested
- **Effort**: 1 week (~37 tests)

### 2.2 Service Layer Tests [HIGH]
- **Source**: [TEST-REPORT.md](TEST-REPORT.md) §5
- **Services**: `correlation_service.py` (936 LOC), `ml_baseline_service.py` (753 LOC), `location_sync.py` (661 LOC)
- **Target**: Key method coverage
- **Impact**: HIGH — data integrity logic untested
- **Effort**: 1 week (~20 tests)

### 2.3 Add Coverage Gating to CI [MEDIUM]
- **Source**: [TEST-REPORT.md](TEST-REPORT.md) §8.1
- **Fix**: `pytest --cov=device_anomaly --cov-fail-under=50`
- **Target**: Start at 50%, increase to 70% over 2 months
- **Impact**: MEDIUM — prevents coverage regression
- **Effort**: 1 hour

---

## Phase 3: Performance & Data Safety (1-2 weeks)

### 3.1 Add Pagination to Unbounded Queries [HIGH]
- **Source**: [AUDIT-REPORT.md](AUDIT-REPORT.md) §4.1
- **Problem**: 10+ endpoints call `.all()` on potentially large tables
- **Files**:
  - `anomalies.py:143`, `insights.py:464,898`
  - `device_actions.py:585`, `baselines.py:149,474`
  - `dashboard.py:297,1122,1270`
  - `costs/forecasts.py:51,84,110,274`
- **Fix**: Add `limit()`/`offset()` with default page size of 100
- **Impact**: HIGH — OOM risk in production with large datasets
- **Effort**: 2 days

### 3.2 Frontend Error Feedback [MEDIUM]
- **Source**: [UI-IMPROVEMENTS.md](UI-IMPROVEMENTS.md) §4
- **Problem**: 8 mutations in CostManagement.tsx have no `onError`, Fleet.tsx logs to console only
- **Fix**: Add toast notifications for all mutation errors
- **Impact**: MEDIUM — users don't know when saves fail
- **Effort**: 1 day

### 3.3 Query Invalidation Optimization [LOW-MEDIUM]
- **Source**: [UI-IMPROVEMENTS.md](UI-IMPROVEMENTS.md) §3
- **Problem**: 43 instances of overly broad invalidation (e.g., `['dashboard']`)
- **Fix**: Use specific keys like `['dashboard', 'stats']`
- **Impact**: LOW-MEDIUM — unnecessary re-fetches
- **Effort**: 1 day

---

## Phase 4: Architecture & Features (3-4 weeks)

### 4.1 Alert Delivery System [HIGH]
- **Source**: [FEATURE-PROPOSALS.md](FEATURE-PROPOSALS.md) §5.1
- **Problem**: Alerts only go to `logger.warning()` — no email or Slack
- **Fix**: SendGrid email integration + Slack webhook
- **Impact**: HIGH — core product feature missing
- **Effort**: 1 week

### 4.2 Auto-Remediation Execution [HIGH]
- **Source**: [FEATURE-PROPOSALS.md](FEATURE-PROPOSALS.md) §5.2
- **Problem**: "Execute fix" button is a no-op
- **Fix**: Wire to MobiControl API for device actions
- **Impact**: HIGH — key differentiating feature
- **Effort**: 1 week

### 4.3 User Management API [MEDIUM]
- **Source**: [FEATURE-PROPOSALS.md](FEATURE-PROPOSALS.md) §5.3
- **Problem**: Header-based auth only, no user CRUD
- **Fix**: JWT auth, user management endpoints, role assignment UI
- **Impact**: MEDIUM — security, multi-tenancy
- **Effort**: 2 weeks

### 4.4 General Audit Logging [MEDIUM]
- **Source**: [FEATURE-PROPOSALS.md](FEATURE-PROPOSALS.md) §5.4
- **Problem**: Only cost changes logged
- **Fix**: General `AuditLog` model, log all write operations
- **Impact**: MEDIUM — compliance, debugging
- **Effort**: 1 week

---

## Phase 5: UI/UX Polish (2-3 weeks)

### 5.1 Component Splitting [MEDIUM]
- **Source**: [UI-IMPROVEMENTS.md](UI-IMPROVEMENTS.md) §1
- **Files to split**:
  - `Dashboard.tsx` (1,340 lines) → KPI cards, activity feed, chart sections
  - `SecurityPosture.tsx` (1,094 lines) → 5 tab components
  - `CostManagement.tsx` → NFF, Alerts, History tabs
  - `Sidebar.tsx` (605 lines) → nav items, search, badge
- **Impact**: MEDIUM — maintainability, code review efficiency
- **Effort**: 3 days

### 5.2 Accessibility Fixes [MEDIUM]
- **Source**: [UI-IMPROVEMENTS.md](UI-IMPROVEMENTS.md) §2
- **Issues**:
  - 40+ SVGs without `aria-hidden`
  - Color contrast: `text-slate-600` on `bg-slate-800` (WCAG fail)
  - Form inputs without associated `<label>` elements
- **Impact**: MEDIUM — WCAG compliance, usability
- **Effort**: 2 days

### 5.3 Responsive Design [LOW]
- **Source**: [UI-IMPROVEMENTS.md](UI-IMPROVEMENTS.md) §5
- **Issues**: Hardcoded `max-h-[XXXpx]`, no tablet breakpoints
- **Fix**: Dynamic heights, tablet-specific responsive classes
- **Impact**: LOW — enterprise users primarily on desktop
- **Effort**: 2 days

### 5.4 Frontend Test Infrastructure [MEDIUM]
- **Source**: [TEST-REPORT.md](TEST-REPORT.md) §7
- **Fix**: Install `@testing-library/react`, add vitest tests for utilities + key components
- **Impact**: MEDIUM — zero frontend test coverage
- **Effort**: 1 week

---

## Phase 6: Scalability (Month 2+)

### 6.1 Job Queue Migration [LARGE]
- **Source**: [FEATURE-PROPOSALS.md](FEATURE-PROPOSALS.md) §7.1
- **Problem**: Raw Redis `rpush/lpop` — no retry, no DLQ, no priority
- **Fix**: Migrate to Celery or RQ
- **Effort**: 2-3 weeks

### 6.2 Database Scaling [LARGE]
- **Source**: [FEATURE-PROPOSALS.md](FEATURE-PROPOSALS.md) §7.2
- **Problem**: Single PostgreSQL, pool size hardcoded at 5
- **Fix**: PgBouncer, read replicas, configurable pool
- **Effort**: 3-4 weeks

### 6.3 ML Pipeline Improvements [LARGE]
- **Source**: [FEATURE-PROPOSALS.md](FEATURE-PROPOSALS.md) §7.3
- **Problem**: Full retrain daily on 90 days data
- **Fix**: Model versioning, canary deployment, incremental updates
- **Effort**: 4-6 weeks

### 6.4 Data Retention Policies [MEDIUM]
- **Source**: [FEATURE-PROPOSALS.md](FEATURE-PROPOSALS.md) §5.5
- **Problem**: Data grows unbounded
- **Fix**: Configurable retention, automated cleanup, archival
- **Effort**: 1 week

---

## Effort Summary

| Phase | Duration | Items | Cumulative |
|-------|----------|-------|------------|
| Already Done | — | 21 fixes | 21 |
| Phase 1: Reliability | 1-2 weeks | 4 items | 25 |
| Phase 2: Test Coverage | 2-3 weeks | 3 items | 28 |
| Phase 3: Performance | 1-2 weeks | 3 items | 31 |
| Phase 4: Features | 3-4 weeks | 4 items | 35 |
| Phase 5: UI/UX | 2-3 weeks | 4 items | 39 |
| Phase 6: Scalability | 8+ weeks | 4 items | 43 |

---

## Metrics to Track

| Metric | Current | Phase 2 Target | Phase 4 Target |
|--------|---------|----------------|----------------|
| Backend test coverage | ~25% | 55% | 70% |
| Frontend test coverage | 0% | 5% | 25% |
| ruff errors | 0 | 0 | 0 |
| TypeScript errors | 0 | 0 | 0 |
| Broad `except Exception` | 64 | <20 | <5 |
| Endpoints with pagination | ~0 | 10+ | All |
| Mock endpoints in prod | 6 | 0 | 0 |
| Avg response time (p95) | Unknown | Measured | <500ms |

---

## Quick Wins (< 1 day each)

1. Sanitize `_parse_connection_error()` — 1 hour
2. Add `pytest-cov` to CI — 1 hour
3. Mark mock security endpoints as preview — 2 hours
4. Fix 3 flaky test patterns — 2 hours
5. Add toast notifications for mutation errors — 4 hours
6. Optimize query invalidation keys — 4 hours
