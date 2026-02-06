# Test & Reliability Audit Report

**Project:** StellaSentinal
**Date:** 2026-02-06
**Auditor Role:** Test & Reliability Engineer
**Verdict:** CRITICAL -- The project has zero tests, zero CI/CD, zero dependency management, and significant production-safety gaps across every file.

---

## Executive Summary

The StellaSentinal codebase consists of 4 Python modules and 1 HTML dashboard with 1 JSON data file, organized across two subsystems: an "office" daemon/mission-control layer and a "projects" AI-service layer. The project is in a **pre-production prototype state** with severe reliability risks:

- **0% test coverage** -- No test files, test directories, or test framework configuration exist anywhere in the repository.
- **No CI/CD pipeline** -- No GitHub Actions, GitLab CI, Jenkinsfile, or any automation whatsoever.
- **No dependency management** -- No `requirements.txt`, `setup.py`, `pyproject.toml`, or any packaging configuration.
- **No linting or formatting** -- No `.flake8`, `pyproject.toml`, `setup.cfg`, or pre-commit hooks.
- **No `.gitignore`** -- Risk of committing sensitive or unnecessary files.
- **Hardcoded paths** -- Every Python module hardcodes `/Users/clawdbot/.openclaw/workspace`, making the project non-portable and guaranteeing failure on any other machine.
- **Minimal error handling** -- Most functions will crash on missing files, malformed data, or unexpected input.
- **No graceful shutdown** -- The daemon runs an infinite loop with no signal handling or PID management.

This codebase cannot be safely deployed to any environment in its current state.

---

## Current Test Coverage

### Test Files Found: **NONE**

| Category | Status |
|---|---|
| Unit test files | None |
| Integration test files | None |
| End-to-end test files | None |
| Performance test files | None |
| Test directories (`tests/`, `test/`) | None |
| Test configuration (`pytest.ini`, `conftest.py`, `tox.ini`) | None |
| Test framework in dependencies | No dependency file exists |
| Code coverage configuration | None |

**Effective test coverage: 0.0%**

---

## Critical Untested Paths

### 1. ScrantonCore.run() -- Infinite Daemon Loop
- **File:** `/home/user/StellaSentinal/office/core/daemon.py`, lines 18-27
- **Function:** `ScrantonCore.run()`
- **Risk:** This is the main entry point for the background daemon. It runs an infinite `while True` loop calling three methods in sequence. A failure in any method is caught by a bare `except Exception` that only prints to stdout (not logged), then continues. There is no mechanism to detect if the daemon is stuck, no watchdog, and no way to gracefully stop it.
- **Priority:** P0 -- CRITICAL

### 2. ScrantonCore.sync_fleet_health() -- Unvalidated Subprocess Execution
- **File:** `/home/user/StellaSentinal/office/core/daemon.py`, line 31
- **Function:** `ScrantonCore.sync_fleet_health()`
- **Risk:** Calls `subprocess.run()` to execute an external Python script (`bridge.py`) whose path is constructed from the hardcoded workspace. The return code is never checked. If the script does not exist, the subprocess will fail silently (output is captured but never inspected). The referenced file `bridge.py` does not exist in this repository.
- **Priority:** P0 -- CRITICAL

### 3. ScrantonCore.audit_worker_liveness() -- Unvalidated Subprocess Execution
- **File:** `/home/user/StellaSentinal/office/core/daemon.py`, line 39
- **Function:** `ScrantonCore.audit_worker_liveness()`
- **Risk:** Identical pattern to `sync_fleet_health()`. Calls `verify_worker_liveness.py` which does not exist in this repository. Silent failure guaranteed.
- **Priority:** P0 -- CRITICAL

### 4. MissionControl.load_state() -- Unprotected File Read & JSON Parse
- **File:** `/home/user/StellaSentinal/office/scripts/mission_control_daemon.py`, line 19
- **Function:** `MissionControl.load_state()`
- **Risk:** Calls `WORKFORCE_PATH.read_text()` with no error handling. Will raise `FileNotFoundError` if the file is missing, `PermissionError` if unreadable, or `json.JSONDecodeError` if the JSON is malformed. Since this is called in `__init__`, the entire MissionControl class cannot be instantiated if anything goes wrong with the file.
- **Priority:** P0 -- CRITICAL

### 5. MissionControl.save_state() -- Non-Atomic File Write
- **File:** `/home/user/StellaSentinal/office/scripts/mission_control_daemon.py`, line 22
- **Function:** `MissionControl.save_state()`
- **Risk:** Directly overwrites the workforce.json file using `write_text()`. If the process crashes mid-write (power failure, OOM kill, signal), the file will be left in a corrupt/truncated state with no backup or recovery mechanism. No file locking means concurrent access will cause data corruption.
- **Priority:** P1 -- HIGH

### 6. NextBestActionEngine.recommend() -- Unvalidated Dictionary Access
- **File:** `/home/user/StellaSentinal/projects/SOTI-Advanced-Analytics-Plus/ai-service/nba_engine.py`, lines 7-22
- **Function:** `NextBestActionEngine.recommend()`
- **Risk:** Accesses `anomaly['type']`, `anomaly['root_cause_hint']`, `anomaly['summary']`, and `anomaly['affected_cohort']` without any validation. Will raise `KeyError` if any expected key is missing. Will raise `TypeError` if `anomaly` is not a dict. Only handles two anomaly types (`connectivity` and `hardware`); all others fall through to a generic response with no logging.
- **Priority:** P1 -- HIGH

### 7. SAAPAnomalyPredictor.predict() -- Unfitted Model Risk
- **File:** `/home/user/StellaSentinal/projects/SOTI-Advanced-Analytics-Plus/ai-service/anomaly_predictor.py`, lines 20-25
- **Function:** `SAAPAnomalyPredictor.predict()`
- **Risk:** If `predict()` is called before `train()`, scikit-learn will raise `sklearn.exceptions.NotFittedError`. There is no guard, no flag tracking whether the model has been trained, and no model persistence (the model is lost when the process exits). Additionally, no validation is performed on `current_data` (could be empty, wrong shape, contain NaN/Inf values).
- **Priority:** P1 -- HIGH

### 8. SAAPAnomalyPredictor.train() -- No Input Validation
- **File:** `/home/user/StellaSentinal/projects/SOTI-Advanced-Analytics-Plus/ai-service/anomaly_predictor.py`, lines 15-18
- **Function:** `SAAPAnomalyPredictor.train()`
- **Risk:** Passes `feature_matrix` directly to `IsolationForest.fit()` with no validation. Empty arrays, arrays with NaN/Inf, single-row arrays, or non-numeric data will cause cryptic scikit-learn errors or silently produce a useless model.
- **Priority:** P1 -- HIGH

---

## CI/CD Assessment

### Current State: **NONEXISTENT**

| Component | Status |
|---|---|
| `.github/workflows/` | Missing |
| `.gitlab-ci.yml` | Missing |
| `Jenkinsfile` | Missing |
| `Makefile` | Missing |
| `Dockerfile` | Missing |
| `docker-compose.yml` | Missing |
| Pre-commit hooks | Missing (only sample hooks from git init) |
| Deployment scripts | Missing |
| Environment configuration (`.env`, `.env.example`) | Missing |

### What Is Missing

1. **Automated test execution** -- No pipeline to run tests (there are no tests to run).
2. **Linting and static analysis** -- No `flake8`, `pylint`, `mypy`, `ruff`, or `black` integration.
3. **Dependency installation step** -- No `requirements.txt` means no reproducible environment.
4. **Build verification** -- No syntax check, import check, or smoke test.
5. **Deployment automation** -- No deployment target, no staging/production configuration.
6. **Branch protection** -- No evidence of required reviews or status checks.
7. **Security scanning** -- No `bandit`, `safety`, or dependency vulnerability scanning.

The `CONTINUOUS_BUILD.md` file at `/home/user/StellaSentinal/office/CONTINUOUS_BUILD.md` describes a conceptual continuous build strategy but contains no actual CI/CD configuration -- it is a human-readable planning document only.

---

## Error Handling Gaps

### Gap 1: Bare Exception Swallowing in Daemon Loop
- **File:** `/home/user/StellaSentinal/office/core/daemon.py`, lines 25-26
- **Scenario:** Any exception from `sync_fleet_health()`, `check_trading_alpha()`, or `audit_worker_liveness()` is caught, printed to stdout, and silently discarded.
- **Impact:** Persistent failures (e.g., missing script, permission denied) will generate an infinite stream of error prints to stdout every 60 seconds with no escalation, no alerting, and no circuit-breaking. If stdout is not captured (common for background daemons), errors are completely lost.

### Gap 2: No Handling of Missing/Corrupt JSON State File
- **File:** `/home/user/StellaSentinal/office/scripts/mission_control_daemon.py`, line 19
- **Scenario:** `workforce.json` is deleted, moved, has invalid permissions, or contains invalid JSON.
- **Impact:** `MissionControl.__init__()` will crash with an unhandled exception, preventing the entire mission control system from starting. No fallback to defaults, no auto-recovery.

### Gap 3: Subprocess Failures Are Invisible
- **File:** `/home/user/StellaSentinal/office/core/daemon.py`, lines 31, 39
- **Scenario:** The spawned scripts (`bridge.py`, `verify_worker_liveness.py`) fail with a non-zero exit code, write errors to stderr, or hang indefinitely.
- **Impact:** `subprocess.run()` with `capture_output=True` stores stdout/stderr in the result object, but the result is never inspected. A hanging subprocess will block the daemon loop forever since no `timeout` parameter is set.

### Gap 4: KeyError on Malformed Anomaly Input
- **File:** `/home/user/StellaSentinal/projects/SOTI-Advanced-Analytics-Plus/ai-service/nba_engine.py`, lines 8, 15
- **Scenario:** The `anomaly` dictionary is missing `type`, `root_cause_hint`, `summary`, or `affected_cohort` keys.
- **Impact:** Unhandled `KeyError` will crash the recommendation engine. In a production pipeline, this means a single malformed telemetry record can halt all anomaly processing.

### Gap 5: NotFittedError on Cold-Start Prediction
- **File:** `/home/user/StellaSentinal/projects/SOTI-Advanced-Analytics-Plus/ai-service/anomaly_predictor.py`, lines 23-24
- **Scenario:** `predict()` is called on a freshly instantiated `SAAPAnomalyPredictor` before any `train()` call.
- **Impact:** `sklearn.exceptions.NotFittedError` will crash the predictor. There is no model persistence, so every process restart creates an unfitted model.

### Gap 6: No Timeout on Subprocess Calls
- **File:** `/home/user/StellaSentinal/office/core/daemon.py`, lines 31, 39
- **Scenario:** The called subprocess hangs (e.g., network timeout in `bridge.py`, deadlock in `verify_worker_liveness.py`).
- **Impact:** The entire daemon hangs indefinitely. The `while True` loop stops advancing. All three periodic tasks cease execution because they run sequentially.

---

## Logging & Monitoring Gaps

### Current State

| Module | Logging Mechanism | Structured? | Level Control? |
|---|---|---|---|
| `daemon.py` | `print()` to stdout | No | No |
| `mission_control_daemon.py` | `print()` to stdout | No | No |
| `nba_engine.py` | None | N/A | N/A |
| `anomaly_predictor.py` | `logging.getLogger("SAAP.ML")` | Partially | Yes (via stdlib) |

### What Is Missing

1. **Centralized logging configuration** -- Only `anomaly_predictor.py` uses the `logging` module. All other modules use `print()`, which cannot be filtered by level, routed to files, or captured by log aggregation systems.
2. **Structured logging (JSON)** -- No structured log format for machine-parseable log aggregation (e.g., ELK, Datadog, CloudWatch).
3. **Request/correlation IDs** -- No tracing mechanism to correlate events across the daemon, mission control, and AI service.
4. **Metrics collection** -- No Prometheus metrics, StatsD counters, or custom metric emission for:
   - Daemon heartbeat success/failure counts
   - Subprocess execution duration and exit codes
   - Anomaly detection counts and latency
   - NBA recommendation distribution
   - State file read/write latency
5. **Health check endpoint** -- No HTTP health endpoint for external monitoring systems to query.
6. **Alerting integration** -- No mechanism to escalate persistent errors to PagerDuty, Slack, email, or any notification channel.
7. **Audit trail** -- `MissionControl.save_state()` writes state without logging what changed or who triggered the change.
8. **Log rotation** -- If stdout is redirected to a file, there is no logrotate configuration to prevent disk exhaustion.

---

## Proposed Testing Strategy

### Unit Tests Needed

#### Module: `daemon.py` (`/home/user/StellaSentinal/office/core/daemon.py`)

| Test Case | What to Assert | Priority |
|---|---|---|
| `test_scranton_core_init_defaults` | `pulse_interval` is 60 | P2 |
| `test_sync_fleet_health_calls_subprocess` | Subprocess is called with correct path to `bridge.py`; mock `subprocess.run` | P0 |
| `test_sync_fleet_health_handles_subprocess_failure` | Non-zero exit code is detected and logged (requires code fix first) | P0 |
| `test_audit_worker_liveness_calls_subprocess` | Subprocess is called with correct path to `verify_worker_liveness.py`; mock `subprocess.run` | P0 |
| `test_audit_worker_liveness_handles_subprocess_failure` | Non-zero exit code is detected and logged | P0 |
| `test_check_trading_alpha_noop` | Verify the function runs without error (currently a pass-only stub) | P2 |
| `test_run_loop_catches_exceptions` | Inject exception in `sync_fleet_health`; verify loop continues and error is logged | P0 |
| `test_run_loop_timing` | Verify `time.sleep` is called with `pulse_interval` value; mock `time.sleep` | P1 |

#### Module: `mission_control_daemon.py` (`/home/user/StellaSentinal/office/scripts/mission_control_daemon.py`)

| Test Case | What to Assert | Priority |
|---|---|---|
| `test_load_state_valid_json` | Returns parsed dict from valid `workforce.json` fixture | P0 |
| `test_load_state_file_not_found` | Raises or handles `FileNotFoundError` gracefully | P0 |
| `test_load_state_invalid_json` | Raises or handles `json.JSONDecodeError` gracefully | P0 |
| `test_load_state_empty_file` | Handles zero-byte file without crashing | P1 |
| `test_save_state_writes_valid_json` | Written file is valid JSON matching `self.state` | P0 |
| `test_save_state_preserves_structure` | All keys from original state are preserved after save | P1 |
| `test_run_15m_pulse_calls_save_state` | Verify `save_state()` is called during pulse | P1 |
| `test_mission_control_init_loads_state` | `__init__` populates `self.state` correctly | P0 |

#### Module: `nba_engine.py` (`/home/user/StellaSentinal/projects/SOTI-Advanced-Analytics-Plus/ai-service/nba_engine.py`)

| Test Case | What to Assert | Priority |
|---|---|---|
| `test_recommend_connectivity_profile_anomaly` | Returns "Rollback Profile" action with correct target and impact | P0 |
| `test_recommend_hardware_thermal_anomaly` | Returns "Throttle Background Apps" action | P0 |
| `test_recommend_unknown_anomaly_type` | Returns "Manual Investigation Required" fallback | P0 |
| `test_recommend_missing_type_key` | Handles missing `type` key without `KeyError` crash | P0 |
| `test_recommend_missing_root_cause_hint` | Handles missing `root_cause_hint` key | P0 |
| `test_recommend_missing_summary_key` | Handles missing `summary` key | P0 |
| `test_recommend_missing_affected_cohort` | Handles missing `affected_cohort` key | P0 |
| `test_recommend_none_input` | Handles `None` input without `TypeError` crash | P1 |
| `test_recommend_empty_dict` | Handles `{}` input without crash | P1 |
| `test_recommend_connectivity_non_profile_cause` | Connectivity anomaly without "Profile" in root_cause_hint falls to next check | P1 |

#### Module: `anomaly_predictor.py` (`/home/user/StellaSentinal/projects/SOTI-Advanced-Analytics-Plus/ai-service/anomaly_predictor.py`)

| Test Case | What to Assert | Priority |
|---|---|---|
| `test_init_default_contamination` | Model initialized with `contamination=0.05` | P2 |
| `test_init_custom_contamination` | Custom contamination value is passed through | P2 |
| `test_train_valid_data` | `model.fit()` is called; no exception raised | P0 |
| `test_train_empty_array` | Handles empty `feature_matrix` gracefully | P0 |
| `test_train_nan_values` | Handles NaN in training data (should warn or raise) | P1 |
| `test_train_single_row` | Handles single-sample training data | P1 |
| `test_predict_after_train` | Returns tuple of (predictions, scores) with correct shapes | P0 |
| `test_predict_before_train` | Handles `NotFittedError` gracefully | P0 |
| `test_predict_empty_data` | Handles empty `current_data` array | P0 |
| `test_predict_mismatched_features` | Handles data with different number of features than training data | P1 |
| `test_predict_returns_correct_labels` | Predictions contain only -1 and 1 values | P1 |
| `test_logger_configured` | Logger name is "SAAP.ML" | P2 |

### Integration Tests Needed

| Test Case | Components | What to Assert | Priority |
|---|---|---|---|
| `test_daemon_to_mission_control_state_sync` | `daemon.py` + `mission_control_daemon.py` | Daemon loop triggers mission control state updates correctly | P0 |
| `test_anomaly_to_nba_pipeline` | `anomaly_predictor.py` + `nba_engine.py` | Anomalies detected by the predictor produce valid NBA recommendations | P0 |
| `test_workforce_json_round_trip` | `mission_control_daemon.py` + `workforce.json` | Load -> modify -> save -> reload preserves data integrity | P0 |
| `test_daemon_subprocess_with_real_scripts` | `daemon.py` + external scripts | Subprocess calls succeed when target scripts exist (requires fixtures) | P1 |
| `test_full_anomaly_flow` | `anomaly_predictor.py` + `nba_engine.py` | Train -> predict -> recommend produces actionable output | P0 |

### End-to-End Tests Needed

| Test Case | What to Validate | Priority |
|---|---|---|
| `test_daemon_startup_and_first_pulse` | Daemon starts, completes one full cycle of all three tasks, and logs appropriately | P0 |
| `test_mission_control_serves_dashboard` | `index.html` loads correctly and `workforce.json` is accessible (if served via HTTP) | P1 |
| `test_system_recovery_after_state_corruption` | Delete/corrupt `workforce.json`, restart system, verify recovery or clear error | P0 |
| `test_daemon_survives_subprocess_crash` | Kill a subprocess mid-execution; verify daemon continues | P0 |

### Performance Tests Needed

| Test Case | What to Measure | Priority |
|---|---|---|
| `test_anomaly_predictor_latency` | `predict()` completes within SLA for fleet-sized input (e.g., 10,000 devices x 20 features) | P1 |
| `test_anomaly_predictor_training_time` | `train()` completes within acceptable time for expected data volume | P1 |
| `test_nba_engine_throughput` | `recommend()` handles 1000+ anomalies/second | P2 |
| `test_workforce_json_large_state` | `load_state()`/`save_state()` perform acceptably with large kanban/watercooler arrays | P2 |
| `test_daemon_memory_stability` | Daemon does not leak memory over 1000+ cycles | P1 |

---

## Reliability Concerns

### 1. Daemon Hangs on Subprocess (Deadlock Risk)
- **File:** `/home/user/StellaSentinal/office/core/daemon.py`, lines 31, 39
- **Issue:** `subprocess.run()` has no `timeout` parameter. If the child process hangs (e.g., waiting for network, stuck in I/O), the daemon's main loop blocks forever. All three periodic tasks stop executing.
- **Severity:** CRITICAL

### 2. No Signal Handling / Graceful Shutdown
- **File:** `/home/user/StellaSentinal/office/core/daemon.py`, lines 18-27
- **Issue:** The `while True` loop has no mechanism to exit gracefully. `SIGTERM` (sent by `kill`, systemd, Docker) will cause an abrupt termination during `time.sleep()` or subprocess execution, potentially leaving child processes orphaned or state files corrupt.
- **Severity:** HIGH

### 3. State File Race Condition
- **File:** `/home/user/StellaSentinal/office/scripts/mission_control_daemon.py`, lines 19, 22
- **Issue:** If `daemon.py` and `mission_control_daemon.py` run concurrently (which the architecture implies), they could both read/write `workforce.json` simultaneously. There is no file locking (`fcntl.flock`, `portalocker`) or atomic write (write-to-temp-then-rename) pattern.
- **Severity:** HIGH

### 4. Hardcoded Absolute Paths Guarantee Cross-Environment Failure
- **Files:** `/home/user/StellaSentinal/office/core/daemon.py` line 7, `/home/user/StellaSentinal/office/scripts/mission_control_daemon.py` line 7
- **Issue:** Both files hardcode `WORKSPACE = Path("/Users/clawdbot/.openclaw/workspace")`. This path will not exist on any machine except the original developer's macOS system. The project will fail immediately on Linux servers, Docker containers, CI runners, or any other developer's machine.
- **Severity:** CRITICAL

### 5. No Model Persistence (ML State Loss)
- **File:** `/home/user/StellaSentinal/projects/SOTI-Advanced-Analytics-Plus/ai-service/anomaly_predictor.py`
- **Issue:** The `IsolationForest` model exists only in memory. Every process restart requires retraining. There is no `joblib.dump()`/`joblib.load()` or equivalent serialization. In production, a restart means the anomaly detection system is blind until retrained.
- **Severity:** HIGH

### 6. Silent Subprocess Failures
- **File:** `/home/user/StellaSentinal/office/core/daemon.py`, lines 31, 39
- **Issue:** `subprocess.run()` return value (including `returncode`, `stdout`, `stderr`) is discarded. Failed subprocesses produce no visible indication of failure. The daemon will appear healthy while its critical tasks are silently failing.
- **Severity:** HIGH

### 7. Unbounded Error Loop
- **File:** `/home/user/StellaSentinal/office/core/daemon.py`, lines 20-27
- **Issue:** If a persistent error occurs (e.g., filesystem permission denied), the daemon will print the same error message every 60 seconds indefinitely. There is no backoff, no circuit breaker, and no maximum retry count. Over time this could fill disk (if stdout is redirected to a file) or flood a log aggregation system.
- **Severity:** MEDIUM

### 8. Referenced Scripts Do Not Exist
- **File:** `/home/user/StellaSentinal/office/core/daemon.py`, lines 31, 39
- **Issue:** The daemon references `projects/SOTI-Advanced-Analytics-Plus/backend/bridge.py` and `office/scripts/verify_worker_liveness.py`, neither of which exists in this repository. The daemon's two primary operational tasks (`sync_fleet_health` and `audit_worker_liveness`) are guaranteed to fail.
- **Severity:** CRITICAL

---

## Infrastructure & DevOps Gaps

### Missing Configuration Files

| File | Purpose | Impact |
|---|---|---|
| `requirements.txt` or `pyproject.toml` | Python dependency declaration | Cannot reproduce environment; `numpy`, `scikit-learn` are imported but not declared |
| `.gitignore` | Exclude build artifacts, `.pyc`, `__pycache__`, `.env` | Risk of committing secrets, cache files, or large binaries |
| `Dockerfile` | Containerized deployment | No standardized deployment target |
| `docker-compose.yml` | Multi-service orchestration | Daemon + Mission Control need coordinated startup |
| `.github/workflows/*.yml` | CI/CD automation | No automated testing, linting, or deployment |
| `.env.example` | Document required environment variables | Hardcoded paths should be environment variables |
| `conftest.py` / `pytest.ini` | Test framework configuration | No test infrastructure |
| `__init__.py` (in each package dir) | Python package markers | Modules cannot be imported as packages |
| `Makefile` or `justfile` | Developer workflow automation | No standardized commands for test/lint/run |
| `systemd` unit file or `supervisord` config | Process management for daemon | No way to manage daemon lifecycle in production |
| `.flake8` / `ruff.toml` | Linting configuration | No code quality enforcement |
| `mypy.ini` or `py.typed` | Type checking configuration | No static type analysis |

### Deployment Concerns

1. **No containerization** -- The hardcoded paths and lack of dependency management make deployment to any non-developer machine impossible without manual intervention.
2. **No process supervisor** -- The daemon (`daemon.py`) has no systemd unit, no supervisord config, and no Docker entrypoint. If it crashes, nothing restarts it.
3. **No secrets management** -- While no secrets are currently in the code, there is no `.env` pattern, no vault integration, and no mechanism for injecting credentials.
4. **No database** -- State is stored in a flat JSON file (`workforce.json`) with no backup, no migration strategy, and no concurrency control.
5. **No network configuration** -- The dashboard (`index.html`) is a static file with no serving mechanism, no CORS configuration, and no API backend to fetch live data.
6. **No resource limits** -- The daemon has no memory limits, no CPU limits, and no disk usage monitoring.

---

## Priority Action Items

Listed in order of urgency. Items 1-5 are blockers for any form of deployment.

### 1. Remove Hardcoded Paths and Introduce Environment Configuration (P0)
- Replace `Path("/Users/clawdbot/.openclaw/workspace")` in `daemon.py` (line 7) and `mission_control_daemon.py` (line 7) with environment variable reads (e.g., `os.environ.get("STELLA_WORKSPACE", fallback_default)`).
- Create a `.env.example` documenting all required variables.

### 2. Create `requirements.txt` with Pinned Dependencies (P0)
- At minimum: `numpy`, `scikit-learn`, `joblib` (for model persistence).
- Pin exact versions for reproducibility (e.g., `scikit-learn==1.4.2`).

### 3. Add a Minimal Test Suite with pytest (P0)
- Create `tests/` directory with `conftest.py`.
- Write unit tests for `NextBestActionEngine.recommend()` (easiest, highest value-to-effort ratio -- pure function with no I/O).
- Write unit tests for `SAAPAnomalyPredictor` train/predict cycle.
- Write unit tests for `MissionControl.load_state()` with fixture files (valid JSON, invalid JSON, missing file).
- Target: Cover all critical paths identified in this report.

### 4. Set Up CI/CD Pipeline (P0)
- Create `.github/workflows/ci.yml` with: install dependencies, run `pytest`, run `flake8` or `ruff`.
- Add branch protection requiring CI pass before merge.

### 5. Add Error Handling to All File I/O and Subprocess Calls (P0)
- `mission_control_daemon.py`: Wrap `load_state()` in try/except for `FileNotFoundError`, `json.JSONDecodeError`, `PermissionError`.
- `daemon.py`: Check `subprocess.run()` return codes, add `timeout` parameter, log stdout/stderr on failure.
- `nba_engine.py`: Use `.get()` for dictionary access with defaults; validate input structure.

### 6. Replace `print()` with `logging` Module Throughout (P1)
- Configure root logger with format including timestamp, level, module.
- Replace all `print()` calls in `daemon.py` and `mission_control_daemon.py`.
- Add logging to `nba_engine.py` (currently has none).

### 7. Add Graceful Shutdown to Daemon (P1)
- Register `signal.signal(signal.SIGTERM, handler)` and `signal.signal(signal.SIGINT, handler)` to set a shutdown flag.
- Replace `while True` with `while not self.shutdown_requested`.
- Add `timeout` to all `subprocess.run()` calls.

### 8. Implement Model Persistence for AnomalyPredictor (P1)
- Add `save_model()` and `load_model()` methods using `joblib.dump()`/`joblib.load()`.
- Add an `is_fitted` property to guard `predict()` calls.
- Store model artifacts in a configurable directory.

### 9. Add `.gitignore` and `__init__.py` Files (P1)
- Create `.gitignore` excluding `__pycache__/`, `*.pyc`, `.env`, `*.pkl`, `.DS_Store`, `node_modules/`, `*.egg-info/`.
- Add `__init__.py` to `office/`, `office/core/`, `office/scripts/`, `projects/SOTI-Advanced-Analytics-Plus/`, and `projects/SOTI-Advanced-Analytics-Plus/ai-service/` to enable proper Python imports.

### 10. Implement Atomic State File Writes with Locking (P2)
- Replace direct `write_text()` with a write-to-temp-file-then-rename pattern in `MissionControl.save_state()`.
- Add `fcntl.flock()` or use `portalocker` for cross-process file locking.
- Create periodic backups of `workforce.json` before overwrite.

---

*End of Test & Reliability Audit Report*
