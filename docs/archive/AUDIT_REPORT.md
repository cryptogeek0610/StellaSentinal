# AUDIT_REPORT

## Findings

### Critical
- None found in this pass.

### High
- None remaining in this pass.

### Medium
- None remaining in this pass.

### Low
- Feature parity depends on streaming buffer history length; if a device has fewer than the max rolling window events, rolling features are partially missing and imputed. This is now logged, but still a source of lower-confidence early scoring (`src/device_anomaly/streaming/feature_computer.py:247-335`).

## Changes Applied This Pass
- Baseline resolution now prefers production data-driven baselines and falls back to legacy artifacts, with a versioned schema adapter for `/api/baselines/*` (`src/device_anomaly/models/baseline_store.py:1-170`, `src/device_anomaly/api/routes/baselines.py:28-210`).
- Cohort z-score stats are computed from train-only data, persisted to `cohort_stats.json`, and applied to validation, batch inference, and streaming (`src/device_anomaly/features/cohort_stats.py:1-206`, `src/device_anomaly/pipeline/training.py:229-556`, `src/device_anomaly/streaming/feature_computer.py:256-315`).
- Dashboard isolation-forest stats now surface trained config from metadata and include defaults for comparison (`src/device_anomaly/api/routes/dashboard.py:1050-1202`).
- Streaming feature computation now uses the canonical `DeviceFeatureBuilder` with stored feature spec + normalization norms for parity with batch scoring; incremental mode remains available via `STREAMING_FEATURE_MODE` (`src/device_anomaly/streaming/feature_computer.py:99-375`, `src/device_anomaly/features/device_features.py:1-640`, `src/device_anomaly/api/routes/streaming.py:95-142`).
- Training metadata now stores `feature_spec` and `feature_norms` for reproducible inference across batch and streaming (`src/device_anomaly/pipeline/training.py:235-560`).
- Added tests for streaming vs batch feature parity, streaming state rehydration, and API contracts for baselines/anomalies (`tests/test_streaming_parity.py`, `tests/test_api_contracts.py`).
- Added streaming state persistence (`STREAMING_STATE_PATH`) and drift metrics logging for parity monitoring (`src/device_anomaly/streaming/telemetry_stream.py:202-342`, `src/device_anomaly/api/routes/streaming.py:105-214`, `src/device_anomaly/streaming/drift_monitor.py:1-214`).
- Added verify script + frontend lint/typecheck config to make local checks runnable (`scripts/verify.py`, `frontend/.eslintrc.cjs`, `Makefile:55-56`).

## Runbook (Operations)
- Start services: `docker-compose up -d` or `make up` (see `README.md:51-78`).
- Trigger training: POST `/api/training/start` with date range; monitor `/api/training/status` (`src/device_anomaly/api/routes/training.py:508-620`).
- Baseline storage: production baselines live under the latest model output directory as `baselines.json` (data-driven schema). Legacy artifacts remain in `artifacts/{source}_baselines.json` for back-compat (`src/device_anomaly/models/baseline_store.py:34-170`).
- Debug empty baseline suggestions: confirm `models/production/**/baselines.json` exists and `training_metadata.json` references `baselines_path`; check `/api/baselines/suggestions` returns 404 when neither production nor legacy baselines exist (`src/device_anomaly/api/routes/baselines.py:32-119`).
- Cohort stats: computed on train split and saved to `cohort_stats.json`; inference and streaming load from the latest model metadata (`src/device_anomaly/pipeline/training.py:229-547`, `src/device_anomaly/features/cohort_stats.py:146-206`).
- Feature spec + norms: stored in `training_metadata.json` as `feature_spec` and `feature_norms`; streaming warns when norms are missing (`src/device_anomaly/pipeline/training.py:235-560`, `src/device_anomaly/streaming/feature_computer.py:247-335`).
- Streaming feature mode: set `STREAMING_FEATURE_MODE=canonical` (default) for parity or `incremental` for low-latency; non-canonical mode logs a warning (`src/device_anomaly/streaming/feature_computer.py:131-162`).
- Streaming state: use `TelemetryBuffer.snapshot()` and `TelemetryBuffer.from_snapshot()` for state rehydration in deployments that persist buffer state (`src/device_anomaly/streaming/telemetry_stream.py:130-245`).
- Streaming state persistence: set `STREAMING_STATE_PATH` and `STREAMING_STATE_MAX_BYTES` to persist buffers; corrupt or oversized snapshots are skipped with warnings (`src/device_anomaly/api/routes/streaming.py:177-235`, `src/device_anomaly/streaming/telemetry_stream.py:288-360`).
- Drift metrics: `STREAMING_DRIFT_*` env vars control lightweight PSI logging; logs emit aggregated metrics only (`src/device_anomaly/streaming/drift_monitor.py:1-214`, `src/device_anomaly/streaming/feature_computer.py:183-337`).
- Score batch window: POST `/api/automation/score` with date range (`src/device_anomaly/api/routes/automation.py:462-523`).
- Check anomalies: GET `/api/anomalies` (`src/device_anomaly/api/routes/anomalies.py:29-77`).
- Streaming: enable `ENABLE_STREAMING=true` and watch logs for `AnomalyStreamProcessor` (`src/device_anomaly/api/main.py:24-40`, `src/device_anomaly/streaming/anomaly_processor.py:58-183`).
- Verify checks: run `make verify` or `python scripts/verify.py` to execute backend tests and frontend lint/typecheck/build (`scripts/verify.py:1-70`, `Makefile:33`).

## Next improvements
- None remaining in this pass.
