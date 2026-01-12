# FEATURE_PARITY

## Chosen Approach
Option 1: Single canonical feature library. Streaming, batch inference, and training will use `DeviceFeatureBuilder` with a shared feature spec + training-derived feature norms. Streaming computes features from its per-device buffer using the same builder and applies stored cohort stats for parity.

## Feature Spec (Canonical)
- Spec version: `device_features_v1`
- Default window: 14 (rows/events per device)
- Rolling windows: [7, 14, 30]
- Min periods: 3
- Derived features: enabled
- Volatility features: enabled
- Cohort z-scores: computed from stored cohort stats (train-only)

## Feature Mapping Table

| Feature group | Example features | Definition | Windowing requirement | Online computable | Current skew source | Parity plan |
| --- | --- | --- | --- | --- | --- | --- |
| Raw metrics | `TotalBatteryLevelDrop`, `Download`, `Upload` | Direct telemetry values | None | Yes | None (streaming already uses raw) | Use canonical builder, same raw metrics |
| Rolling stats | `*_roll_mean`, `*_roll_std`, `*_roll_min`, `*_roll_max`, `*_roll_median`, `*_roll_7d_mean`, `*_roll_30d_std` | Per-device rolling stats across configured windows | Requires last N events (N = window size) | Yes, if buffer holds >= window | Streaming used full-buffer incremental stats (not windowed) | Compute via canonical builder on buffer |
| Delta + trend | `*_delta`, `*_pct_change`, `*_trend_7d` | Change vs previous event + 7-event trend proxy | Requires >= 2 events (trend requires >= 7) | Yes | Streaming only computed delta + pct using mean; no trend | Compute via canonical builder |
| Temporal context | `hour_of_day`, `day_of_week`, `hour_sin`, `day_cos` | Timestamp-derived cyclical context | None | Yes | Streaming omitted temporal features | Compute via canonical builder |
| Derived efficiency | `BatteryDrainPerHour`, `BatteryDrainPerMB`, `DropRate`, `StorageUtilization` | Domain-specific ratios and efficiencies | None | Yes | Streaming computed a subset with different formulas | Compute via canonical builder |
| Volatility | `*_cv` | Coefficient of variation from rolling mean/std | Requires rolling stats | Yes | Streaming omitted | Compute via canonical builder |
| Cohort z-scores | `*_cohort_z` | Robust z-score per cohort using median/MAD | Requires cohort stats | Yes (with stored stats) | Streaming used incremental cohort stats when store missing | Always apply stored cohort stats; log if missing |
| Cross-domain | `DeviceHealthScore`, `AnomalyRiskScore`, `BatteryNetworkStress` | Composite signals across domains | Depends on derived features + normalization | Yes | Streaming used different formulas and local normalization | Use training-derived normalization norms from metadata |

## Parity Guarantees
- Exact parity between streaming and batch features when the streaming buffer contains at least the largest rolling window (>= 30 events) for a device.
- If history is shorter, rolling features may be NaN and will be imputed during scoring; this is logged and marked as “partial history”.

## Modes
- `STREAMING_FEATURE_MODE=canonical` (default): uses `DeviceFeatureBuilder` over the buffer for parity with batch scoring.
- `STREAMING_FEATURE_MODE=incremental`: uses incremental rollups for lower latency; logs a warning because parity is reduced.

## State Persistence
- `STREAMING_STATE_PATH` (optional): when set, streaming buffers are saved on shutdown and restored on startup.
- `STREAMING_STATE_MAX_BYTES` (optional, default 10MB): prevents oversized snapshots.
- Corrupt or oversized snapshots are ignored with warnings; streaming starts with an empty buffer.
- The service needs write access to the state path.

## Drift Metrics
- `STREAMING_DRIFT_ENABLED` (default true): enables lightweight drift metrics.
- Metrics are logged periodically as `streaming_drift_metrics` with PSI-like scores and missing/short-history counts.
- Key knobs: `STREAMING_DRIFT_WINDOW_SIZE`, `STREAMING_DRIFT_INTERVAL_SEC`, `STREAMING_DRIFT_BINS`, `STREAMING_DRIFT_WARN_PSI`.
- `STREAMING_DRIFT_FEATURES` overrides the default feature list used for drift monitoring.

## Performance Guardrails
- `STREAMING_FEATURE_COMPUTE_WARN_MS` emits a warning when canonical feature computation exceeds the configured threshold (set to `0` to disable).

## Tolerances (Tests)
- Feature parity: abs diff <= 1e-6 for numeric features present in both paths.
- Anomaly score parity: abs diff <= 1e-4 when features are present and imputation is identical.

## Monitoring Hooks
- Log when streaming history is shorter than required for full rolling windows.
- Log when cohort stats or feature norms are missing (parity cannot be guaranteed).

## Edge Case Handling

### Out-of-Order Event Arrivals
- **Problem**: Events may arrive at the streaming service out of timestamp order (network delays, retries, etc.).
- **Solution**: In canonical mode, `_buffer_to_dataframe()` sorts events by timestamp before computing features. This ensures rolling window alignment matches batch processing.
- **Guarantee**: Features computed for event at timestamp T will be identical whether events arrived in order or out of order.

### NaN and Infinity Values
- **Problem**: Telemetry may contain NaN or infinity values that contaminate rolling statistics and downstream features.
- **Solution**:
  - `DeviceBuffer.add_event()` only includes values that pass `np.isfinite()` in running statistics.
  - `_buffer_to_dataframe()` sanitizes metrics, excluding non-finite values from the DataFrame.
- **Guarantee**: Rolling means, standard deviations, and derived features will never contain NaN or infinity from input data.

### Mixed Types and Missing Values
- **Problem**: Metrics may arrive with unexpected types (strings) or missing values (None).
- **Solution**:
  - Non-numeric metric values are silently dropped.
  - None values are excluded from statistics.
- **Guarantee**: Only valid numeric values contribute to features.

---

## RUNBOOK

### Debugging Feature Skew

1. **Check feature mode**: Ensure `STREAMING_FEATURE_MODE=canonical` is set in production.
   ```bash
   echo $STREAMING_FEATURE_MODE  # Should be "canonical" or unset (defaults to canonical)
   ```

2. **Verify cohort stats loaded**: Check logs on startup for:
   ```
   INFO: Streaming cohort stats loaded from models/cohort_stats.json
   ```
   If you see `WARNING: Streaming cohort stats not found`, cohort z-scores will be missing.

3. **Check feature norms**: Look for:
   ```
   INFO: Feature norms resolved from training metadata
   ```
   Missing norms will cause cross-domain features to differ.

4. **Compare features manually**:
   ```python
   from device_anomaly.features.device_features import DeviceFeatureBuilder
   from device_anomaly.streaming.feature_computer import StreamingFeatureComputer

   # Compute batch features
   builder = DeviceFeatureBuilder()
   batch_features = builder.transform(df).iloc[-1]

   # Compare to streaming features for same timestamp
   ```

### Reading Drift Metrics

Drift metrics are logged as structured JSON with `event=streaming_feature_drift`:

```json
{
  "event": "streaming_feature_drift",
  "psi": {"BatteryDrop_roll_mean": 0.15, "Upload_delta": 0.02},
  "drift_warnings": ["BatteryDrop_roll_mean"],
  "missing_feature_rates": {"SomeMetric": 0.1}
}
```

- **PSI > 0.2**: Significant drift; investigate data pipeline or model freshness.
- **PSI > 0.1**: Moderate drift; monitor closely.
- **missing_feature_rates > 0.5**: Feature is frequently absent; check telemetry source.

### Enabling State Persistence

1. **Set environment variables**:
   ```bash
   export STREAMING_STATE_PATH=/data/streaming/state.json
   export STREAMING_STATE_MAX_BYTES=10000000  # 10MB limit
   ```

2. **Verify write permissions**:
   ```bash
   touch /data/streaming/state.json && rm /data/streaming/state.json
   ```

3. **Check logs on shutdown**:
   ```
   INFO: Streaming state persisted to /data/streaming/state.json
   ```

4. **Check logs on startup**:
   ```
   INFO: Streaming state restored from /data/streaming/state.json
   ```

5. **Troubleshooting**:
   - If state is not restored, check for `WARNING: Streaming state not restored` in logs.
   - Corrupt or oversized files are ignored; delete and restart if needed.

### Confirming Cohort Stats and Baselines

1. **Check cohort stats exist**:
   ```bash
   ls -la models/cohort_stats.json
   cat models/cohort_stats.json | python -m json.tool | head -20
   ```

2. **Check baselines exist**:
   ```bash
   ls -la models/baselines.json
   cat models/baselines.json | python -m json.tool | head -20
   ```

3. **API verification**:
   ```bash
   curl http://localhost:8000/api/baselines/suggestions | python -m json.tool
   # Should return suggestions when anomalies exist

   curl http://localhost:8000/api/dashboard/stats | python -m json.tool
   # Should reflect actual data counts
   ```

4. **Verify trained config in dashboard**:
   ```bash
   curl http://localhost:8000/api/dashboard/isolation-forest/stats | python -m json.tool
   # config.n_estimators should match training config, not defaults
   ```
