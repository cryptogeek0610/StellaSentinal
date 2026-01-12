from device_anomaly.streaming.drift_monitor import DriftMonitorConfig, StreamingDriftMonitor


def test_drift_monitor_emits_metrics():
    config = DriftMonitorConfig(
        window_size=4,
        bins=4,
        warn_psi=0.2,
        log_interval_sec=1,
        min_samples=2,
    )
    monitor = StreamingDriftMonitor(["metric"], config)

    metrics = None
    for idx in range(4):
        metrics = monitor.update({"metric": float(idx)})
    assert metrics is not None
    assert metrics["event"] == "streaming_drift_baseline_ready"

    metrics = None
    for idx in range(4):
        metrics = monitor.update({"metric": float(idx + 10)})

    assert metrics is not None
    assert metrics["event"] == "streaming_feature_drift"
    assert "metric" in metrics["psi"]


def test_drift_monitor_missing_feature_rates():
    config = DriftMonitorConfig(
        window_size=3,
        bins=3,
        warn_psi=0.2,
        log_interval_sec=1,
        min_samples=2,
    )
    monitor = StreamingDriftMonitor(["metric"], config)

    for idx in range(3):
        monitor.update({"metric": float(idx)})

    metrics = None
    for _ in range(3):
        metrics = monitor.update({})

    assert metrics is not None
    assert metrics["missing_feature_rates"]["metric"] == 1.0
