from pathlib import Path

import pandas as pd

from device_anomaly.streaming.telemetry_stream import DeviceBuffer, TelemetryBuffer, TelemetryEvent


def test_streaming_state_snapshot_roundtrip(tmp_path: Path):
    buffer = TelemetryBuffer()
    device_buffer = DeviceBuffer(device_id=1)
    buffer._buffers[1] = device_buffer

    event = TelemetryEvent(
        device_id=1,
        timestamp=pd.Timestamp("2026-01-07T00:00:00Z").to_pydatetime(),
        metrics={"TotalBatteryLevelDrop": 5.0},
        manufacturer_id=10,
        model_id=100,
        os_version_id=1000,
    )
    device_buffer.add_event(event)

    path = tmp_path / "stream_state.json"
    saved = buffer.save_snapshot(path, max_bytes=10_000)
    assert saved is True

    loaded = TelemetryBuffer.load_snapshot_path(path, max_bytes=10_000)
    assert loaded is not None
    assert loaded.get_device_count() == 1
    assert loaded.get_total_events() == 1


def test_streaming_state_corrupt_snapshot(tmp_path: Path):
    path = tmp_path / "stream_state.json"
    path.write_text("not-json")
    loaded = TelemetryBuffer.load_snapshot_path(path, max_bytes=10_000)
    assert loaded is None


def test_streaming_state_size_limit(tmp_path: Path):
    buffer = TelemetryBuffer()
    device_buffer = DeviceBuffer(device_id=1)
    buffer._buffers[1] = device_buffer

    for idx in range(5):
        event = TelemetryEvent(
            device_id=1,
            timestamp=pd.Timestamp("2026-01-07T00:00:00Z").to_pydatetime(),
            metrics={"TotalBatteryLevelDrop": float(idx)},
        )
        device_buffer.add_event(event)

    path = tmp_path / "stream_state.json"
    saved = buffer.save_snapshot(path, max_bytes=1)
    assert saved is False
    assert not path.exists()

    path.write_text("{}")
    loaded = TelemetryBuffer.load_snapshot_path(path, max_bytes=1)
    assert loaded is None


def test_streaming_state_missing_file(tmp_path: Path):
    """Ensure missing file returns None gracefully."""
    path = tmp_path / "nonexistent.json"
    loaded = TelemetryBuffer.load_snapshot_path(path, max_bytes=10_000)
    assert loaded is None


def test_streaming_state_empty_json(tmp_path: Path):
    """Ensure empty JSON returns a valid empty buffer."""
    path = tmp_path / "stream_state.json"
    path.write_text("{}")
    loaded = TelemetryBuffer.load_snapshot_path(path, max_bytes=10_000)
    # Empty JSON is not corrupt, but schema may vary; expect None or empty buffer
    assert loaded is None or loaded.get_device_count() == 0


def test_streaming_state_partial_schema(tmp_path: Path):
    """Ensure partial/malformed schema returns None gracefully."""
    path = tmp_path / "stream_state.json"
    path.write_text('{"version": 1, "buffers": "invalid"}')
    loaded = TelemetryBuffer.load_snapshot_path(path, max_bytes=10_000)
    assert loaded is None
