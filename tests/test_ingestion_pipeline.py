from __future__ import annotations

from datetime import UTC, datetime, timedelta

from device_anomaly.config.settings import reset_settings
from device_anomaly.services.ingestion_pipeline import run_ingestion_batch


class DummyWatermarkStore:
    def __init__(self) -> None:
        self._values = {}
        self._metadata = {}

    def _key(self, source_db: str, table_name: str) -> str:
        return f"{source_db}.{table_name}"

    def get_watermark(
        self, source_db: str, table_name: str, default_lookback_hours: int | None = None
    ):
        if default_lookback_hours is None:
            default_lookback_hours = 1
        return self._values.get(
            self._key(source_db, table_name),
            datetime.now(UTC) - timedelta(hours=default_lookback_hours),
        )

    def get_watermark_metadata(self, source_db: str, table_name: str):
        return self._metadata.get(self._key(source_db, table_name))

    def set_watermark(
        self,
        source_db: str,
        table_name: str,
        watermark_value,
        watermark_column=None,
        rows_extracted=0,
        metadata=None,
    ):
        self._values[self._key(source_db, table_name)] = watermark_value
        if metadata is not None:
            self._metadata[self._key(source_db, table_name)] = metadata
        return True, None


def test_dry_run_ingestion_records_metrics_and_watermarks(monkeypatch):
    monkeypatch.setenv("ENABLE_MC_TIMESERIES", "true")
    reset_settings()

    metrics = []

    def record_metric(**kwargs):
        metrics.append(kwargs)

    watermark_store = DummyWatermarkStore()

    result = run_ingestion_batch(
        xsight_tables=["cs_BatteryStat"],
        mc_tables=["DeviceStatInt"],
        dry_run=True,
        record_metric=record_metric,
        watermark_store=watermark_store,
    )

    assert result.total_rows_fetched >= 0
    assert len(metrics) == 2
    assert any(m["table_name"] == "cs_BatteryStat" for m in metrics)
    assert any(m["table_name"] == "DeviceStatInt" for m in metrics)
    assert watermark_store.get_watermark("xsight", "cs_BatteryStat") is not None
    assert watermark_store.get_watermark("mobicontrol", "DeviceStatInt") is not None


def test_ingestion_records_skipped_when_mc_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_MC_TIMESERIES", "false")
    reset_settings()

    metrics = []

    def record_metric(**kwargs):
        metrics.append(kwargs)

    watermark_store = DummyWatermarkStore()

    run_ingestion_batch(
        xsight_tables=["cs_BatteryStat"],
        mc_tables=["DeviceStatInt"],
        dry_run=True,
        record_metric=record_metric,
        watermark_store=watermark_store,
    )

    skip_metrics = [m for m in metrics if m.get("rows_skipped") == 1]
    assert any(m.get("warning") == "disabled_mc_timeseries" for m in skip_metrics)
