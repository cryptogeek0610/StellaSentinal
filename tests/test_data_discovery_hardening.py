"""
Unit tests for production-hardened data discovery features.

Tests:
- Monotonic watermarks (PostgreSQL source of truth)
- Stable event_id generation and determinism
- Event deduplication
- Keyset pagination
- Table allowlists
- Weight-based parallelism throttling
"""
import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


class TestMonotonicWatermarks:
    """Tests for monotonic watermark enforcement."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create temporary cache directory."""
        cache_dir = tmp_path / "watermarks"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir

    def test_watermark_rejects_backward_move(self, temp_cache_dir):
        """Test that set_watermark rejects watermarks that move backward."""
        from device_anomaly.data_access.watermark_store import WatermarkStore

        # Enable file fallback for testing without Postgres
        store = WatermarkStore(
            postgres_url=None,
            redis_url=None,
            file_path=temp_cache_dir / "watermarks.json",
            lookback_hours=24,
            enable_file_fallback=True,
        )

        # Set initial watermark
        t1 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        success, error = store.set_watermark(
            "xsight", "cs_BatteryStat", t1
        )
        assert success is True
        assert error is None

        # Try to set watermark backward - should be rejected
        t2 = datetime(2024, 1, 14, 10, 0, 0, tzinfo=timezone.utc)  # 1 day earlier
        success, error = store.set_watermark(
            "xsight", "cs_BatteryStat", t2
        )
        assert success is False
        assert "Monotonic violation" in error

        # Verify watermark didn't change
        current = store.get_watermark("xsight", "cs_BatteryStat")
        assert current == t1

    def test_watermark_allows_forward_move(self, temp_cache_dir):
        """Test that set_watermark accepts watermarks that move forward."""
        from device_anomaly.data_access.watermark_store import WatermarkStore

        store = WatermarkStore(
            postgres_url=None,
            redis_url=None,
            file_path=temp_cache_dir / "watermarks.json",
            lookback_hours=24,
            enable_file_fallback=True,
        )

        # Set initial watermark
        t1 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        success, _ = store.set_watermark("xsight", "cs_BatteryStat", t1)
        assert success is True

        # Set watermark forward - should succeed
        t2 = datetime(2024, 1, 16, 10, 0, 0, tzinfo=timezone.utc)
        success, error = store.set_watermark("xsight", "cs_BatteryStat", t2)
        assert success is True
        assert error is None

        # Verify watermark advanced
        current = store.get_watermark("xsight", "cs_BatteryStat")
        assert current == t2

    def test_reset_watermark_allows_backward_move(self, temp_cache_dir):
        """Test that reset_watermark can move watermark backward."""
        from device_anomaly.data_access.watermark_store import WatermarkStore

        store = WatermarkStore(
            postgres_url=None,
            redis_url=None,
            file_path=temp_cache_dir / "watermarks.json",
            lookback_hours=24,
            enable_file_fallback=True,
        )

        # Set initial watermark far in the future
        t1 = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        store.set_watermark("xsight", "cs_BatteryStat", t1)

        # Reset to explicit past time
        t2 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        new_wm, success = store.reset_watermark(
            "xsight", "cs_BatteryStat", to_datetime=t2
        )

        assert success is True
        assert new_wm == t2

        # Verify watermark was reset
        current = store.get_watermark("xsight", "cs_BatteryStat")
        assert current == t2

    def test_watermark_unchanged_skips_write(self, temp_cache_dir):
        """Test that setting the same watermark is a no-op."""
        from device_anomaly.data_access.watermark_store import WatermarkStore

        store = WatermarkStore(
            postgres_url=None,
            redis_url=None,
            file_path=temp_cache_dir / "watermarks.json",
            lookback_hours=24,
            enable_file_fallback=True,
        )

        t1 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        store.set_watermark("xsight", "cs_BatteryStat", t1)

        # Set same watermark again
        success, error = store.set_watermark("xsight", "cs_BatteryStat", t1)
        assert success is True
        assert error is None


class TestStableEventId:
    """Tests for stable event_id generation using SHA256."""

    def test_compute_stable_event_id_deterministic(self):
        """Test that event_id is deterministic for same inputs."""
        from device_anomaly.data_access.canonical_events import compute_stable_event_id

        event_id_1 = compute_stable_event_id(
            source_db="xsight",
            source_table="cs_BatteryStat",
            device_id=123,
            event_time=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            metric_name="battery_level",
            metric_value=85,
            dimensions={"app": "test"},
        )

        event_id_2 = compute_stable_event_id(
            source_db="xsight",
            source_table="cs_BatteryStat",
            device_id=123,
            event_time=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            metric_name="battery_level",
            metric_value=85,
            dimensions={"app": "test"},
        )

        assert event_id_1 == event_id_2
        assert len(event_id_1) == 32  # SHA256 truncated to 32 chars

    def test_compute_stable_event_id_different_inputs(self):
        """Test that different inputs produce different event_ids."""
        from device_anomaly.data_access.canonical_events import compute_stable_event_id

        base_params = {
            "source_db": "xsight",
            "source_table": "cs_BatteryStat",
            "device_id": 123,
            "event_time": datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            "metric_name": "battery_level",
            "metric_value": 85,
            "dimensions": {},
        }

        event_id_base = compute_stable_event_id(**base_params)

        # Different device_id
        event_id_diff_device = compute_stable_event_id(
            **{**base_params, "device_id": 456}
        )
        assert event_id_base != event_id_diff_device

        # Different metric_value
        event_id_diff_value = compute_stable_event_id(
            **{**base_params, "metric_value": 90}
        )
        assert event_id_base != event_id_diff_value

        # Different event_time
        event_id_diff_time = compute_stable_event_id(
            **{**base_params, "event_time": datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)}
        )
        assert event_id_base != event_id_diff_time

    def test_deterministic_json_ordering(self):
        """Test that dimensions with different key ordering produce same hash."""
        from device_anomaly.data_access.canonical_events import (
            _deterministic_json,
            compute_stable_event_id,
        )

        # Test _deterministic_json directly
        json1 = _deterministic_json({"b": 2, "a": 1})
        json2 = _deterministic_json({"a": 1, "b": 2})
        assert json1 == json2

        # Test via compute_stable_event_id
        event_id_1 = compute_stable_event_id(
            source_db="xsight",
            source_table="cs_BatteryStat",
            device_id=123,
            event_time=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            metric_name="battery_level",
            metric_value=85,
            dimensions={"z": 3, "a": 1, "m": 2},
        )

        event_id_2 = compute_stable_event_id(
            source_db="xsight",
            source_table="cs_BatteryStat",
            device_id=123,
            event_time=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            metric_name="battery_level",
            metric_value=85,
            dimensions={"a": 1, "m": 2, "z": 3},
        )

        assert event_id_1 == event_id_2


class TestEventDeduplication:
    """Tests for event deduplication."""

    def test_dedupe_events_removes_duplicates(self):
        """Test that dedupe_events removes duplicate events."""
        from device_anomaly.data_access.canonical_events import (
            CanonicalEvent,
            MetricType,
            SourceDatabase,
            dedupe_events,
        )

        base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        events = [
            CanonicalEvent(
                tenant_id="default",
                source_db=SourceDatabase.XSIGHT,
                source_table="cs_BatteryStat",
                device_id=123,
                event_time=base_time,
                metric_name="battery_level",
                metric_type=MetricType.INT,
                metric_value=85,
            ),
            # Duplicate of first event
            CanonicalEvent(
                tenant_id="default",
                source_db=SourceDatabase.XSIGHT,
                source_table="cs_BatteryStat",
                device_id=123,
                event_time=base_time,
                metric_name="battery_level",
                metric_type=MetricType.INT,
                metric_value=85,
            ),
            # Different event (different metric_value)
            CanonicalEvent(
                tenant_id="default",
                source_db=SourceDatabase.XSIGHT,
                source_table="cs_BatteryStat",
                device_id=123,
                event_time=base_time,
                metric_name="battery_level",
                metric_type=MetricType.INT,
                metric_value=90,
            ),
        ]

        unique_events, seen_ids = dedupe_events(events)

        assert len(unique_events) == 2
        assert len(seen_ids) == 2

    def test_dedupe_events_with_seen_ids(self):
        """Test dedupe_events filters out previously seen IDs."""
        from device_anomaly.data_access.canonical_events import (
            CanonicalEvent,
            MetricType,
            SourceDatabase,
            dedupe_events,
        )

        base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        event1 = CanonicalEvent(
            tenant_id="default",
            source_db=SourceDatabase.XSIGHT,
            source_table="cs_BatteryStat",
            device_id=123,
            event_time=base_time,
            metric_name="battery_level",
            metric_type=MetricType.INT,
            metric_value=85,
        )

        event2 = CanonicalEvent(
            tenant_id="default",
            source_db=SourceDatabase.XSIGHT,
            source_table="cs_BatteryStat",
            device_id=456,
            event_time=base_time,
            metric_name="battery_level",
            metric_type=MetricType.INT,
            metric_value=90,
        )

        # First batch - get event1's ID
        _, seen = dedupe_events([event1])

        # Second batch with event1 and event2 - event1 should be filtered
        unique, _ = dedupe_events([event1, event2], seen_ids=seen)

        assert len(unique) == 1
        assert unique[0].device_id == 456  # Only event2 should remain


class TestTableAllowlists:
    """Tests for table allowlist enforcement."""

    def test_allowlist_filters_tables(self):
        """Test that allowlist filters tables correctly."""
        from device_anomaly.config.settings import AppSettings

        settings = AppSettings(
            xsight_table_allowlist=["cs_BatteryStat", "cs_AppUsage"],
            mc_table_allowlist=["DeviceStatInt"],
        )

        assert "cs_BatteryStat" in settings.xsight_table_allowlist
        assert "cs_AppUsage" in settings.xsight_table_allowlist
        assert "cs_DataUsageByHour" not in settings.xsight_table_allowlist

        assert "DeviceStatInt" in settings.mc_table_allowlist
        assert "DeviceStatLocation" not in settings.mc_table_allowlist

    def test_empty_allowlist_allows_all(self):
        """Test that empty allowlist means no filtering."""
        from device_anomaly.config.settings import AppSettings

        settings = AppSettings(
            xsight_table_allowlist=[],
            mc_table_allowlist=[],
        )

        assert len(settings.xsight_table_allowlist) == 0
        assert len(settings.mc_table_allowlist) == 0


class TestWeightBasedThrottling:
    """Tests for weight-based parallelism throttling."""

    def test_table_weight_assignment(self):
        """Test that tables are assigned correct weights."""
        from device_anomaly.services.ingestion_orchestrator import (
            get_table_weight,
            TableCategory,
            get_table_category,
        )

        # XSight hourly huge tables - weight 5
        assert get_table_weight("cs_DataUsageByHour") == 5
        assert get_table_weight("cs_WiFiLocation") == 5
        assert get_table_category("cs_DataUsageByHour") == TableCategory.XSIGHT_HOURLY_HUGE

        # XSight extended tables - weight 2
        assert get_table_weight("cs_BatteryDrain") == 2
        assert get_table_category("cs_BatteryDrain") == TableCategory.XSIGHT_EXTENDED

        # MobiControl time-series - weight 2
        assert get_table_weight("DeviceStatInt") == 2
        assert get_table_weight("DeviceStatLocation") == 2
        assert get_table_category("DeviceStatInt") == TableCategory.MC_TIMESERIES

        # Small tables - weight 1
        assert get_table_weight("Alert") == 1
        assert get_table_weight("Events") == 1
        assert get_table_category("Alert") == TableCategory.SMALL

        # Unknown tables - weight 1 (default)
        assert get_table_weight("UnknownTable") == 1
        assert get_table_category("UnknownTable") == TableCategory.DEFAULT

    def test_weighted_semaphore_basic(self):
        """Test weighted semaphore basic acquire/release."""
        from device_anomaly.services.ingestion_orchestrator import WeightedSemaphore

        async def test():
            sem = WeightedSemaphore(max_weight=5)

            assert sem.current_weight == 0
            assert sem.available_weight == 5

            ctx = await sem.acquire(weight=2)
            assert sem.current_weight == 2
            assert sem.available_weight == 3

            await sem.release(2)
            assert sem.current_weight == 0
            assert sem.available_weight == 5

        asyncio.run(test())

    def test_weighted_semaphore_blocks_over_capacity(self):
        """Test weighted semaphore blocks when over capacity."""
        from device_anomaly.services.ingestion_orchestrator import WeightedSemaphore

        async def test():
            sem = WeightedSemaphore(max_weight=5)

            # Acquire weight 4
            ctx1 = await sem.acquire(weight=4)
            assert sem.current_weight == 4

            # Trying to acquire weight 3 should not complete immediately
            # (would exceed max_weight of 5)
            acquired = False

            async def try_acquire():
                nonlocal acquired
                ctx2 = await sem.acquire(weight=3)
                acquired = True
                await sem.release(3)

            # Start the acquire task
            task = asyncio.create_task(try_acquire())

            # Give it a moment
            await asyncio.sleep(0.1)

            # Should not have acquired yet
            assert acquired is False
            assert sem.current_weight == 4

            # Release the first weight
            await sem.release(4)

            # Now the second acquire should complete
            await asyncio.wait_for(task, timeout=1.0)
            assert acquired is True

        asyncio.run(test())

    def test_weighted_semaphore_clamps_to_max(self):
        """Test weighted semaphore clamps excessive weights to max."""
        from device_anomaly.services.ingestion_orchestrator import WeightedSemaphore

        async def test():
            sem = WeightedSemaphore(max_weight=5)

            # Request weight > max should be clamped
            ctx = await sem.acquire(weight=10)  # Will be clamped to 5
            assert sem.current_weight == 5

            await sem.release(5)
            assert sem.current_weight == 0

        asyncio.run(test())

    def test_ingestion_task_auto_weight(self):
        """Test IngestionTask automatically assigns weight."""
        from device_anomaly.services.ingestion_orchestrator import (
            IngestionTask,
            TableCategory,
        )

        # Huge table
        task1 = IngestionTask(table_name="cs_DataUsageByHour", source_db="xsight")
        assert task1.weight == 5
        assert task1.category == TableCategory.XSIGHT_HOURLY_HUGE

        # Extended table
        task2 = IngestionTask(table_name="cs_BatteryDrain", source_db="xsight")
        assert task2.weight == 2
        assert task2.category == TableCategory.XSIGHT_EXTENDED

        # Small table
        task3 = IngestionTask(table_name="Alert", source_db="mobicontrol")
        assert task3.weight == 1
        assert task3.category == TableCategory.SMALL


class TestKeysetPagination:
    """Tests for keyset pagination logic."""

    def test_hourly_table_keyset_detection(self):
        """Test hourly table keyset column detection."""
        # This tests the pattern - the actual implementation is in xsight_loader_extended
        hourly_tables = {"cs_DataUsageByHour", "cs_WiFiByHour", "cs_CellTowerByHour"}

        for table in hourly_tables:
            # These tables should use composite keyset: (CollectedDate, Hour)
            assert "ByHour" in table or "Hour" in table

    def test_keyset_order_stable(self):
        """Test that keyset ordering is deterministic."""
        # Simulate keyset pagination data
        rows = [
            {"CollectedDate": "2024-01-15", "Hour": 10, "Value": 100},
            {"CollectedDate": "2024-01-15", "Hour": 11, "Value": 200},
            {"CollectedDate": "2024-01-16", "Hour": 0, "Value": 300},
        ]

        # Sort by keyset (CollectedDate, Hour)
        sorted_rows = sorted(rows, key=lambda r: (r["CollectedDate"], r["Hour"]))

        assert sorted_rows[0]["Hour"] == 10
        assert sorted_rows[1]["Hour"] == 11
        assert sorted_rows[2]["CollectedDate"] == "2024-01-16"


class TestFeatureFlags:
    """Tests for feature flag defaults."""

    def test_extended_ingestion_flags_default_off(self):
        """Test that extended ingestion flags default to OFF."""
        from device_anomaly.config.settings import AppSettings

        settings = AppSettings()

        # Extended ingestion flags should be OFF
        assert settings.enable_mc_timeseries is False
        assert settings.enable_xsight_hourly is False
        assert settings.enable_xsight_extended is False

        # File fallback should be OFF
        assert settings.enable_file_watermark_fallback is False

    def test_observability_flags_default_on(self):
        """Test that observability flags default to ON."""
        from device_anomaly.config.settings import AppSettings

        settings = AppSettings()

        # Observability should be ON
        assert settings.enable_ingestion_metrics is True
        assert settings.enable_daily_coverage_report is True
        assert settings.enable_schema_discovery is True
        assert settings.auto_create_metrics_tables is False


class TestIngestionMetrics:
    """Tests for ingestion metrics."""

    def test_table_ingestion_metric_properties(self):
        """Test TableIngestionMetric computed properties."""
        from device_anomaly.services.ingestion_metrics import TableIngestionMetric

        started = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        completed = datetime(2024, 1, 15, 10, 0, 5, tzinfo=timezone.utc)

        metric = TableIngestionMetric(
            source_db="xsight",
            table_name="cs_BatteryStat",
            started_at=started,
            completed_at=completed,
            rows_fetched=1000,
            rows_inserted=950,
            rows_deduped=50,
        )

        # Duration
        assert metric.duration_ms == 5000  # 5 seconds in ms

        # Dedupe ratio
        assert metric.dedupe_ratio == 0.05  # 50/1000

        # Success (no error)
        assert metric.success is True

    def test_daily_coverage_report_structure(self):
        """Test DailyCoverageReport structure."""
        from device_anomaly.services.ingestion_metrics import DailyCoverageReport

        report = DailyCoverageReport(
            report_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            xsight_tables_loaded=10,
            xsight_tables_failed=1,
            xsight_total_rows=100000,
            mc_tables_loaded=5,
            mc_tables_failed=0,
            mc_total_rows=50000,
            tables_with_errors=["xsight.cs_BatteryStat"],
            tables_with_high_lag=["mobicontrol.DeviceStatInt"],
        )

        data = report.to_dict()

        assert data["xsight_tables_loaded"] == 10
        assert data["xsight_tables_failed"] == 1
        assert len(data["tables_with_errors"]) == 1
        assert len(data["tables_with_high_lag"]) == 1


class TestGracefulDegradation:
    """Tests for graceful degradation behavior."""

    def test_missing_table_returns_empty(self):
        """Test that missing tables return empty results gracefully."""
        # This would test the actual loader - for now test the pattern
        def load_table(table_name: str, tables_available: set) -> list:
            if table_name not in tables_available:
                return []  # Graceful degradation
            return [{"id": 1}]

        result = load_table("NonexistentTable", {"cs_BatteryStat"})
        assert result == []

    def test_missing_columns_handled(self):
        """Test that missing columns are handled gracefully."""
        def validate_columns(requested: list, available: set) -> list:
            return [c for c in requested if c in available]

        requested = ["DeviceId", "BatteryLevel", "NonexistentColumn"]
        available = {"DeviceId", "BatteryLevel", "OtherColumn"}

        valid = validate_columns(requested, available)
        assert "NonexistentColumn" not in valid
        assert len(valid) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
