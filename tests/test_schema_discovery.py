"""
Unit tests for schema discovery module.

Tests runtime table/view discovery, caching, and high-value table identification
without requiring a real SQL Server connection.
"""

import json
from datetime import UTC, datetime

import pytest


class MockCursor:
    """Mock SQL cursor for testing."""

    def __init__(self, results: list[tuple]):
        self._results = results
        self._index = 0

    def fetchall(self):
        return self._results

    def fetchone(self):
        if self._index < len(self._results):
            result = self._results[self._index]
            self._index += 1
            return result
        return None


class MockConnection:
    """Mock SQL connection for testing."""

    def __init__(self, results: dict[str, list[tuple]]):
        self._results = results
        self._last_query = ""

    def execute(self, query, params=None):
        # Extract table pattern from query for result lookup
        self._last_query = str(query)
        for key, results in self._results.items():
            if key in self._last_query:
                return MockCursor(results)
        return MockCursor([])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockEngine:
    """Mock SQLAlchemy engine for testing."""

    def __init__(self, results: dict[str, list[tuple]]):
        self._results = results

    def connect(self):
        return MockConnection(self._results)


class TestSchemaDiscovery:
    """Tests for schema_discovery module."""

    @pytest.fixture
    def mock_xsight_tables(self):
        """Mock XSight table data."""
        return {
            "INFORMATION_SCHEMA.TABLES": [
                ("dbo", "cs_BatteryStat", "BASE TABLE"),
                ("dbo", "cs_AppUsage", "BASE TABLE"),
                ("dbo", "cs_DataUsage", "BASE TABLE"),
                ("dbo", "cs_DataUsageByHour", "BASE TABLE"),
                ("dbo", "vw_mcDevice", "VIEW"),
            ],
            "sys.tables": [
                ("dbo", "cs_BatteryStat", 700000),
                ("dbo", "cs_AppUsage", 500000),
                ("dbo", "cs_DataUsage", 300000),
                ("dbo", "cs_DataUsageByHour", 104000000),
            ],
            "INFORMATION_SCHEMA.COLUMNS": [
                ("DeviceId", "int", None, "NO", 1),
                ("CollectedDate", "date", None, "NO", 2),
                ("TotalBatteryLevelDrop", "int", None, "YES", 3),
            ],
        }

    @pytest.fixture
    def mock_mc_tables(self):
        """Mock MobiControl table data."""
        return {
            "INFORMATION_SCHEMA.TABLES": [
                ("dbo", "DeviceStatInt", "BASE TABLE"),
                ("dbo", "DeviceStatLocation", "BASE TABLE"),
                ("dbo", "DeviceStatString", "BASE TABLE"),
                ("dbo", "DevInfo", "BASE TABLE"),
                ("dbo", "MainLog", "BASE TABLE"),
            ],
            "sys.tables": [
                ("dbo", "DeviceStatInt", 764000),
                ("dbo", "DeviceStatLocation", 619000),
                ("dbo", "DeviceStatString", 349000),
                ("dbo", "DevInfo", 5800),
                ("dbo", "MainLog", 1044000),
            ],
            "INFORMATION_SCHEMA.COLUMNS": [
                ("DeviceId", "int", None, "NO", 1),
                ("TimeStamp", "datetime", None, "NO", 2),
                ("StatType", "int", None, "NO", 3),
                ("IntValue", "int", None, "YES", 4),
                ("ServerDateTime", "datetime", None, "YES", 5),
            ],
        }

    def test_column_info_type_detection(self):
        """Test ColumnInfo type detection methods."""
        from device_anomaly.data_access.schema_discovery import ColumnInfo

        int_col = ColumnInfo(name="value", data_type="int")
        assert int_col.is_numeric is True
        assert int_col.is_datetime is False
        assert int_col.is_string is False

        dt_col = ColumnInfo(name="ts", data_type="datetime2")
        assert dt_col.is_numeric is False
        assert dt_col.is_datetime is True
        assert dt_col.is_string is False

        str_col = ColumnInfo(name="name", data_type="nvarchar", max_length=255)
        assert str_col.is_numeric is False
        assert str_col.is_datetime is False
        assert str_col.is_string is True

    def test_table_info_properties(self):
        """Test TableInfo properties."""
        from device_anomaly.data_access.schema_discovery import (
            ColumnInfo,
            SourceDatabase,
            TableInfo,
        )

        cols = [
            ColumnInfo(name="DeviceId", data_type="int"),
            ColumnInfo(name="ServerDateTime", data_type="datetime"),
            ColumnInfo(name="Value", data_type="float"),
        ]

        table = TableInfo(
            name="DeviceStatInt",
            schema_name="dbo",
            table_type="BASE TABLE",
            row_count=764000,
            columns=cols,
            has_device_id=True,
            has_timestamp=True,
            timestamp_column="ServerDateTime",
            device_id_column="DeviceId",
            is_high_value=True,
            source_db=SourceDatabase.MOBICONTROL,
        )

        assert table.full_name == "dbo.DeviceStatInt"
        assert table.is_view is False
        assert table.is_time_series is True
        assert table.is_high_value is True

    def test_high_value_pattern_matching(self):
        """Test high-value table pattern matching."""
        from device_anomaly.data_access.schema_discovery import (
            SourceDatabase,
            _is_high_value_table,
        )

        # XSight patterns
        assert _is_high_value_table("cs_BatteryStat", SourceDatabase.XSIGHT, 1000) is True
        assert _is_high_value_table("cs_DataUsageByHour", SourceDatabase.XSIGHT, 10000) is True
        assert _is_high_value_table("cs_BatteryStat_Last7", SourceDatabase.XSIGHT, 10000) is False
        assert _is_high_value_table("random_table", SourceDatabase.XSIGHT, 10000) is False

        # MobiControl patterns
        assert _is_high_value_table("DeviceStatInt", SourceDatabase.MOBICONTROL, 1000) is True
        assert _is_high_value_table("MainLog", SourceDatabase.MOBICONTROL, 1000) is True
        assert _is_high_value_table("Alert", SourceDatabase.MOBICONTROL, 100) is True
        assert _is_high_value_table("SomeOtherTable", SourceDatabase.MOBICONTROL, 100) is False

    def test_timestamp_column_identification(self):
        """Test timestamp column identification."""
        from device_anomaly.data_access.schema_discovery import (
            ColumnInfo,
            SourceDatabase,
            _identify_time_columns,
        )

        # Known timestamp column
        cols = [
            ColumnInfo(name="DeviceId", data_type="int"),
            ColumnInfo(name="ServerDateTime", data_type="datetime"),
            ColumnInfo(name="Value", data_type="int"),
        ]
        has_ts, ts_col = _identify_time_columns(cols, SourceDatabase.MOBICONTROL)
        assert has_ts is True
        assert ts_col == "ServerDateTime"

        # XSight timestamp column
        cols = [
            ColumnInfo(name="DeviceId", data_type="int"),
            ColumnInfo(name="CollectedDate", data_type="date"),
        ]
        has_ts, ts_col = _identify_time_columns(cols, SourceDatabase.XSIGHT)
        assert has_ts is True
        assert ts_col == "CollectedDate"

        # No timestamp column
        cols = [
            ColumnInfo(name="DeviceId", data_type="int"),
            ColumnInfo(name="Value", data_type="int"),
        ]
        has_ts, ts_col = _identify_time_columns(cols, SourceDatabase.MOBICONTROL)
        assert has_ts is False
        assert ts_col is None

    def test_device_column_identification(self):
        """Test device ID column identification."""
        from device_anomaly.data_access.schema_discovery import (
            ColumnInfo,
            _identify_device_columns,
        )

        cols = [
            ColumnInfo(name="DeviceId", data_type="int"),
            ColumnInfo(name="Value", data_type="int"),
        ]
        has_dev, dev_col = _identify_device_columns(cols)
        assert has_dev is True
        assert dev_col == "DeviceId"

        cols = [
            ColumnInfo(name="Id", data_type="int"),
            ColumnInfo(name="Value", data_type="int"),
        ]
        has_dev, dev_col = _identify_device_columns(cols)
        assert has_dev is False
        assert dev_col is None

    def test_priority_score_calculation(self):
        """Test priority score calculation for tables."""
        from device_anomaly.data_access.schema_discovery import (
            ColumnInfo,
            SourceDatabase,
            TableInfo,
            _calculate_priority_score,
        )

        # High-value time-series table with many rows
        table1 = TableInfo(
            name="DeviceStatInt",
            row_count=764000,
            columns=[
                ColumnInfo(name="DeviceId", data_type="int"),
                ColumnInfo(name="ServerDateTime", data_type="datetime"),
                ColumnInfo(name="StatType", data_type="int"),
                ColumnInfo(name="IntValue", data_type="int"),
            ],
            has_device_id=True,
            has_timestamp=True,
            is_high_value=True,
            source_db=SourceDatabase.MOBICONTROL,
        )
        score1 = _calculate_priority_score(table1)

        # Small table with no time-series
        table2 = TableInfo(
            name="Config",
            row_count=10,
            columns=[ColumnInfo(name="Key", data_type="varchar")],
            has_device_id=False,
            has_timestamp=False,
            is_high_value=False,
            source_db=SourceDatabase.MOBICONTROL,
        )
        score2 = _calculate_priority_score(table2)

        assert score1 > score2

    def test_database_schema_serialization(self):
        """Test DatabaseSchema to_dict and serialization."""
        from device_anomaly.data_access.schema_discovery import (
            ColumnInfo,
            DatabaseSchema,
            SourceDatabase,
            TableInfo,
        )

        table = TableInfo(
            name="cs_BatteryStat",
            row_count=700000,
            columns=[ColumnInfo(name="DeviceId", data_type="int")],
            is_high_value=True,
            source_db=SourceDatabase.XSIGHT,
        )

        schema = DatabaseSchema(
            database_name="SOTI_XSight_dw",
            source_db=SourceDatabase.XSIGHT,
            tables={"cs_BatteryStat": table},
            views={},
        )

        # Test to_dict
        data = schema.to_dict()
        assert data["database_name"] == "SOTI_XSight_dw"
        assert data["source_db"] == "xsight"
        assert "cs_BatteryStat" in data["tables"]

        # Test JSON serialization
        json_str = json.dumps(data)
        assert "cs_BatteryStat" in json_str

    def test_curated_table_list(self):
        """Test get_curated_table_list returns expected tables."""
        from device_anomaly.data_access.schema_discovery import (
            HIGH_VALUE_PATTERNS,
            SourceDatabase,
        )

        # Check XSight explicit tables
        xsight_explicit = HIGH_VALUE_PATTERNS[SourceDatabase.XSIGHT]["explicit"]
        assert "cs_BatteryStat" in xsight_explicit
        assert "cs_DataUsageByHour" in xsight_explicit

        # Check MobiControl explicit tables
        mc_explicit = HIGH_VALUE_PATTERNS[SourceDatabase.MOBICONTROL]["explicit"]
        assert "DeviceStatInt" in mc_explicit
        assert "DeviceStatLocation" in mc_explicit
        assert "MainLog" in mc_explicit


class TestWatermarkStore:
    """Tests for watermark_store module."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create temporary cache directory."""
        return tmp_path / "watermarks"

    def test_watermark_get_default(self, temp_cache_dir):
        """Test get_watermark returns default when no watermark exists."""
        from device_anomaly.data_access.watermark_store import WatermarkStore

        store = WatermarkStore(
            postgres_url=None,
            redis_url=None,
            file_path=temp_cache_dir / "watermarks.json",
            lookback_hours=24,
            enable_file_fallback=True,
        )

        wm = store.get_watermark("xsight", "cs_BatteryStat")

        # Should be approximately now - 24 hours
        now = datetime.now(UTC)
        diff = now - wm
        assert 23 * 3600 <= diff.total_seconds() <= 25 * 3600

    def test_watermark_set_and_get(self, temp_cache_dir):
        """Test setting and getting watermarks."""
        from device_anomaly.data_access.watermark_store import WatermarkStore

        store = WatermarkStore(
            postgres_url=None,
            redis_url=None,
            file_path=temp_cache_dir / "watermarks.json",
            lookback_hours=24,
            enable_file_fallback=True,
        )

        test_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

        # Set watermark
        success, _ = store.set_watermark(
            source_db="mobicontrol",
            table_name="DeviceStatInt",
            watermark_value=test_time,
            watermark_column="ServerDateTime",
            rows_extracted=1000,
        )
        assert success is True

        # Get watermark
        wm = store.get_watermark("mobicontrol", "DeviceStatInt")
        assert wm == test_time

    def test_watermark_reset(self, temp_cache_dir):
        """Test watermark reset functionality."""
        from device_anomaly.data_access.watermark_store import WatermarkStore

        store = WatermarkStore(
            postgres_url=None,
            redis_url=None,
            file_path=temp_cache_dir / "watermarks.json",
            lookback_hours=48,
            enable_file_fallback=True,
        )

        # Set a recent watermark
        recent = datetime(2024, 1, 20, 0, 0, 0, tzinfo=UTC)
        store.set_watermark("xsight", "cs_AppUsage", recent)

        # Reset to 48 hours back
        new_wm, _ = store.reset_watermark("xsight", "cs_AppUsage")

        # Should be approximately now - 48 hours
        now = datetime.now(UTC)
        diff = now - new_wm
        assert 47 * 3600 <= diff.total_seconds() <= 49 * 3600


class TestCanonicalEvents:
    """Tests for canonical_events module."""

    def test_metric_type_inference(self):
        """Test metric type inference."""
        from device_anomaly.data_access.canonical_events import MetricType, _infer_metric_type

        assert _infer_metric_type(42) == MetricType.INT
        assert _infer_metric_type(3.14) == MetricType.FLOAT
        assert _infer_metric_type("hello") == MetricType.STRING
        assert _infer_metric_type(True) == MetricType.BOOL
        assert _infer_metric_type({"key": "value"}) == MetricType.JSON

    def test_metric_name_normalization(self):
        """Test metric name normalization to snake_case."""
        from device_anomaly.data_access.canonical_events import _normalize_metric_name

        assert _normalize_metric_name("TotalBatteryLevelDrop") == "total_battery_level_drop"
        assert _normalize_metric_name("DeviceId") == "device_id"
        assert _normalize_metric_name("WiFiSignalStrength") == "wi_fi_signal_strength"
        assert _normalize_metric_name("some-value!@#") == "some_value"

    def test_canonical_event_hash(self):
        """Test canonical event hash generation for idempotency."""
        from device_anomaly.data_access.canonical_events import (
            CanonicalEvent,
            MetricType,
            SourceDatabase,
        )

        event1 = CanonicalEvent(
            tenant_id="default",
            source_db=SourceDatabase.MOBICONTROL,
            source_table="DeviceStatInt",
            device_id=123,
            event_time=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
            metric_name="battery_level",
            metric_type=MetricType.INT,
            metric_value=85,
        )

        event2 = CanonicalEvent(
            tenant_id="default",
            source_db=SourceDatabase.MOBICONTROL,
            source_table="DeviceStatInt",
            device_id=123,
            event_time=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
            metric_name="battery_level",
            metric_type=MetricType.INT,
            metric_value=85,
        )

        # Same events should have same hash
        assert event1.raw_hash == event2.raw_hash

        # Different value should have different hash
        event3 = CanonicalEvent(
            tenant_id="default",
            source_db=SourceDatabase.MOBICONTROL,
            source_table="DeviceStatInt",
            device_id=123,
            event_time=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
            metric_name="battery_level",
            metric_type=MetricType.INT,
            metric_value=90,  # Different value
        )
        assert event1.raw_hash != event3.raw_hash

    def test_mc_stat_int_normalization(self):
        """Test DeviceStatInt row normalization."""
        from device_anomaly.data_access.canonical_events import normalize_mc_stat_int_row

        row = {
            "DeviceId": 123,
            "TimeStamp": "2024-01-15T10:30:00",
            "StatType": 1,  # battery_level
            "IntValue": 85,
            "ServerDateTime": "2024-01-15T10:30:05",
        }

        event = normalize_mc_stat_int_row(row, tenant_id="test")

        assert event.device_id == 123
        assert event.metric_name == "battery_level"
        assert event.metric_value == 85
        assert event.source_table == "DeviceStatInt"
        assert event.dimensions["stat_type_code"] == 1

    def test_mc_stat_type_mapping_uses_discovered(self):
        """Test StatType mapping uses discovered overrides."""
        from device_anomaly.data_access.canonical_events import normalize_mc_stat_int_row
        from device_anomaly.data_access.stat_type_mapper import (
            add_stat_type_mapping,
            clear_discovered_stat_types,
        )

        clear_discovered_stat_types()
        add_stat_type_mapping(99, "CustomMetric")

        row = {
            "DeviceId": 123,
            "TimeStamp": "2024-01-15T10:30:00",
            "StatType": 99,
            "IntValue": 7,
            "ServerDateTime": "2024-01-15T10:30:05",
        }

        event = normalize_mc_stat_int_row(row, tenant_id="test")
        assert event.metric_name == "custom_metric"

    def test_mc_location_normalization(self):
        """Test DeviceStatLocation row normalization."""
        from device_anomaly.data_access.canonical_events import normalize_mc_location_row

        row = {
            "DeviceId": 456,
            "TimeStamp": "2024-01-15T10:30:00",
            "Latitude": 37.7749,
            "Longitude": -122.4194,
            "Altitude": 10.5,
            "Heading": 180.0,
            "Speed": 5.5,
            "ServerDateTime": "2024-01-15T10:30:05",
        }

        events = normalize_mc_location_row(row, tenant_id="test")

        assert len(events) == 5  # lat, long, alt, heading, speed
        assert any(e.metric_name == "location_latitude" for e in events)
        assert any(e.metric_name == "location_speed" for e in events)

    def test_canonical_event_to_dict(self):
        """Test canonical event serialization."""
        from device_anomaly.data_access.canonical_events import (
            CanonicalEvent,
            MetricType,
            SourceDatabase,
        )

        event = CanonicalEvent(
            tenant_id="default",
            source_db=SourceDatabase.XSIGHT,
            source_table="cs_BatteryStat",
            device_id=789,
            event_time=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
            metric_name="total_battery_level_drop",
            metric_type=MetricType.INT,
            metric_value=15,
            dimensions={"app_id": 42},
        )

        data = event.to_dict()

        assert data["tenant_id"] == "default"
        assert data["source_db"] == "xsight"
        assert data["device_id"] == 789
        assert data["metric_name"] == "total_battery_level_drop"
        assert data["dimensions"]["app_id"] == 42
        assert len(data["event_id"]) == 32


class TestDataProfilerCurated:
    """Tests for data_profiler curated table functions."""

    def test_curated_xsight_tables_includes_extended(self):
        """Test get_curated_xsight_tables includes extended tables."""
        from device_anomaly.data_access.data_profiler import get_curated_xsight_tables

        tables = get_curated_xsight_tables(include_extended=True, include_discovered=False)

        # Should include original tables
        assert "cs_BatteryStat" in tables
        assert "cs_AppUsage" in tables

        # Should include extended tables
        assert "cs_DataUsageByHour" in tables
        assert "cs_WiFiLocation" in tables
        assert "cs_LastKnown" in tables

    def test_curated_mc_tables_includes_timeseries(self):
        """Test get_curated_mc_tables includes time-series tables."""
        from device_anomaly.data_access.data_profiler import get_curated_mc_tables

        tables = get_curated_mc_tables(include_discovered=False)

        assert "DeviceStatInt" in tables
        assert "DeviceStatLocation" in tables
        assert "MainLog" in tables
        assert "Alert" in tables

    def test_extended_tables_constant(self):
        """Test DW_TELEMETRY_TABLES_EXTENDED constant."""
        from device_anomaly.data_access.data_profiler import DW_TELEMETRY_TABLES_EXTENDED

        # Should have more than the original 5 tables
        assert len(DW_TELEMETRY_TABLES_EXTENDED) > 5

        # Should include high-value hourly tables
        assert "cs_DataUsageByHour" in DW_TELEMETRY_TABLES_EXTENDED
        assert "cs_BatteryLevelDrop" in DW_TELEMETRY_TABLES_EXTENDED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
