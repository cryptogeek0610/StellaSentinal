"""
Tests for the anomaly drift fix.

These tests verify that:
1. Re-scoring the same data doesn't create duplicate anomalies
2. Temporal features don't drift when re-scoring static data
3. Data freshness checks prevent unnecessary scoring
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestTemporalFeatureStability:
    """Test that temporal features don't drift over time."""

    def test_days_since_last_checkin_uses_row_timestamp(self):
        """DaysSinceLastCheckin should be computed relative to row Timestamp, not scoring time."""
        from device_anomaly.data_access.unified_loader import _add_temporal_features

        # Create test data with fixed timestamps
        df = pd.DataFrame({
            "DeviceId": [1, 2, 3],
            "Timestamp": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 2),
                datetime(2025, 1, 3),
            ],
            "LastCheckInTime": [
                datetime(2024, 12, 31),  # 1 day before Timestamp
                datetime(2024, 12, 31),  # 2 days before Timestamp
                datetime(2024, 12, 31),  # 3 days before Timestamp
            ],
        })

        # Compute features with different end_dates (simulating different scoring runs)
        result1 = _add_temporal_features(df.copy(), "2025-01-10")  # 10 days after data
        result2 = _add_temporal_features(df.copy(), "2025-01-20")  # 20 days after data

        # The DaysSinceLastCheckin should be the same regardless of when scoring runs
        # because it's computed relative to the row's Timestamp, not the end_date
        pd.testing.assert_series_equal(
            result1["DaysSinceLastCheckin"],
            result2["DaysSinceLastCheckin"],
            check_names=False,
        )

    def test_days_since_last_checkin_values_correct(self):
        """Verify DaysSinceLastCheckin is computed correctly."""
        from device_anomaly.data_access.unified_loader import _add_temporal_features

        df = pd.DataFrame({
            "DeviceId": [1],
            "Timestamp": [datetime(2025, 1, 10)],
            "LastCheckInTime": [datetime(2025, 1, 5)],  # 5 days before
        })

        result = _add_temporal_features(df.copy(), "2025-01-20")

        # Should be 5 days (Timestamp - LastCheckInTime), not 15 days (end_date - LastCheckInTime)
        assert result["DaysSinceLastCheckin"].iloc[0] == 5

    def test_days_since_last_checkin_capped_at_365(self):
        """Verify DaysSinceLastCheckin is capped to prevent extreme values."""
        from device_anomaly.data_access.unified_loader import _add_temporal_features

        df = pd.DataFrame({
            "DeviceId": [1],
            "Timestamp": [datetime(2025, 1, 1)],
            "LastCheckInTime": [datetime(2020, 1, 1)],  # 5 years ago
        })

        result = _add_temporal_features(df.copy(), "2025-01-01")

        # Should be capped at 365
        assert result["DaysSinceLastCheckin"].iloc[0] == 365

    def test_disconnect_recency_uses_last_checkin_time(self):
        """DisconnectRecencyHours should be computed relative to LastCheckInTime."""
        from device_anomaly.data_access.unified_loader import _add_disconnect_flags

        df = pd.DataFrame({
            "DeviceId": [1],
            "LastCheckInTime": [datetime(2025, 1, 10, 12, 0)],
            "LastDisconnTime": [datetime(2025, 1, 10, 10, 0)],  # 2 hours before check-in
        })

        # Different end dates should produce the same result
        result1 = _add_disconnect_flags(df.copy(), "2025-01-15")
        result2 = _add_disconnect_flags(df.copy(), "2025-01-25")

        pd.testing.assert_series_equal(
            result1["DisconnectRecencyHours"],
            result2["DisconnectRecencyHours"],
            check_names=False,
        )


class TestDataFreshnessCheck:
    """Test the data freshness check logic."""

    def test_check_source_data_freshness_first_run(self):
        """First scoring run should always proceed."""
        from device_anomaly.data_access.watermark_store import WatermarkStore

        store = WatermarkStore(postgres_url=None, enable_file_fallback=False)
        store._get_stored_watermark = MagicMock(return_value=None)

        has_new_data, max_ts, reason = store.check_source_data_freshness()

        assert has_new_data is True
        assert "First scoring run" in reason or "No source watermarks" in reason

    def test_check_source_data_freshness_no_new_data(self):
        """Should skip scoring when source data hasn't changed."""
        from device_anomaly.data_access.watermark_store import WatermarkStore

        last_scoring_time = datetime(2025, 1, 10, 12, 0, tzinfo=timezone.utc)
        source_watermark = datetime(2025, 1, 10, 10, 0, tzinfo=timezone.utc)  # Before scoring

        store = WatermarkStore(postgres_url=None, enable_file_fallback=False)

        def mock_get_watermark(source_db, table_name):
            if source_db == "scoring" and table_name == "last_run":
                return last_scoring_time
            return source_watermark

        store._get_stored_watermark = mock_get_watermark

        has_new_data, max_ts, reason = store.check_source_data_freshness()

        assert has_new_data is False
        assert "No new data since last scoring" in reason

    def test_check_source_data_freshness_has_new_data(self):
        """Should proceed with scoring when new source data exists."""
        from device_anomaly.data_access.watermark_store import WatermarkStore

        last_scoring_time = datetime(2025, 1, 10, 12, 0, tzinfo=timezone.utc)
        source_watermark = datetime(2025, 1, 10, 14, 0, tzinfo=timezone.utc)  # After scoring

        store = WatermarkStore(postgres_url=None, enable_file_fallback=False)

        def mock_get_watermark(source_db, table_name):
            if source_db == "scoring" and table_name == "last_run":
                return last_scoring_time
            return source_watermark

        store._get_stored_watermark = mock_get_watermark

        has_new_data, max_ts, reason = store.check_source_data_freshness()

        assert has_new_data is True
        assert "New data available" in reason


class TestUpsertPersistence:
    """Test that upsert prevents duplicate anomalies."""

    def test_persist_anomaly_results_uses_upsert(self):
        """Verify that persist_anomaly_results uses PostgreSQL upsert."""
        # This test verifies the code structure - actual DB testing requires integration tests
        import inspect
        from device_anomaly.data_access.anomaly_persistence import persist_anomaly_results

        source = inspect.getsource(persist_anomaly_results)

        # Verify upsert-related patterns are present
        assert "on_conflict_do_update" in source, "Should use PostgreSQL ON CONFLICT DO UPDATE"
        assert "pg_insert" in source or "insert" in source, "Should use insert statement"
        assert "index_elements" in source, "Should specify conflict index elements"

    def test_persistence_result_dataclass(self):
        """Verify PersistenceResult dataclass exists and has expected fields."""
        from device_anomaly.data_access.anomaly_persistence import PersistenceResult

        result = PersistenceResult()

        assert hasattr(result, "total_processed")
        assert hasattr(result, "new_inserted")
        assert hasattr(result, "existing_updated")
        assert hasattr(result, "errors")


class TestSchemaConstraint:
    """Test that the schema has the unique constraint defined."""

    def test_anomaly_result_has_unique_constraint(self):
        """Verify AnomalyResult schema defines the unique constraint."""
        from device_anomaly.database.schema import AnomalyResult

        # Check that __table_args__ is defined
        assert hasattr(AnomalyResult, "__table_args__")

        # Check for Index in table args
        table_args = AnomalyResult.__table_args__
        assert isinstance(table_args, tuple)

        # Find the unique index
        from sqlalchemy import Index
        unique_indexes = [
            arg for arg in table_args
            if isinstance(arg, Index) and arg.unique
        ]

        assert len(unique_indexes) > 0, "Should have at least one unique index"

        # Verify the index covers the right columns
        unique_idx = unique_indexes[0]
        column_names = [c.name for c in unique_idx.columns]
        assert "tenant_id" in column_names
        assert "device_id" in column_names
        assert "timestamp" in column_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
