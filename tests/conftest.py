"""
Shared pytest fixtures for the AnomalyDetection test suite.

This file centralizes common test fixtures to reduce duplication
and ensure consistent test setup across all test modules.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tiny_telemetry_path(fixtures_dir: Path) -> Path:
    """Return the path to the tiny telemetry CSV fixture."""
    return fixtures_dir / "tiny_telemetry.csv"


@pytest.fixture
def tiny_telemetry_df(tiny_telemetry_path: Path) -> pd.DataFrame:
    """Load the tiny telemetry CSV as a DataFrame with parsed timestamps."""
    df = pd.read_csv(tiny_telemetry_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory for watermarks, etc."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Create a temporary model artifacts directory."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    return model_dir


@pytest.fixture
def temp_results_db(tmp_path: Path) -> Path:
    """Create a temporary results database path and reset the connection."""
    db_path = tmp_path / "results.db"
    os.environ["RESULTS_DB_PATH"] = str(db_path)

    # Reset database connection singletons
    from device_anomaly.database import connection as db_connection
    db_connection._ENGINE = None
    db_connection._SESSION_FACTORY = None

    return db_path


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_telemetry_df() -> pd.DataFrame:
    """Generate a synthetic telemetry DataFrame with realistic patterns."""
    np.random.seed(42)
    n_rows = 100

    return pd.DataFrame({
        "DeviceId": np.repeat([1, 2, 3, 4, 5], n_rows // 5),
        "Timestamp": pd.date_range("2025-01-01", periods=n_rows, freq="H"),
        "TotalBatteryLevelDrop": np.random.normal(10, 2, size=n_rows),
        "Rssi": np.random.normal(-60, 5, size=n_rows),
        "AvgCpuUsage": np.random.normal(30, 10, size=n_rows),
        "AvgMemUsage": np.random.normal(50, 15, size=n_rows),
    })


@pytest.fixture
def sample_data_with_anomalies() -> pd.DataFrame:
    """Generate telemetry data with known anomalies for testing detection."""
    np.random.seed(42)

    # Normal data
    normal = pd.DataFrame({
        "DeviceId": 1,
        "Timestamp": pd.date_range("2025-01-01", periods=30, freq="D"),
        "TotalBatteryLevelDrop": np.random.normal(10, 1, size=30),
        "Rssi": np.random.normal(-60, 2, size=30),
    })

    # Anomalous data point
    anomaly = pd.DataFrame({
        "DeviceId": [1],
        "Timestamp": [pd.Timestamp("2025-02-01")],
        "TotalBatteryLevelDrop": [120],  # Extreme value
        "Rssi": [-110],  # Extreme value
    })

    return pd.concat([normal, anomaly], ignore_index=True)


# =============================================================================
# Baseline Fixtures
# =============================================================================

@pytest.fixture
def sample_baselines() -> Dict[str, Any]:
    """Return a sample baselines payload for testing."""
    return {
        "schema_version": "data_driven_v1",
        "baseline_type": "data_driven",
        "generated_at": "2025-01-01T00:00:00Z",
        "baselines": {
            "TotalBatteryLevelDrop": {
                "global": {"median": 5.0, "mad": 1.0, "sample_count": 10},
                "by_device_type": {},
                "by_hour": None,
                "thresholds": {"p95": 9.5, "p99": 12.0},
            },
            "Rssi": {
                "global": {"median": -60.0, "mad": 5.0, "sample_count": 10},
                "by_device_type": {},
                "by_hour": None,
                "thresholds": {"p95": -50.0, "p99": -45.0},
            },
        },
    }


@pytest.fixture
def baselines_file(temp_model_dir: Path, sample_baselines: Dict[str, Any]) -> Path:
    """Create a baselines.json file in the temp model directory."""
    baselines_path = temp_model_dir / "baselines.json"
    baselines_path.write_text(json.dumps(sample_baselines))
    return baselines_path


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_db_cursor() -> MagicMock:
    """Create a mock database cursor for schema discovery tests."""
    cursor = MagicMock()
    cursor._results = []
    cursor._index = 0

    def fetchone():
        if cursor._index < len(cursor._results):
            result = cursor._results[cursor._index]
            cursor._index += 1
            return result
        return None

    def fetchall():
        return cursor._results

    cursor.fetchone = fetchone
    cursor.fetchall = fetchall
    return cursor


@pytest.fixture
def mock_db_connection(mock_db_cursor: MagicMock) -> MagicMock:
    """Create a mock database connection."""
    connection = MagicMock()
    connection.cursor.return_value = mock_db_cursor
    return connection


class DummyWatermarkStore:
    """A simple in-memory watermark store for testing."""

    def __init__(self):
        self._watermarks: Dict[str, datetime] = {}

    def get_watermark(self, source: str, table: str) -> datetime | None:
        key = f"{source}:{table}"
        return self._watermarks.get(key)

    def set_watermark(self, source: str, table: str, ts: datetime) -> tuple[bool, str | None]:
        key = f"{source}:{table}"
        current = self._watermarks.get(key)

        if current and ts < current:
            return False, f"Monotonic violation: {ts} < {current}"

        self._watermarks[key] = ts
        return True, None

    def reset_watermark(self, source: str, table: str, ts: datetime) -> tuple[bool, str | None]:
        key = f"{source}:{table}"
        self._watermarks[key] = ts
        return True, None


@pytest.fixture
def dummy_watermark_store() -> DummyWatermarkStore:
    """Create a dummy in-memory watermark store for testing."""
    return DummyWatermarkStore()


# =============================================================================
# API Client Fixtures
# =============================================================================

@pytest.fixture
def api_client(temp_results_db: Path):
    """Create a FastAPI test client with isolated database."""
    from fastapi.testclient import TestClient
    from device_anomaly.api.main import app

    return TestClient(app)


@pytest.fixture
def api_client_with_baselines(api_client, temp_model_dir: Path, baselines_file: Path, monkeypatch):
    """Create an API client with baselines configured."""
    monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(temp_model_dir))
    return api_client


# =============================================================================
# Pytest Markers Registration
# =============================================================================

def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no external dependencies)")
    config.addinivalue_line("markers", "integration: Integration tests (may require database/services)")
    config.addinivalue_line("markers", "slow: Slow tests (ML training, large data processing)")
