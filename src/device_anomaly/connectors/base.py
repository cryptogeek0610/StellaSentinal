"""Base connector interface for all data sources.

This module defines the abstract base class that all connectors must implement.
It ensures a consistent API across XSight, MobiControl, and other data sources.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

# Canonical column names expected in all telemetry DataFrames
CANONICAL_COLUMNS = [
    "device_id",
    "tenant_id",
    "timestamp",
    "source",
]


@dataclass
class ConnectorConfig:
    """Configuration for a data connector."""

    name: str
    source_type: str  # 'xsight_dw', 'mobicontrol_db', 'mobicontrol_api', 'synthetic'
    enabled: bool = True

    # Connection settings (populated from environment/config)
    host: str | None = None
    port: int | None = None
    database: str | None = None
    username: str | None = None
    password: str | None = None

    # API settings (for REST connectors)
    base_url: str | None = None
    client_id: str | None = None
    client_secret: str | None = None

    # Rate limiting and retry settings
    rate_limit_requests_per_second: float = 10.0
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0

    # Additional connector-specific settings
    extra: dict[str, Any] = field(default_factory=dict)


class BaseConnector(ABC):
    """Abstract base class for all data connectors.

    All connectors must implement these methods to ensure a consistent
    interface for data ingestion across XSight, MobiControl, and other sources.

    Example:
        class XSightConnector(BaseConnector):
            def connect(self):
                self.engine = create_engine(self.connection_string)
                self._connected = True

            def load_telemetry(self, start_date, end_date, ...):
                query = self._build_query(start_date, end_date)
                df = pd.read_sql(query, self.engine)
                return self._normalize_columns(df)
    """

    def __init__(self, config: ConnectorConfig):
        """Initialize the connector with configuration.

        Args:
            config: ConnectorConfig with connection details
        """
        self.config = config
        self._connected = False
        self._last_error: str | None = None

    @property
    def is_connected(self) -> bool:
        """Check if connector is currently connected."""
        return self._connected

    @property
    def source_name(self) -> str:
        """Return the source name (e.g., 'xsight', 'mobicontrol')."""
        return self.config.source_type

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the data source.

        Raises:
            ConnectionError: If connection cannot be established
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the data source."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if connection is healthy.

        Returns:
            True if connection is healthy, False otherwise
        """
        pass

    @abstractmethod
    def load_telemetry(
        self,
        start_date: datetime,
        end_date: datetime,
        device_ids: list[str] | None = None,
        tenant_id: str | None = None,
        metrics: list[str] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Load telemetry data from the source.

        Args:
            start_date: Start of time range (inclusive)
            end_date: End of time range (inclusive)
            device_ids: Optional list of device IDs to filter
            tenant_id: Optional tenant ID for multi-tenant filtering
            metrics: Optional list of metric names to include
            limit: Optional row limit for safety

        Returns:
            DataFrame with columns: device_id, tenant_id, timestamp, source, + metric columns

        Raises:
            ConnectionError: If not connected
            ValueError: If invalid parameters
        """
        pass

    @abstractmethod
    def list_devices(
        self,
        tenant_id: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """List available devices from the source.

        Args:
            tenant_id: Optional tenant ID filter
            limit: Optional row limit

        Returns:
            DataFrame with device information
        """
        pass

    @abstractmethod
    def list_metrics(self) -> list[dict[str, Any]]:
        """List available metrics from this source.

        Returns:
            List of metric definitions with name, category, unit, data_type
        """
        pass

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate that a DataFrame matches the canonical schema.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        if df is None or df.empty:
            return True  # Empty is valid

        # Check required columns
        required = ["device_id", "timestamp"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            self._last_error = f"Missing required columns: {missing}"
            return False

        # Validate timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            self._last_error = "Column 'timestamp' must be datetime type"
            return False

        return True

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to canonical format.

        Converts source-specific column names (e.g., 'DeviceId', 'Timestamp')
        to canonical format (e.g., 'device_id', 'timestamp').

        Args:
            df: DataFrame with source-specific column names

        Returns:
            DataFrame with normalized column names
        """
        # Common column mappings
        column_mappings = {
            # XSight columns
            "DeviceId": "device_id",
            "Timestamp": "timestamp",
            "TenantId": "tenant_id",
            "TotalBatteryLevelDrop": "battery_drop",
            "TotalFreeStorageKb": "free_storage_kb",
            "Download": "download_bytes",
            "Upload": "upload_bytes",
            "OfflineTime": "offline_time_min",
            "DisconnectCount": "disconnect_count",
            "WiFiSignalStrength": "wifi_signal_dbm",
            "ConnectionTime": "connection_time_min",
            # MobiControl columns (add mappings as needed)
        }

        df = df.copy()
        df.rename(columns=column_mappings, inplace=True)

        # Add source column if not present
        if "source" not in df.columns:
            df["source"] = self.source_name

        return df

    def get_metadata(self) -> dict[str, Any]:
        """Return connector metadata for observability.

        Returns:
            Dict with connector info: source, version, connection status, etc.
        """
        return {
            "name": self.config.name,
            "source_type": self.config.source_type,
            "enabled": self.config.enabled,
            "connected": self._connected,
            "last_error": self._last_error,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(source='{self.config.source_type}', connected={self._connected})>"
