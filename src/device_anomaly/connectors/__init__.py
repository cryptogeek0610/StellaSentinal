"""Data connectors for multiple telemetry sources.

This module provides a pluggable connector architecture for ingesting
telemetry data from various sources:
- XSight Database (SQL Server)
- MobiControl (SQL Database + REST API)
- Synthetic data (for testing)

Usage:
    from device_anomaly.connectors import ConnectorRegistry

    # Get a connector by name
    connector_class = ConnectorRegistry.get('xsight_dw')
    connector = connector_class(config)
    connector.connect()
    df = connector.load_telemetry(start_date, end_date)
"""

from device_anomaly.connectors.base import BaseConnector, ConnectorConfig
from device_anomaly.connectors.registry import ConnectorManager, ConnectorRegistry

__all__ = [
    "BaseConnector",
    "ConnectorConfig",
    "ConnectorManager",
    "ConnectorRegistry",
]
