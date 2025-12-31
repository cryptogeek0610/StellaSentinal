"""Connector registry for managing data source connectors.

This module provides a factory pattern for registering and instantiating
data connectors. It allows dynamic registration of new connector types
and provides a unified interface for creating connector instances.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Type

from device_anomaly.connectors.base import BaseConnector, ConnectorConfig


class ConnectorRegistry:
    """Registry for data connectors.

    Provides a central registry for all available connector types.
    Connectors are registered by source_type and can be instantiated
    with their configurations.

    Example:
        # Register a connector
        ConnectorRegistry.register('xsight_dw', XSightConnector)

        # Create an instance
        config = ConnectorConfig(name='prod_xsight', source_type='xsight_dw', ...)
        connector = ConnectorRegistry.create(config)
    """

    _connectors: Dict[str, Type[BaseConnector]] = {}

    @classmethod
    def register(cls, source_type: str, connector_class: Type[BaseConnector]) -> None:
        """Register a connector class for a source type.

        Args:
            source_type: Unique identifier for the source (e.g., 'xsight_dw', 'mobicontrol_db')
            connector_class: The connector class to register

        Raises:
            ValueError: If source_type is already registered
        """
        if source_type in cls._connectors:
            raise ValueError(
                f"Connector for source_type '{source_type}' is already registered. "
                f"Use unregister() first to replace it."
            )
        cls._connectors[source_type] = connector_class

    @classmethod
    def unregister(cls, source_type: str) -> None:
        """Remove a connector from the registry.

        Args:
            source_type: The source type to unregister

        Raises:
            KeyError: If source_type is not registered
        """
        if source_type not in cls._connectors:
            raise KeyError(f"Connector for source_type '{source_type}' is not registered")
        del cls._connectors[source_type]

    @classmethod
    def get(cls, source_type: str) -> Type[BaseConnector]:
        """Get a connector class by source type.

        Args:
            source_type: The source type to look up

        Returns:
            The connector class

        Raises:
            KeyError: If source_type is not registered
        """
        if source_type not in cls._connectors:
            available = ", ".join(cls._connectors.keys()) or "none"
            raise KeyError(
                f"Connector for source_type '{source_type}' is not registered. "
                f"Available types: {available}"
            )
        return cls._connectors[source_type]

    @classmethod
    def create(cls, config: ConnectorConfig) -> BaseConnector:
        """Create a connector instance from configuration.

        Args:
            config: ConnectorConfig with source_type and connection details

        Returns:
            Configured connector instance

        Raises:
            KeyError: If config.source_type is not registered
        """
        connector_class = cls.get(config.source_type)
        return connector_class(config)

    @classmethod
    def list_connectors(cls) -> List[str]:
        """List all registered connector source types.

        Returns:
            List of registered source type names
        """
        return list(cls._connectors.keys())

    @classmethod
    def is_registered(cls, source_type: str) -> bool:
        """Check if a source type is registered.

        Args:
            source_type: The source type to check

        Returns:
            True if registered, False otherwise
        """
        return source_type in cls._connectors

    @classmethod
    def clear(cls) -> None:
        """Clear all registered connectors.

        Primarily useful for testing.
        """
        cls._connectors.clear()


class ConnectorManager:
    """Manages multiple connector instances.

    Provides a higher-level interface for working with multiple
    data sources simultaneously. Handles connection lifecycle
    and provides aggregated data loading.
    """

    def __init__(self):
        """Initialize the connector manager."""
        self._active_connectors: Dict[str, BaseConnector] = {}

    def add_connector(self, config: ConnectorConfig) -> BaseConnector:
        """Add and connect a new data source.

        Args:
            config: Configuration for the connector

        Returns:
            The connected connector instance

        Raises:
            ValueError: If a connector with this name already exists
        """
        if config.name in self._active_connectors:
            raise ValueError(f"Connector with name '{config.name}' already exists")

        connector = ConnectorRegistry.create(config)
        if config.enabled:
            connector.connect()
        self._active_connectors[config.name] = connector
        return connector

    def remove_connector(self, name: str) -> None:
        """Remove and disconnect a data source.

        Args:
            name: The connector name to remove

        Raises:
            KeyError: If connector name is not found
        """
        if name not in self._active_connectors:
            raise KeyError(f"Connector '{name}' not found")

        connector = self._active_connectors[name]
        if connector.is_connected:
            connector.disconnect()
        del self._active_connectors[name]

    def get_connector(self, name: str) -> BaseConnector:
        """Get a connector by name.

        Args:
            name: The connector name

        Returns:
            The connector instance

        Raises:
            KeyError: If connector name is not found
        """
        if name not in self._active_connectors:
            raise KeyError(f"Connector '{name}' not found")
        return self._active_connectors[name]

    def list_connectors(self) -> List[str]:
        """List all active connector names.

        Returns:
            List of connector names
        """
        return list(self._active_connectors.keys())

    def get_connected_sources(self) -> List[str]:
        """List names of all currently connected sources.

        Returns:
            List of connected connector names
        """
        return [
            name for name, connector in self._active_connectors.items()
            if connector.is_connected
        ]

    def disconnect_all(self) -> None:
        """Disconnect all active connectors."""
        for connector in self._active_connectors.values():
            if connector.is_connected:
                connector.disconnect()

    def connect_all(self) -> None:
        """Connect all enabled connectors."""
        for connector in self._active_connectors.values():
            if connector.config.enabled and not connector.is_connected:
                connector.connect()

    def health_check(self) -> Dict[str, bool]:
        """Check health of all connectors.

        Returns:
            Dict mapping connector names to health status
        """
        return {
            name: connector.test_connection() if connector.is_connected else False
            for name, connector in self._active_connectors.items()
        }

    def __enter__(self) -> "ConnectorManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - disconnect all."""
        self.disconnect_all()
