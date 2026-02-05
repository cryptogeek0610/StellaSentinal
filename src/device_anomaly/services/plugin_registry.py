import importlib
import logging
from typing import Dict, Any, Type
from device_anomaly.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

class PluginRegistry:
    """Dynamic registry for loading StellaSentinal data connectors."""
    def __init__(self):
        self._connectors: Dict[str, Type[BaseConnector]] = {}

    def register_connector(self, name: str, connector_cls: Type[BaseConnector]):
        self._connectors[name] = connector_cls
        logger.info(f"Registered connector: {name}")

    def get_connector(self, name: str) -> Type[BaseConnector]:
        return self._connectors.get(name)

    def load_from_path(self, module_path: str):
        """Load a connector plugin from a python module path."""
        try:
            module = importlib.import_module(module_path)
            # Implementation logic to discover and register classes
        except ImportError as e:
            logger.error(f"Failed to load plugin {module_path}: {e}")

plugin_registry = PluginRegistry()
