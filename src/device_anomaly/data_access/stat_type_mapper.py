"""
StatType Mapper for MobiControl DeviceStatInt table.

Maps integer StatType codes to human-readable metric names.
Supports runtime discovery from database.

Known StatType codes (MobiControl standard):
- 1: BatteryLevel
- 2: TotalStorage
- 3: AvailableStorage
- 4: TotalRAM
- 5: AvailableRAM
- etc.
"""
from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# Known StatType mappings from MobiControl documentation
# These are common across deployments but may vary
KNOWN_STAT_TYPES: dict[int, str] = {
    1: "battery_level",
    2: "total_storage",
    3: "available_storage",
    4: "total_ram",
    5: "available_ram",
    6: "cpu_usage",
    7: "memory_usage",
    8: "wifi_signal_strength",
    9: "cellular_signal_strength",
    10: "device_temperature",
    11: "battery_temperature",
    12: "uptime_seconds",
    13: "screen_on_time",
    14: "network_rx_bytes",
    15: "network_tx_bytes",
}

# Cache for discovered stat types from database
_discovered_stat_types: dict[int, str] = {}


def get_stat_type_name(stat_type: int) -> str:
    """
    Get human-readable name for a StatType code.

    Args:
        stat_type: Integer StatType code from DeviceStatInt

    Returns:
        Human-readable metric name (e.g., "battery_level")
        Falls back to "stat_type_{code}" if unknown
    """
    # Check discovered types first (from database)
    if stat_type in _discovered_stat_types:
        return _discovered_stat_types[stat_type]

    # Check known types
    if stat_type in KNOWN_STAT_TYPES:
        return KNOWN_STAT_TYPES[stat_type]

    # Fallback to generic name
    return f"stat_type_{stat_type}"


def discover_stat_types(engine: Engine) -> dict[int, str]:
    """
    Discover StatType mappings from MobiControl database.

    Queries the StatTypes table (if it exists) to get the full mapping.
    Falls back to known types if table doesn't exist.

    Args:
        engine: SQLAlchemy engine connected to MobiControlDB

    Returns:
        Dictionary mapping StatType codes to names
    """
    global _discovered_stat_types

    # Try to query StatTypes table
    query = text("""
        SELECT StatTypeId, StatTypeName
        FROM dbo.StatTypes
        WHERE StatTypeName IS NOT NULL
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query).fetchall()

            discovered = {}
            for row in result:
                stat_type_id = row[0]
                stat_type_name = _normalize_stat_type_name(row[1])
                discovered[stat_type_id] = stat_type_name

            if discovered:
                _discovered_stat_types = discovered
                logger.info(f"Discovered {len(discovered)} StatTypes from database")
                return discovered

    except Exception as e:
        logger.debug(f"Could not discover StatTypes from database: {e}")

    # Try DeviceStatType table (alternate schema)
    query2 = text("""
        SELECT Id, Name
        FROM dbo.DeviceStatType
        WHERE Name IS NOT NULL
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query2).fetchall()

            discovered = {}
            for row in result:
                stat_type_id = row[0]
                stat_type_name = _normalize_stat_type_name(row[1])
                discovered[stat_type_id] = stat_type_name

            if discovered:
                _discovered_stat_types = discovered
                logger.info(f"Discovered {len(discovered)} StatTypes from DeviceStatType table")
                return discovered

    except Exception as e:
        logger.debug(f"Could not discover from DeviceStatType: {e}")

    # Fall back to known types
    logger.info("Using known StatType mappings (database discovery unavailable)")
    return KNOWN_STAT_TYPES.copy()


def _normalize_stat_type_name(name: str) -> str:
    """
    Normalize StatType name to snake_case metric name.

    Examples:
        "BatteryLevel" -> "battery_level"
        "Total Storage" -> "total_storage"
        "WiFi Signal" -> "wifi_signal"
    """
    import re

    if not name:
        return ""

    # Replace spaces with underscores
    name = name.replace(" ", "_")

    # Convert CamelCase to snake_case
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)

    # Lowercase
    name = name.lower()

    # Remove non-alphanumeric except underscores
    name = re.sub(r"[^a-z0-9_]", "", name)

    # Remove consecutive underscores
    name = re.sub(r"_+", "_", name)

    return name.strip("_")


def get_all_stat_types() -> dict[int, str]:
    """
    Get all known StatType mappings.

    Returns discovered types if available, otherwise known types.
    """
    if _discovered_stat_types:
        return _discovered_stat_types.copy()
    return KNOWN_STAT_TYPES.copy()


def add_stat_type_mapping(stat_type: int, name: str) -> None:
    """
    Add or update a StatType mapping at runtime.

    Use this for custom stat types specific to a deployment.
    """
    _discovered_stat_types[stat_type] = _normalize_stat_type_name(name)


def clear_discovered_stat_types() -> None:
    """Clear discovered stat types (for testing)."""
    global _discovered_stat_types
    _discovered_stat_types = {}
