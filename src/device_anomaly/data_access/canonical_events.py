"""
Canonical telemetry event normalization.

Converts raw data from XSight and MobiControl into a unified event format
suitable for ML feature extraction and anomaly detection.

Canonical Event Schema:
- tenant_id: str - Tenant identifier
- source_db: str - Source database (xsight|mobicontrol)
- source_table: str - Original table name
- device_id: int - Device identifier
- event_time: datetime - Event timestamp (UTC)
- metric_name: str - Normalized metric name
- metric_type: str - Type (int|float|string|json)
- metric_value: Any - The actual value
- dimensions: dict - Additional context (app_id, network_type, etc.)
- event_id: str - SHA256 hash for idempotency (stable, deterministic)

PRODUCTION-HARDENED:
- event_id uses SHA256 with deterministic JSON serialization
- Includes dimensions in hash for true uniqueness
- Supports deduplication via event_id
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import pandas as pd

from device_anomaly.data_access.stat_type_mapper import get_stat_type_name

logger = logging.getLogger(__name__)


def _deterministic_json(obj: Any) -> str:
    """Convert object to deterministic JSON string.

    Keys are sorted, no whitespace, consistent formatting.
    This ensures the same data always produces the same string.
    """
    if obj is None:
        return "null"
    if isinstance(obj, (int, float, bool)):
        return json.dumps(obj)
    if isinstance(obj, str):
        return json.dumps(obj)
    if isinstance(obj, datetime):
        return json.dumps(obj.isoformat())
    if isinstance(obj, dict):
        sorted_items = sorted(obj.items(), key=lambda x: str(x[0]))
        return "{" + ",".join(
            f"{json.dumps(str(k))}:{_deterministic_json(v)}"
            for k, v in sorted_items
        ) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_deterministic_json(item) for item in obj) + "]"
    # Fallback: convert to string
    return json.dumps(str(obj))


def compute_stable_event_id(
    source_db: str,
    source_table: str,
    device_id: int,
    event_time: datetime,
    metric_name: str,
    metric_value: Any,
    dimensions: dict[str, Any],
) -> str:
    """Compute a stable SHA256 event_id for idempotency.

    The hash includes:
    - source_db, source_table
    - device_id
    - event_time (ISO format with timezone)
    - metric_name
    - metric_value (deterministic JSON)
    - dimensions (deterministic JSON with sorted keys)

    Returns first 32 characters of SHA256 hex digest.
    """
    # Ensure event_time has timezone
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=UTC)

    # Build hash input with deterministic serialization
    hash_parts = [
        source_db,
        source_table,
        str(device_id),
        event_time.isoformat(),
        metric_name,
        _deterministic_json(metric_value),
        _deterministic_json(dimensions),
    ]
    hash_input = "|".join(hash_parts)

    # SHA256 for cryptographic stability
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:32]


class MetricType(StrEnum):
    """Metric value types."""
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    JSON = "json"


class SourceDatabase(StrEnum):
    """Source database identifiers."""
    XSIGHT = "xsight"
    MOBICONTROL = "mobicontrol"


@dataclass
class CanonicalEvent:
    """
    Unified telemetry event format.

    All data from XSight and MobiControl is normalized to this format
    before feature extraction and ML processing.

    event_id is a stable SHA256 hash that uniquely identifies the event.
    It can be used for:
    - Deduplication during ingestion
    - Idempotent writes to storage
    - Tracking which events have been processed
    """
    tenant_id: str
    source_db: SourceDatabase
    source_table: str
    device_id: int
    event_time: datetime
    metric_name: str
    metric_type: MetricType
    metric_value: Any
    dimensions: dict[str, Any] = field(default_factory=dict)
    event_id: str = ""

    def __post_init__(self):
        """Generate event_id if not provided."""
        # Ensure event_time is timezone-aware first
        if self.event_time.tzinfo is None:
            self.event_time = self.event_time.replace(tzinfo=UTC)

        # Generate stable event_id if not provided
        if not self.event_id:
            self.event_id = compute_stable_event_id(
                source_db=self.source_db.value,
                source_table=self.source_table,
                device_id=self.device_id,
                event_time=self.event_time,
                metric_name=self.metric_name,
                metric_value=self.metric_value,
                dimensions=self.dimensions,
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "source_db": self.source_db.value,
            "source_table": self.source_table,
            "device_id": self.device_id,
            "event_time": self.event_time.isoformat(),
            "metric_name": self.metric_name,
            "metric_type": self.metric_type.value,
            "metric_value": self.metric_value,
            "dimensions": self.dimensions,
            "event_id": self.event_id,
        }

    def to_feature_row(self) -> dict[str, Any]:
        """Convert to feature row format for ML."""
        return {
            "device_id": self.device_id,
            "timestamp": self.event_time,
            self.metric_name: self.metric_value,
            **{f"dim_{k}": v for k, v in self.dimensions.items()},
        }

    # Backward compatibility alias
    @property
    def raw_hash(self) -> str:
        """Deprecated: Use event_id instead."""
        return self.event_id


# ============================================================================
# MobiControl StatType mappings
# ============================================================================

# DeviceStatInt StatType codes -> human readable names
MC_STAT_INT_TYPES: dict[int, str] = {
    # Battery metrics
    1: "battery_level",
    2: "battery_temperature",
    3: "battery_voltage",
    4: "battery_health",
    5: "battery_status",
    6: "charging_source",

    # Memory metrics
    10: "available_ram",
    11: "total_ram",
    12: "available_storage",
    13: "total_storage",
    14: "available_external_storage",
    15: "total_external_storage",

    # Network metrics
    20: "wifi_signal_strength",
    21: "cell_signal_strength",
    22: "network_type",
    23: "data_connection_state",

    # Device state
    30: "screen_state",
    31: "device_idle_state",
    32: "power_save_mode",

    # App metrics
    40: "running_app_count",
    41: "installed_app_count",

    # Security
    50: "security_patch_level",
    51: "encryption_status",
}

# DeviceStatString StatType codes -> human readable names
MC_STAT_STRING_TYPES: dict[int, str] = {
    1: "current_ssid",
    2: "current_bssid",
    3: "cell_operator",
    4: "cell_operator_name",
    5: "network_country",
    10: "foreground_app",
    11: "active_user",
    20: "device_mode",
    21: "compliance_state",
}


def _infer_metric_type(value: Any) -> MetricType:
    """Infer metric type from value."""
    if isinstance(value, bool):
        return MetricType.BOOL
    elif isinstance(value, int):
        return MetricType.INT
    elif isinstance(value, float):
        return MetricType.FLOAT
    elif isinstance(value, (dict, list)):
        return MetricType.JSON
    else:
        return MetricType.STRING


def _normalize_metric_name(name: str) -> str:
    """Normalize metric name to snake_case."""
    import re
    # Convert CamelCase to snake_case
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    # Remove special characters
    name = re.sub(r'[^a-z0-9_]', '_', name)
    # Remove duplicate underscores
    name = re.sub(r'_+', '_', name)
    return name.strip('_')


def normalize_xsight_row(
    row: dict[str, Any],
    source_table: str,
    tenant_id: str = "default",
    timestamp_col: str = "CollectedDate",
    device_id_col: str = "DeviceId",
    exclude_cols: set | None = None,
) -> Generator[CanonicalEvent, None, None]:
    """
    Convert a single XSight row to canonical events.

    Each numeric/meaningful column becomes a separate event.

    Args:
        row: Dictionary representing a row from XSight
        source_table: Source table name (e.g., cs_BatteryStat)
        tenant_id: Tenant identifier
        timestamp_col: Column containing timestamp
        device_id_col: Column containing device ID
        exclude_cols: Columns to exclude from conversion

    Yields:
        CanonicalEvent for each metric in the row
    """
    exclude_cols = exclude_cols or {
        timestamp_col, device_id_col, "Hour", "AppId", "ConnectionTypeId",
        "AccessPointId", "NetworkTypeId", "PresetAppId",
    }

    device_id = row.get(device_id_col)
    timestamp = row.get(timestamp_col)

    if device_id is None or timestamp is None:
        logger.debug(f"Skipping row without device_id or timestamp: {row}")
        return

    # Convert timestamp
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except Exception:
            timestamp = datetime.now(UTC)
    elif not isinstance(timestamp, datetime):
        timestamp = datetime.now(UTC)

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)

    # Extract dimensions (lookup columns)
    dimensions = {}
    dimension_cols = {"AppId", "ConnectionTypeId", "AccessPointId",
                      "NetworkTypeId", "PresetAppId", "Hour"}
    for col in dimension_cols:
        if col in row and row[col] is not None:
            dimensions[_normalize_metric_name(col)] = row[col]

    # Convert each metric column
    for col, value in row.items():
        if col in exclude_cols:
            continue
        if value is None:
            continue

        metric_name = _normalize_metric_name(col)
        metric_type = _infer_metric_type(value)

        # Skip non-numeric values for most XSight tables
        if metric_type == MetricType.STRING and len(str(value)) > 100:
            continue

        yield CanonicalEvent(
            tenant_id=tenant_id,
            source_db=SourceDatabase.XSIGHT,
            source_table=source_table,
            device_id=int(device_id),
            event_time=timestamp,
            metric_name=metric_name,
            metric_type=metric_type,
            metric_value=value,
            dimensions=dimensions.copy(),
        )


def normalize_mc_stat_int_row(
    row: dict[str, Any],
    tenant_id: str = "default",
) -> CanonicalEvent:
    """
    Convert a DeviceStatInt row to canonical event.

    Args:
        row: Dictionary with DeviceId, TimeStamp, StatType, IntValue, ServerDateTime

    Returns:
        CanonicalEvent
    """
    device_id = row.get("DeviceId")
    timestamp = row.get("ServerDateTime") or row.get("TimeStamp")
    stat_type = row.get("StatType", 0)
    int_value = row.get("IntValue", 0)

    # Map stat type to metric name (prefer discovered mappings, fallback to static map)
    metric_name = get_stat_type_name(stat_type)
    if metric_name.startswith("stat_type_"):
        metric_name = MC_STAT_INT_TYPES.get(stat_type, metric_name)

    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)

    return CanonicalEvent(
        tenant_id=tenant_id,
        source_db=SourceDatabase.MOBICONTROL,
        source_table="DeviceStatInt",
        device_id=int(device_id),
        event_time=timestamp,
        metric_name=metric_name,
        metric_type=MetricType.INT,
        metric_value=int_value,
        dimensions={"stat_type_code": stat_type},
    )


def normalize_mc_stat_string_row(
    row: dict[str, Any],
    tenant_id: str = "default",
) -> CanonicalEvent:
    """Convert a DeviceStatString row to canonical event."""
    device_id = row.get("DeviceId")
    timestamp = row.get("ServerDateTime") or row.get("TimeStamp")
    stat_type = row.get("StatType", 0)
    str_value = row.get("StrValue", "")

    metric_name = MC_STAT_STRING_TYPES.get(stat_type, f"stat_string_{stat_type}")

    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)

    return CanonicalEvent(
        tenant_id=tenant_id,
        source_db=SourceDatabase.MOBICONTROL,
        source_table="DeviceStatString",
        device_id=int(device_id),
        event_time=timestamp,
        metric_name=metric_name,
        metric_type=MetricType.STRING,
        metric_value=str_value,
        dimensions={"stat_type_code": stat_type},
    )


def normalize_mc_location_row(
    row: dict[str, Any],
    tenant_id: str = "default",
) -> list[CanonicalEvent]:
    """
    Convert a DeviceStatLocation row to canonical events.

    Returns multiple events: one for each location metric.
    """
    device_id = row.get("DeviceId")
    timestamp = row.get("ServerDateTime") or row.get("TimeStamp")

    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)

    events = []
    base_dims = {
        "latitude": row.get("Latitude"),
        "longitude": row.get("Longitude"),
    }

    # Location metrics
    metrics = [
        ("latitude", row.get("Latitude"), MetricType.FLOAT),
        ("longitude", row.get("Longitude"), MetricType.FLOAT),
        ("altitude", row.get("Altitude"), MetricType.FLOAT),
        ("heading", row.get("Heading"), MetricType.FLOAT),
        ("speed", row.get("Speed"), MetricType.FLOAT),
    ]

    for metric_name, value, metric_type in metrics:
        if value is not None:
            events.append(CanonicalEvent(
                tenant_id=tenant_id,
                source_db=SourceDatabase.MOBICONTROL,
                source_table="DeviceStatLocation",
                device_id=int(device_id),
                event_time=timestamp,
                metric_name=f"location_{metric_name}",
                metric_type=metric_type,
                metric_value=value,
                dimensions=base_dims,
            ))

    return events


def normalize_mc_net_traffic_row(
    row: dict[str, Any],
    tenant_id: str = "default",
) -> list[CanonicalEvent]:
    """
    Convert a DeviceStatNetTraffic row to canonical events.
    """
    device_id = row.get("DeviceId")
    timestamp = row.get("ServerDateTime") or row.get("TimeStamp")

    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)

    dimensions = {
        "interface_type": row.get("InterfaceType"),
        "interface_id": row.get("InterfaceID"),
        "application": row.get("Application"),
    }

    events = []

    # Upload/Download metrics
    if row.get("Upload") is not None:
        events.append(CanonicalEvent(
            tenant_id=tenant_id,
            source_db=SourceDatabase.MOBICONTROL,
            source_table="DeviceStatNetTraffic",
            device_id=int(device_id),
            event_time=timestamp,
            metric_name="network_upload_bytes",
            metric_type=MetricType.INT,
            metric_value=row["Upload"],
            dimensions=dimensions,
        ))

    if row.get("Download") is not None:
        events.append(CanonicalEvent(
            tenant_id=tenant_id,
            source_db=SourceDatabase.MOBICONTROL,
            source_table="DeviceStatNetTraffic",
            device_id=int(device_id),
            event_time=timestamp,
            metric_name="network_download_bytes",
            metric_type=MetricType.INT,
            metric_value=row["Download"],
            dimensions=dimensions,
        ))

    return events


def normalize_mc_main_log_row(
    row: dict[str, Any],
    tenant_id: str = "default",
) -> CanonicalEvent:
    """Convert a MainLog row to canonical event."""
    device_id = row.get("DeviceId")
    timestamp = row.get("DateTime")

    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)

    return CanonicalEvent(
        tenant_id=tenant_id,
        source_db=SourceDatabase.MOBICONTROL,
        source_table="MainLog",
        device_id=int(device_id) if device_id else 0,
        event_time=timestamp,
        metric_name="log_event",
        metric_type=MetricType.JSON,
        metric_value={
            "event_id": row.get("EventId"),
            "severity": row.get("Severity"),
            "event_class": row.get("EventClass"),
            "message": row.get("ResTxt", "")[:500],  # Truncate long messages
        },
        dimensions={
            "event_id": row.get("EventId"),
            "severity": row.get("Severity"),
            "event_class": row.get("EventClass"),
        },
    )


def normalize_mc_alert_row(
    row: dict[str, Any],
    tenant_id: str = "default",
) -> CanonicalEvent:
    """Convert an Alert row to canonical event."""
    timestamp = row.get("SetDateTime")

    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)

    # Extract device ID from DevId string if present
    dev_id = row.get("DevId", "")
    device_id = 0
    if dev_id and dev_id.isdigit():
        device_id = int(dev_id)

    return CanonicalEvent(
        tenant_id=tenant_id,
        source_db=SourceDatabase.MOBICONTROL,
        source_table="Alert",
        device_id=device_id,
        event_time=timestamp,
        metric_name="alert",
        metric_type=MetricType.JSON,
        metric_value={
            "alert_key": row.get("AlertKey"),
            "alert_name": row.get("AlertName"),
            "severity": row.get("AlertSeverity"),
            "status": row.get("Status"),
        },
        dimensions={
            "alert_key": row.get("AlertKey"),
            "severity": row.get("AlertSeverity"),
            "status": row.get("Status"),
        },
    )


def dataframe_to_canonical_events(
    df: pd.DataFrame,
    source_db: SourceDatabase,
    source_table: str,
    tenant_id: str = "default",
    timestamp_col: str | None = None,
    device_id_col: str = "DeviceId",
) -> list[CanonicalEvent]:
    """
    Convert a DataFrame to list of canonical events.

    Dispatches to appropriate normalizer based on source table.
    """
    events = []

    # Auto-detect timestamp column
    if timestamp_col is None:
        ts_candidates = ["ServerDateTime", "TimeStamp", "DateTime",
                        "CollectedDate", "CollectedTime", "SetDateTime"]
        for col in ts_candidates:
            if col in df.columns:
                timestamp_col = col
                break
        if timestamp_col is None:
            logger.warning(f"No timestamp column found in {source_table}")
            return events

    # Dispatch based on source table
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        try:
            if source_table == "DeviceStatInt":
                events.append(normalize_mc_stat_int_row(row_dict, tenant_id))
            elif source_table == "DeviceStatString":
                events.append(normalize_mc_stat_string_row(row_dict, tenant_id))
            elif source_table == "DeviceStatLocation":
                events.extend(normalize_mc_location_row(row_dict, tenant_id))
            elif source_table == "DeviceStatNetTraffic":
                events.extend(normalize_mc_net_traffic_row(row_dict, tenant_id))
            elif source_table == "MainLog":
                events.append(normalize_mc_main_log_row(row_dict, tenant_id))
            elif source_table == "Alert":
                events.append(normalize_mc_alert_row(row_dict, tenant_id))
            elif source_db == SourceDatabase.XSIGHT:
                events.extend(normalize_xsight_row(
                    row_dict, source_table, tenant_id, timestamp_col, device_id_col
                ))
            else:
                # Generic normalization for unknown tables
                events.extend(normalize_xsight_row(
                    row_dict, source_table, tenant_id, timestamp_col, device_id_col
                ))
        except Exception as e:
            logger.warning(f"Failed to normalize row from {source_table}: {e}")

    return events


def canonical_events_to_dataframe(events: list[CanonicalEvent]) -> pd.DataFrame:
    """
    Convert list of canonical events to a DataFrame.

    Useful for feature engineering and ML pipelines.
    """
    if not events:
        return pd.DataFrame()

    rows = [e.to_dict() for e in events]
    df = pd.DataFrame(rows)

    # Convert event_time to datetime
    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"])

    return df


def pivot_events_to_features(
    events: list[CanonicalEvent],
    time_bucket: str = "1h",
) -> pd.DataFrame:
    """
    Pivot canonical events into feature matrix.

    Groups by device_id and time bucket, with metric_names as columns.

    Args:
        events: List of canonical events
        time_bucket: Time bucket for aggregation (e.g., "1h", "1d")

    Returns:
        DataFrame with device_id, timestamp, and metric columns
    """
    if not events:
        return pd.DataFrame()

    df = canonical_events_to_dataframe(events)

    # Create time bucket
    df["time_bucket"] = df["event_time"].dt.floor(time_bucket)

    # Pivot - aggregate numeric values, take last for strings
    numeric_events = df[df["metric_type"].isin(["int", "float"])]

    if numeric_events.empty:
        return pd.DataFrame()

    # Group and aggregate
    pivot = numeric_events.pivot_table(
        index=["device_id", "time_bucket"],
        columns="metric_name",
        values="metric_value",
        aggfunc="mean",
    ).reset_index()

    pivot.columns.name = None

    return pivot


def dedupe_events(
    events: list[CanonicalEvent],
    seen_ids: set[str] | None = None,
) -> tuple[list[CanonicalEvent], set[str]]:
    """
    Deduplicate events based on event_id.

    Args:
        events: List of events to dedupe
        seen_ids: Optional set of already-seen event_ids (for incremental dedupe)

    Returns:
        Tuple of (deduped_events, all_seen_ids)
    """
    if seen_ids is None:
        seen_ids = set()

    deduped = []
    for event in events:
        if event.event_id not in seen_ids:
            seen_ids.add(event.event_id)
            deduped.append(event)

    if len(deduped) < len(events):
        logger.debug(
            f"Deduped {len(events) - len(deduped)} events "
            f"({len(deduped)}/{len(events)} unique)"
        )

    return deduped, seen_ids


def dedupe_events_df(df: pd.DataFrame, event_id_col: str = "event_id") -> pd.DataFrame:
    """
    Deduplicate a DataFrame of events based on event_id column.

    Args:
        df: DataFrame with event_id column
        event_id_col: Name of the event_id column

    Returns:
        DataFrame with duplicates removed (keeps first occurrence)
    """
    if df.empty or event_id_col not in df.columns:
        return df

    original_count = len(df)
    df_deduped = df.drop_duplicates(subset=[event_id_col], keep="first")

    if len(df_deduped) < original_count:
        logger.debug(
            f"Deduped {original_count - len(df_deduped)} rows "
            f"({len(df_deduped)}/{original_count} unique)"
        )

    return df_deduped


def filter_seen_events(
    events: list[CanonicalEvent],
    seen_ids: set[str],
) -> list[CanonicalEvent]:
    """
    Filter out events that have already been seen.

    This is useful for incremental processing where you want to
    skip events that were already processed in a previous batch.

    Args:
        events: List of events to filter
        seen_ids: Set of event_ids that have been processed

    Returns:
        List of events not in seen_ids
    """
    return [e for e in events if e.event_id not in seen_ids]
