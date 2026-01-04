"""Location mapping service for device-to-location resolution.

Maps devices to locations using configured mapping strategies:
1. custom_attribute: Match device CustomAttributes[attribute_name] == value
2. label: Match LabelDevice entries
3. device_group: Match DeviceGroupId
4. geo_fence: Match lat/lon within radius

Enables Carl's requirement: "Relate any anomalies to location (warehouse 1 vs warehouse 2)"
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy.orm import Session

from device_anomaly.database.schema import LocationMappingType, LocationMetadata

logger = logging.getLogger(__name__)


@dataclass
class ShiftSchedule:
    """A single shift schedule definition."""

    name: str
    start: str  # "HH:MM" format
    end: str  # "HH:MM" format

    def covers_time(self, time: datetime) -> bool:
        """Check if a given time falls within this shift."""
        time_str = time.strftime("%H:%M")
        # Handle overnight shifts (end < start)
        if self.end < self.start:
            return time_str >= self.start or time_str < self.end
        return self.start <= time_str < self.end

    def duration_hours(self) -> float:
        """Calculate shift duration in hours."""
        start_parts = self.start.split(":")
        end_parts = self.end.split(":")
        start_minutes = int(start_parts[0]) * 60 + int(start_parts[1])
        end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])
        if end_minutes < start_minutes:
            end_minutes += 24 * 60  # Overnight shift
        return (end_minutes - start_minutes) / 60.0


@dataclass
class LocationConfig:
    """Parsed location configuration from LocationMetadata."""

    location_id: str
    location_name: str
    parent_region: Optional[str]
    timezone: str
    mapping_type: LocationMappingType
    mapping_attribute: Optional[str]
    mapping_value: Optional[str]
    device_group_id: Optional[int]
    geo_fence: Optional[Dict[str, float]]  # {lat, lon, radius_m}
    shifts: List[ShiftSchedule] = field(default_factory=list)
    baselines: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_db_model(cls, model: LocationMetadata) -> "LocationConfig":
        """Create LocationConfig from database model."""
        # Parse shift schedules
        shifts = []
        if model.shift_schedules_json:
            try:
                shift_data = json.loads(model.shift_schedules_json)
                shifts = [ShiftSchedule(**s) for s in shift_data]
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse shift schedules for {model.location_id}: {e}")

        # Parse geo fence
        geo_fence = None
        if model.geo_fence_json:
            try:
                geo_fence = json.loads(model.geo_fence_json)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse geo fence for {model.location_id}: {e}")

        # Collect baselines
        baselines = {}
        if model.baseline_battery_drain_per_hour is not None:
            baselines["battery_drain_per_hour"] = model.baseline_battery_drain_per_hour
        if model.baseline_disconnect_rate is not None:
            baselines["disconnect_rate"] = model.baseline_disconnect_rate
        if model.baseline_drop_rate is not None:
            baselines["drop_rate"] = model.baseline_drop_rate

        return cls(
            location_id=model.location_id,
            location_name=model.location_name,
            parent_region=model.parent_region,
            timezone=model.timezone or "UTC",
            mapping_type=LocationMappingType(model.mapping_type),
            mapping_attribute=model.mapping_attribute,
            mapping_value=model.mapping_value,
            device_group_id=model.device_group_id,
            geo_fence=geo_fence,
            shifts=shifts,
            baselines=baselines,
        )


@dataclass
class DeviceLocationResult:
    """Result of device-to-location mapping."""

    device_id: int
    location_id: Optional[str]
    location_name: Optional[str]
    confidence: float  # 0.0 to 1.0
    mapping_type: Optional[str]
    current_shift: Optional[ShiftSchedule] = None
    shift_remaining_hours: Optional[float] = None


class LocationMapper:
    """Maps devices to locations using configured strategies.

    Strategies:
    1. custom_attribute: Match device CustomAttributes[attribute_name] == value
    2. label: Match LabelDevice entries
    3. device_group: Match DeviceGroupId
    4. geo_fence: Match lat/lon within radius

    Usage:
        mapper = LocationMapper(db_session, tenant_id="default")
        location = mapper.get_device_location(device_id=123, device_data={"DeviceGroupId": 5})

        # Bulk mapping for DataFrames
        df_with_locations = mapper.bulk_map_devices(devices_df)
    """

    def __init__(self, db_session: Session, tenant_id: str):
        """Initialize the location mapper.

        Args:
            db_session: SQLAlchemy database session
            tenant_id: Tenant ID for multi-tenant filtering
        """
        self.db = db_session
        self.tenant_id = tenant_id
        self._location_cache: Dict[str, LocationConfig] = {}
        self._cache_loaded = False

    def refresh_cache(self) -> None:
        """Reload location configurations from database."""
        self._location_cache.clear()

        locations = (
            self.db.query(LocationMetadata)
            .filter(
                LocationMetadata.tenant_id == self.tenant_id,
                LocationMetadata.is_active == True,  # noqa: E712
            )
            .all()
        )

        for loc in locations:
            config = LocationConfig.from_db_model(loc)
            self._location_cache[loc.location_id] = config

        self._cache_loaded = True
        logger.info(f"Loaded {len(self._location_cache)} location configurations for tenant {self.tenant_id}")

    def _ensure_cache_loaded(self) -> None:
        """Ensure location cache is loaded."""
        if not self._cache_loaded:
            self.refresh_cache()

    def get_all_locations(self) -> List[LocationConfig]:
        """Get all configured locations."""
        self._ensure_cache_loaded()
        return list(self._location_cache.values())

    def get_location_config(self, location_id: str) -> Optional[LocationConfig]:
        """Get configuration for a specific location."""
        self._ensure_cache_loaded()
        return self._location_cache.get(location_id)

    def get_device_location(
        self,
        device_id: int,
        device_data: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> DeviceLocationResult:
        """Determine location_id for a device based on mapping rules.

        Args:
            device_id: The device ID
            device_data: Device metadata dict containing:
                - CustomAttributes: dict of custom attribute key-values
                - LabelDevice: list of label strings
                - DeviceGroupId: int device group ID
                - Latitude, Longitude: float coordinates
            timestamp: Optional timestamp for shift calculation

        Returns:
            DeviceLocationResult with location info and confidence
        """
        self._ensure_cache_loaded()

        for location_id, config in self._location_cache.items():
            match_result = self._check_device_match(device_data, config)
            if match_result[0]:  # matched
                # Calculate current shift if timestamp provided
                current_shift = None
                shift_remaining = None
                if timestamp and config.shifts:
                    for shift in config.shifts:
                        if shift.covers_time(timestamp):
                            current_shift = shift
                            # Calculate remaining hours (simplified)
                            shift_remaining = self._calculate_shift_remaining(timestamp, shift)
                            break

                return DeviceLocationResult(
                    device_id=device_id,
                    location_id=location_id,
                    location_name=config.location_name,
                    confidence=match_result[1],
                    mapping_type=config.mapping_type.value,
                    current_shift=current_shift,
                    shift_remaining_hours=shift_remaining,
                )

        # No match found
        return DeviceLocationResult(
            device_id=device_id,
            location_id=None,
            location_name=None,
            confidence=0.0,
            mapping_type=None,
        )

    def _check_device_match(
        self, device_data: Dict[str, Any], config: LocationConfig
    ) -> Tuple[bool, float]:
        """Check if device matches a location's mapping rules.

        Returns:
            Tuple of (matched: bool, confidence: float)
        """
        if config.mapping_type == LocationMappingType.CUSTOM_ATTRIBUTE:
            return self._match_custom_attribute(device_data, config)
        elif config.mapping_type == LocationMappingType.LABEL:
            return self._match_label(device_data, config)
        elif config.mapping_type == LocationMappingType.DEVICE_GROUP:
            return self._match_device_group(device_data, config)
        elif config.mapping_type == LocationMappingType.GEO_FENCE:
            return self._match_geo_fence(device_data, config)
        else:
            logger.warning(f"Unknown mapping type: {config.mapping_type}")
            return False, 0.0

    def _match_custom_attribute(
        self, device_data: Dict[str, Any], config: LocationConfig
    ) -> Tuple[bool, float]:
        """Match device by CustomAttributes."""
        custom_attrs = device_data.get("CustomAttributes", {})
        if not isinstance(custom_attrs, dict):
            return False, 0.0

        attr_value = custom_attrs.get(config.mapping_attribute)
        if attr_value is not None and str(attr_value) == str(config.mapping_value):
            return True, 1.0  # High confidence for exact match
        return False, 0.0

    def _match_label(
        self, device_data: Dict[str, Any], config: LocationConfig
    ) -> Tuple[bool, float]:
        """Match device by LabelDevice entries."""
        labels = device_data.get("LabelDevice", [])
        if not isinstance(labels, list):
            return False, 0.0

        # Check if any label matches the configured value
        for label in labels:
            if str(label) == str(config.mapping_value):
                return True, 1.0
        return False, 0.0

    def _match_device_group(
        self, device_data: Dict[str, Any], config: LocationConfig
    ) -> Tuple[bool, float]:
        """Match device by DeviceGroupId."""
        device_group_id = device_data.get("DeviceGroupId")
        if device_group_id is not None and int(device_group_id) == config.device_group_id:
            return True, 1.0
        return False, 0.0

    def _match_geo_fence(
        self, device_data: Dict[str, Any], config: LocationConfig
    ) -> Tuple[bool, float]:
        """Match device by geographic location within radius."""
        if not config.geo_fence:
            return False, 0.0

        device_lat = device_data.get("Latitude")
        device_lon = device_data.get("Longitude")

        if device_lat is None or device_lon is None:
            return False, 0.0

        try:
            device_lat = float(device_lat)
            device_lon = float(device_lon)
            fence_lat = float(config.geo_fence.get("lat", 0))
            fence_lon = float(config.geo_fence.get("lon", 0))
            radius_m = float(config.geo_fence.get("radius_m", 0))
        except (ValueError, TypeError):
            return False, 0.0

        # Calculate distance using Haversine formula
        distance = self._haversine_distance(device_lat, device_lon, fence_lat, fence_lon)

        if distance <= radius_m:
            # Confidence decreases with distance from center
            confidence = max(0.5, 1.0 - (distance / radius_m) * 0.5)
            return True, confidence
        return False, 0.0

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in meters between two lat/lon points."""
        R = 6371000  # Earth's radius in meters

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _calculate_shift_remaining(self, timestamp: datetime, shift: ShiftSchedule) -> float:
        """Calculate remaining hours in the current shift."""
        current_time = timestamp.strftime("%H:%M")
        end_parts = shift.end.split(":")
        current_parts = current_time.split(":")

        end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])
        current_minutes = int(current_parts[0]) * 60 + int(current_parts[1])

        # Handle overnight shifts
        if end_minutes < int(shift.start.split(":")[0]) * 60:
            if current_minutes < end_minutes:
                remaining = end_minutes - current_minutes
            else:
                remaining = (24 * 60 - current_minutes) + end_minutes
        else:
            remaining = end_minutes - current_minutes

        return max(0, remaining / 60.0)

    def bulk_map_devices(
        self,
        devices_df: pd.DataFrame,
        timestamp: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Add location_id column to devices DataFrame.

        Args:
            devices_df: DataFrame with device data. Expected columns:
                - device_id or DeviceId
                - CustomAttributes (optional)
                - LabelDevice (optional)
                - DeviceGroupId (optional)
                - Latitude, Longitude (optional)
            timestamp: Optional timestamp for shift calculation

        Returns:
            DataFrame with added columns:
                - location_id
                - location_name
                - location_confidence
                - current_shift_name (if timestamp provided)
                - shift_remaining_hours (if timestamp provided)
        """
        self._ensure_cache_loaded()

        # Initialize result columns
        results = {
            "location_id": [],
            "location_name": [],
            "location_confidence": [],
            "current_shift_name": [],
            "shift_remaining_hours": [],
        }

        # Determine device_id column name
        device_id_col = "device_id" if "device_id" in devices_df.columns else "DeviceId"

        for _, row in devices_df.iterrows():
            device_id = row.get(device_id_col)
            device_data = row.to_dict()

            result = self.get_device_location(device_id, device_data, timestamp)

            results["location_id"].append(result.location_id)
            results["location_name"].append(result.location_name)
            results["location_confidence"].append(result.confidence)
            results["current_shift_name"].append(
                result.current_shift.name if result.current_shift else None
            )
            results["shift_remaining_hours"].append(result.shift_remaining_hours)

        # Add columns to DataFrame
        for col, values in results.items():
            devices_df[col] = values

        return devices_df

    def get_devices_at_location(
        self,
        location_id: str,
        devices_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Filter DataFrame to only devices at a specific location.

        Args:
            location_id: The location ID to filter by
            devices_df: DataFrame with device data

        Returns:
            Filtered DataFrame with only devices at the specified location
        """
        # Ensure location_id column exists
        if "location_id" not in devices_df.columns:
            devices_df = self.bulk_map_devices(devices_df)

        return devices_df[devices_df["location_id"] == location_id]

    def get_location_device_counts(self, devices_df: pd.DataFrame) -> Dict[str, int]:
        """Get count of devices per location.

        Args:
            devices_df: DataFrame with device data

        Returns:
            Dict mapping location_id to device count
        """
        if "location_id" not in devices_df.columns:
            devices_df = self.bulk_map_devices(devices_df)

        counts = devices_df.groupby("location_id").size().to_dict()
        return {k: v for k, v in counts.items() if k is not None}

    def get_current_shift(
        self, location_id: str, timestamp: Optional[datetime] = None
    ) -> Optional[ShiftSchedule]:
        """Get the current shift for a location.

        Args:
            location_id: The location ID
            timestamp: The time to check (defaults to now)

        Returns:
            ShiftSchedule if a shift is active, None otherwise
        """
        config = self.get_location_config(location_id)
        if not config or not config.shifts:
            return None

        check_time = timestamp or datetime.utcnow()
        for shift in config.shifts:
            if shift.covers_time(check_time):
                return shift
        return None

    def get_shift_by_name(
        self, location_id: str, shift_name: str
    ) -> Optional[ShiftSchedule]:
        """Get a specific shift by name for a location.

        Args:
            location_id: The location ID
            shift_name: The shift name (e.g., "Morning", "Night")

        Returns:
            ShiftSchedule if found, None otherwise
        """
        config = self.get_location_config(location_id)
        if not config or not config.shifts:
            return None

        for shift in config.shifts:
            if shift.name.lower() == shift_name.lower():
                return shift
        return None
