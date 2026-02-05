"""Insight categories aligned with Carl's (CEO) requirements.

These categories translate technical anomaly detection into
customer-understandable insight types.

Carl's key feedback: "I did not feel he was talking about things that
would make sense customers could understand or would get their interest."

Categories are organized by business domain:
- Battery: Shift failures, drain patterns, charging issues
- Device: Drops, reboots, abuse patterns
- Network: AP hopping, carrier issues, disconnects
- Apps: Crashes, power consumption
- Cohort: Performance by manufacturer/model/OS
- Location: Cross-location comparisons
"""

from __future__ import annotations

from enum import StrEnum


class InsightCategory(StrEnum):
    """Customer-facing insight categories.

    Each category maps to specific anomaly patterns and has associated
    business language templates for customer communication.
    """

    # =========================================================================
    # BATTERY INSIGHTS
    # Carl: "Batteries that don't last a shift"
    # Carl: "Consistent or periodic rapid drain issues"
    # Carl: "Batteries not fully charged in the morning"
    # Carl: "Charging pattern correlations"
    # =========================================================================

    BATTERY_SHIFT_FAILURE = "battery_shift_failure"
    """Device battery won't last a full work shift."""

    BATTERY_RAPID_DRAIN = "battery_rapid_drain"
    """Battery draining faster than normal/expected."""

    BATTERY_CHARGE_INCOMPLETE = "battery_charge_incomplete"
    """Device wasn't fully charged at shift start."""

    BATTERY_CHARGE_PATTERN = "battery_charge_pattern"
    """Abnormal charging pattern detected (short/interrupted charges)."""

    BATTERY_PERIODIC_DRAIN = "battery_periodic_drain"
    """Consistent/periodic rapid drain pattern (same time daily, weekly)."""

    BATTERY_HEALTH_DEGRADED = "battery_health_degraded"
    """Battery health has degraded below acceptable threshold."""

    # =========================================================================
    # DEVICE PERFORMANCE INSIGHTS
    # Carl: "Devices/people/locations with excessive drops"
    # Carl: "Devices/people/locations with excessive reboots"
    # =========================================================================

    EXCESSIVE_DROPS = "excessive_drops"
    """Device experiencing excessive physical drops."""

    EXCESSIVE_REBOOTS = "excessive_reboots"
    """Device experiencing excessive reboots/restarts."""

    DEVICE_ABUSE_PATTERN = "device_abuse_pattern"
    """Combined pattern of drops/reboots indicating device abuse."""

    DEVICE_PERFORMANCE_DEGRADED = "device_performance_degraded"
    """Device showing degraded performance metrics."""

    # =========================================================================
    # NETWORK INSIGHTS
    # Carl: "AP hopping/stickiness"
    # Carl: "Tower hopping/stickiness"
    # Carl: "Patterns by carrier, network type (5G, 4G)"
    # Carl: "Throughput metrics (upload, download, ping)"
    # Carl: "Server disconnection patterns"
    # =========================================================================

    WIFI_AP_HOPPING = "wifi_ap_hopping"
    """Device excessively switching between WiFi access points."""

    WIFI_STICKINESS = "wifi_stickiness"
    """Device not roaming when it should (stuck on weak AP)."""

    WIFI_DEAD_ZONE = "wifi_dead_zone"
    """Location with consistently poor WiFi coverage."""

    CELLULAR_TOWER_HOPPING = "cellular_tower_hopping"
    """Device excessively switching between cell towers."""

    CELLULAR_CARRIER_ISSUE = "cellular_carrier_issue"
    """Carrier-specific connectivity or performance issue."""

    CELLULAR_TECH_DEGRADATION = "cellular_tech_degradation"
    """Device frequently falling from 5G to 4G to 3G."""

    NETWORK_DISCONNECT_PATTERN = "network_disconnect_pattern"
    """Predictable/periodic network disconnection pattern."""

    NETWORK_THROUGHPUT_ISSUE = "network_throughput_issue"
    """Upload/download speeds significantly below expected."""

    DEVICE_HIDDEN_PATTERN = "device_hidden_pattern"
    """Device showing suspicious offline patterns (taken home, hidden)."""

    # =========================================================================
    # APP INSIGHTS
    # Carl: "Crashes"
    # Carl: "Apps consuming too much power (foreground time vs drain)"
    # =========================================================================

    APP_CRASH_PATTERN = "app_crash_pattern"
    """App experiencing repeated crashes."""

    APP_POWER_DRAIN = "app_power_drain"
    """App consuming disproportionate battery vs usage time."""

    APP_ANR_PATTERN = "app_anr_pattern"
    """App experiencing repeated ANR (App Not Responding) events."""

    APP_PERFORMANCE_ISSUE = "app_performance_issue"
    """App showing performance degradation (slow launch, etc.)."""

    # =========================================================================
    # COHORT/DEVICE TYPE INSIGHTS
    # Carl: "Performance patterns by manufacturer, model, OS version, firmware"
    # Carl: "Combinations that cause problems"
    # =========================================================================

    COHORT_PERFORMANCE_ISSUE = "cohort_performance_issue"
    """Device type/cohort performing worse than others."""

    FIRMWARE_BUG = "firmware_bug"
    """Firmware version causing device issues."""

    OS_VERSION_ISSUE = "os_version_issue"
    """OS version associated with problems."""

    PROBLEM_COMBINATION = "problem_combination"
    """Specific manufacturer+model+OS+firmware combination with issues."""

    # =========================================================================
    # LOCATION INSIGHTS
    # Carl: "Relate any anomalies to location (warehouse 1 vs warehouse 2)"
    # Carl: "Identify negative patterns combining multiple factors"
    # =========================================================================

    LOCATION_ANOMALY_CLUSTER = "location_anomaly_cluster"
    """Multiple devices at a location showing similar issues."""

    LOCATION_PERFORMANCE_GAP = "location_performance_gap"
    """Location performing significantly worse than comparable locations."""

    LOCATION_BASELINE_DEVIATION = "location_baseline_deviation"
    """Location metrics deviating from its historical baseline."""

    LOCATION_INFRASTRUCTURE_ISSUE = "location_infrastructure_issue"
    """Location has infrastructure problems (charging, network, etc.)."""


class InsightSeverity(StrEnum):
    """Severity level for insights."""

    CRITICAL = "critical"
    """Immediate action required. Major business impact."""

    HIGH = "high"
    """Action needed soon. Significant impact if unaddressed."""

    MEDIUM = "medium"
    """Should address when possible. Moderate impact."""

    LOW = "low"
    """Good to know. Minor impact."""

    INFO = "info"
    """Informational. No action required."""


class EntityType(StrEnum):
    """Entity types for insight aggregation."""

    DEVICE = "device"
    """Individual device."""

    USER = "user"
    """User/person assigned to devices."""

    LOCATION = "location"
    """Physical location (warehouse, store, etc.)."""

    REGION = "region"
    """Region grouping multiple locations."""

    COHORT = "cohort"
    """Device cohort (manufacturer+model+OS+firmware)."""

    MANUFACTURER = "manufacturer"
    """Device manufacturer."""

    MODEL = "model"
    """Device model."""

    APP = "app"
    """Application."""

    CARRIER = "carrier"
    """Network carrier."""


# Category metadata for UI and processing
CATEGORY_METADATA = {
    # Battery
    InsightCategory.BATTERY_SHIFT_FAILURE: {
        "domain": "battery",
        "default_severity": InsightSeverity.CRITICAL,
        "affects_productivity": True,
        "primary_metrics": ["BatteryDrainPerHour", "BatteryLevel", "ShiftDurationHours"],
        "icon": "battery-warning",
    },
    InsightCategory.BATTERY_RAPID_DRAIN: {
        "domain": "battery",
        "default_severity": InsightSeverity.HIGH,
        "affects_productivity": True,
        "primary_metrics": ["BatteryDrainPerHour", "TotalBatteryLevelDrop"],
        "icon": "battery-low",
    },
    InsightCategory.BATTERY_CHARGE_INCOMPLETE: {
        "domain": "battery",
        "default_severity": InsightSeverity.MEDIUM,
        "affects_productivity": True,
        "primary_metrics": ["BatteryLevel", "ChargePatternGoodCount"],
        "icon": "battery-charging",
    },
    InsightCategory.BATTERY_CHARGE_PATTERN: {
        "domain": "battery",
        "default_severity": InsightSeverity.MEDIUM,
        "affects_productivity": False,
        "primary_metrics": ["ChargePatternBadCount", "AcChargeCount", "UsbChargeCount"],
        "icon": "battery-bolt",
    },
    InsightCategory.BATTERY_PERIODIC_DRAIN: {
        "domain": "battery",
        "default_severity": InsightSeverity.MEDIUM,
        "affects_productivity": True,
        "primary_metrics": ["BatteryDrainPerHour", "TimeOfDay", "DayOfWeek"],
        "icon": "clock-alert",
    },
    InsightCategory.BATTERY_HEALTH_DEGRADED: {
        "domain": "battery",
        "default_severity": InsightSeverity.HIGH,
        "affects_productivity": True,
        "primary_metrics": ["BatteryHealth", "CycleCount", "FullChargeCapacity"],
        "icon": "battery-off",
    },

    # Device
    InsightCategory.EXCESSIVE_DROPS: {
        "domain": "device",
        "default_severity": InsightSeverity.HIGH,
        "affects_productivity": True,
        "primary_metrics": ["TotalDropCnt", "DropRate"],
        "icon": "phone-down",
    },
    InsightCategory.EXCESSIVE_REBOOTS: {
        "domain": "device",
        "default_severity": InsightSeverity.HIGH,
        "affects_productivity": True,
        "primary_metrics": ["RebootCount", "CrashCount"],
        "icon": "refresh-alert",
    },
    InsightCategory.DEVICE_ABUSE_PATTERN: {
        "domain": "device",
        "default_severity": InsightSeverity.HIGH,
        "affects_productivity": True,
        "primary_metrics": ["TotalDropCnt", "RebootCount"],
        "icon": "alert-triangle",
    },
    InsightCategory.DEVICE_PERFORMANCE_DEGRADED: {
        "domain": "device",
        "default_severity": InsightSeverity.MEDIUM,
        "affects_productivity": True,
        "primary_metrics": ["CpuActiveTime", "AvailableRAM", "AvailableStorage"],
        "icon": "gauge",
    },

    # Network
    InsightCategory.WIFI_AP_HOPPING: {
        "domain": "network",
        "default_severity": InsightSeverity.MEDIUM,
        "affects_productivity": True,
        "primary_metrics": ["UniqueAPsConnected", "WifiDisconnectCount"],
        "icon": "wifi-off",
    },
    InsightCategory.WIFI_STICKINESS: {
        "domain": "network",
        "default_severity": InsightSeverity.LOW,
        "affects_productivity": True,
        "primary_metrics": ["WifiSignalStrength", "UniqueAPsConnected"],
        "icon": "wifi",
    },
    InsightCategory.WIFI_DEAD_ZONE: {
        "domain": "network",
        "default_severity": InsightSeverity.HIGH,
        "affects_productivity": True,
        "primary_metrics": ["WifiSignalStrength", "WifiDropCount"],
        "icon": "wifi-alert",
    },
    InsightCategory.CELLULAR_TOWER_HOPPING: {
        "domain": "network",
        "default_severity": InsightSeverity.MEDIUM,
        "affects_productivity": True,
        "primary_metrics": ["CellTowerChanges", "UniqueCellIds"],
        "icon": "signal",
    },
    InsightCategory.CELLULAR_CARRIER_ISSUE: {
        "domain": "network",
        "default_severity": InsightSeverity.MEDIUM,
        "affects_productivity": True,
        "primary_metrics": ["CellSignalStrength", "CellDropCount"],
        "icon": "signal-off",
    },
    InsightCategory.CELLULAR_TECH_DEGRADATION: {
        "domain": "network",
        "default_severity": InsightSeverity.LOW,
        "affects_productivity": False,
        "primary_metrics": ["TimeOn5G", "TimeOn4G", "TimeOn3G", "NetworkTypeCount"],
        "icon": "signal-low",
    },
    InsightCategory.NETWORK_DISCONNECT_PATTERN: {
        "domain": "network",
        "default_severity": InsightSeverity.HIGH,
        "affects_productivity": True,
        "primary_metrics": ["DisconnectCount", "TimeOnNoNetwork"],
        "icon": "unlink",
    },
    InsightCategory.NETWORK_THROUGHPUT_ISSUE: {
        "domain": "network",
        "default_severity": InsightSeverity.MEDIUM,
        "affects_productivity": True,
        "primary_metrics": ["TotalDownload", "TotalUpload"],
        "icon": "download-off",
    },
    InsightCategory.DEVICE_HIDDEN_PATTERN: {
        "domain": "network",
        "default_severity": InsightSeverity.HIGH,
        "affects_productivity": False,
        "primary_metrics": ["TimeOnNoNetwork", "LastCheckInTime"],
        "icon": "eye-off",
    },

    # Apps
    InsightCategory.APP_CRASH_PATTERN: {
        "domain": "apps",
        "default_severity": InsightSeverity.HIGH,
        "affects_productivity": True,
        "primary_metrics": ["CrashCount", "ANRCount"],
        "icon": "app-window-error",
    },
    InsightCategory.APP_POWER_DRAIN: {
        "domain": "apps",
        "default_severity": InsightSeverity.MEDIUM,
        "affects_productivity": True,
        "primary_metrics": ["TotalBatteryAppDrain", "AppForegroundTime"],
        "icon": "battery-minus",
    },
    InsightCategory.APP_ANR_PATTERN: {
        "domain": "apps",
        "default_severity": InsightSeverity.MEDIUM,
        "affects_productivity": True,
        "primary_metrics": ["ANRCount"],
        "icon": "clock-pause",
    },
    InsightCategory.APP_PERFORMANCE_ISSUE: {
        "domain": "apps",
        "default_severity": InsightSeverity.LOW,
        "affects_productivity": True,
        "primary_metrics": ["LaunchCount", "AverageSessionDuration"],
        "icon": "gauge-low",
    },

    # Cohort
    InsightCategory.COHORT_PERFORMANCE_ISSUE: {
        "domain": "cohort",
        "default_severity": InsightSeverity.MEDIUM,
        "affects_productivity": True,
        "primary_metrics": ["AnomalyRate", "PerformanceScore"],
        "icon": "devices",
    },
    InsightCategory.FIRMWARE_BUG: {
        "domain": "cohort",
        "default_severity": InsightSeverity.HIGH,
        "affects_productivity": True,
        "primary_metrics": ["FirmwareVersion", "CrashCount"],
        "icon": "bug",
    },
    InsightCategory.OS_VERSION_ISSUE: {
        "domain": "cohort",
        "default_severity": InsightSeverity.HIGH,
        "affects_productivity": True,
        "primary_metrics": ["OsVersion", "CrashCount"],
        "icon": "code",
    },
    InsightCategory.PROBLEM_COMBINATION: {
        "domain": "cohort",
        "default_severity": InsightSeverity.HIGH,
        "affects_productivity": True,
        "primary_metrics": ["Manufacturer", "Model", "OsVersion", "FirmwareVersion"],
        "icon": "puzzle",
    },

    # Location
    InsightCategory.LOCATION_ANOMALY_CLUSTER: {
        "domain": "location",
        "default_severity": InsightSeverity.HIGH,
        "affects_productivity": True,
        "primary_metrics": ["AnomalyCount", "DeviceCount"],
        "icon": "map-pin-alert",
    },
    InsightCategory.LOCATION_PERFORMANCE_GAP: {
        "domain": "location",
        "default_severity": InsightSeverity.MEDIUM,
        "affects_productivity": True,
        "primary_metrics": ["PerformanceScore", "PeerComparison"],
        "icon": "chart-gap",
    },
    InsightCategory.LOCATION_BASELINE_DEVIATION: {
        "domain": "location",
        "default_severity": InsightSeverity.MEDIUM,
        "affects_productivity": True,
        "primary_metrics": ["CurrentValue", "BaselineValue", "DeviationPercent"],
        "icon": "trending-down",
    },
    InsightCategory.LOCATION_INFRASTRUCTURE_ISSUE: {
        "domain": "location",
        "default_severity": InsightSeverity.HIGH,
        "affects_productivity": True,
        "primary_metrics": ["ChargingIssues", "NetworkIssues"],
        "icon": "building-alert",
    },
}


def get_categories_by_domain(domain: str) -> list[InsightCategory]:
    """Get all insight categories for a given domain."""
    return [
        cat for cat, meta in CATEGORY_METADATA.items()
        if meta.get("domain") == domain
    ]


def get_category_severity(category: InsightCategory) -> InsightSeverity:
    """Get the default severity for an insight category."""
    meta = CATEGORY_METADATA.get(category, {})
    return meta.get("default_severity", InsightSeverity.MEDIUM)


def get_category_metrics(category: InsightCategory) -> list[str]:
    """Get the primary metrics for an insight category."""
    meta = CATEGORY_METADATA.get(category, {})
    return meta.get("primary_metrics", [])


def affects_productivity(category: InsightCategory) -> bool:
    """Check if an insight category affects worker productivity."""
    meta = CATEGORY_METADATA.get(category, {})
    return meta.get("affects_productivity", True)
