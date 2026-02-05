"""
Shared utilities for LLM prompt construction.

Provides metric translations, severity descriptions, and formatting helpers
to ensure consistent, human-readable output across all LLM prompts.
"""

from __future__ import annotations

# =============================================================================
# METRIC TRANSLATIONS
# =============================================================================

METRIC_TRANSLATIONS: dict[str, dict[str, str]] = {
    # Battery metrics
    "TotalBatteryLevelDrop": {
        "name": "Battery Drain",
        "unit": "%",
        "description": "Battery percentage lost during the day",
        "domain": "battery",
    },
    "TotalDischargeTime_Sec": {
        "name": "Active Usage Time",
        "unit": "hours",
        "description": "Time device was unplugged and in use",
        "domain": "battery",
    },
    "TotalBatteryAppDrain": {
        "name": "App Battery Usage",
        "unit": "%",
        "description": "Battery consumed by applications",
        "domain": "battery",
    },
    "CalculatedBatteryCapacity": {
        "name": "Battery Capacity",
        "unit": "mAh",
        "description": "Current battery capacity",
        "domain": "battery",
    },
    "ChargePatternBadCount": {
        "name": "Poor Charge Events",
        "unit": "count",
        "description": "Number of suboptimal charging sessions",
        "domain": "battery",
    },
    "ChargePatternGoodCount": {
        "name": "Good Charge Events",
        "unit": "count",
        "description": "Number of optimal charging sessions",
        "domain": "battery",
    },
    "ScreenOnTime_Sec": {
        "name": "Screen On Time",
        "unit": "hours",
        "description": "Time the screen was active",
        "domain": "battery",
    },
    "BatteryHealth": {
        "name": "Battery Health",
        "unit": "%",
        "description": "Overall battery condition",
        "domain": "battery",
    },
    "BatteryTemperature": {
        "name": "Battery Temperature",
        "unit": "°C",
        "description": "Battery temperature",
        "domain": "battery",
    },
    # App usage metrics
    "AppForegroundTime": {
        "name": "App Screen Time",
        "unit": "hours",
        "description": "Time spent actively using apps",
        "domain": "usage",
    },
    "AppVisitCount": {
        "name": "App Switches",
        "unit": "count",
        "description": "Number of times apps were opened or switched",
        "domain": "usage",
    },
    "TotalForegroundTime": {
        "name": "Total Screen Time",
        "unit": "hours",
        "description": "Total time device screen was active with apps",
        "domain": "usage",
    },
    "UniqueAppsUsed": {
        "name": "Apps Used",
        "unit": "count",
        "description": "Number of different apps used",
        "domain": "usage",
    },
    "CrashCount": {
        "name": "App Crashes",
        "unit": "count",
        "description": "Application crash events",
        "domain": "usage",
    },
    "ANRCount": {
        "name": "App Freezes",
        "unit": "count",
        "description": "App Not Responding events",
        "domain": "usage",
    },
    "LaunchCount": {
        "name": "App Launches",
        "unit": "count",
        "description": "Number of app launches",
        "domain": "usage",
    },
    "SessionCount": {
        "name": "Usage Sessions",
        "unit": "count",
        "description": "Number of device usage sessions",
        "domain": "usage",
    },
    # Data usage metrics
    "Download": {
        "name": "Data Downloaded",
        "unit": "MB",
        "description": "Mobile/WiFi data received",
        "domain": "throughput",
    },
    "Upload": {
        "name": "Data Uploaded",
        "unit": "MB",
        "description": "Mobile/WiFi data sent",
        "domain": "throughput",
    },
    "TotalDownload": {
        "name": "Total Downloaded",
        "unit": "MB",
        "description": "Total data downloaded",
        "domain": "throughput",
    },
    "TotalUpload": {
        "name": "Total Uploaded",
        "unit": "MB",
        "description": "Total data uploaded",
        "domain": "throughput",
    },
    "WifiDownload": {
        "name": "WiFi Downloaded",
        "unit": "MB",
        "description": "Data downloaded over WiFi",
        "domain": "throughput",
    },
    "MobileDownload": {
        "name": "Mobile Downloaded",
        "unit": "MB",
        "description": "Data downloaded over cellular",
        "domain": "throughput",
    },
    # RF/Signal metrics
    "AvgSignalStrength": {
        "name": "Signal Strength",
        "unit": "dBm",
        "description": "Average wireless signal (-50=excellent, -90=poor)",
        "domain": "rf",
    },
    "MinSignalStrength": {
        "name": "Worst Signal",
        "unit": "dBm",
        "description": "Weakest signal recorded",
        "domain": "rf",
    },
    "MaxSignalStrength": {
        "name": "Best Signal",
        "unit": "dBm",
        "description": "Strongest signal recorded",
        "domain": "rf",
    },
    "TotalDropCnt": {
        "name": "Drop Events",
        "unit": "count",
        "description": "Physical impacts detected by sensors",
        "domain": "rf",
    },
    "TotalSignalReadings": {
        "name": "Signal Samples",
        "unit": "count",
        "description": "Number of signal strength measurements",
        "domain": "rf",
    },
    "DropRate": {
        "name": "Drop Rate",
        "unit": "%",
        "description": "Percentage of connection drops",
        "domain": "rf",
    },
    "WifiSignalStrength": {
        "name": "WiFi Signal",
        "unit": "dBm",
        "description": "WiFi signal strength",
        "domain": "rf",
    },
    "CellSignalStrength": {
        "name": "Cellular Signal",
        "unit": "dBm",
        "description": "Cellular signal strength",
        "domain": "rf",
    },
    # Connectivity metrics
    "DisconnectCount": {
        "name": "Connection Drops",
        "unit": "count",
        "description": "Times device lost server connection",
        "domain": "connectivity",
    },
    "OfflineMinutes": {
        "name": "Offline Time",
        "unit": "minutes",
        "description": "Time device was unreachable",
        "domain": "connectivity",
    },
    "DisconnectFlag": {
        "name": "Disconnected",
        "unit": "flag",
        "description": "Whether device experienced disconnection",
        "domain": "connectivity",
    },
    "OnlineTimePct": {
        "name": "Online Percentage",
        "unit": "%",
        "description": "Percentage of time device was online",
        "domain": "connectivity",
    },
    "Rssi": {
        "name": "RSSI",
        "unit": "dBm",
        "description": "Received Signal Strength Indicator",
        "domain": "connectivity",
    },
    "TimeOnWifi": {
        "name": "WiFi Time",
        "unit": "hours",
        "description": "Time connected to WiFi",
        "domain": "connectivity",
    },
    "TimeOn4G": {
        "name": "4G Time",
        "unit": "hours",
        "description": "Time connected to 4G/LTE",
        "domain": "connectivity",
    },
    "TimeOn5G": {
        "name": "5G Time",
        "unit": "hours",
        "description": "Time connected to 5G",
        "domain": "connectivity",
    },
    "TimeOnNoNetwork": {
        "name": "No Network Time",
        "unit": "hours",
        "description": "Time without network connectivity",
        "domain": "connectivity",
    },
    # Storage metrics
    "TotalFreeStorageKb": {
        "name": "Free Storage",
        "unit": "GB",
        "description": "Available device storage",
        "domain": "storage",
    },
    "AvailableStorage": {
        "name": "Available Storage",
        "unit": "GB",
        "description": "Available storage space",
        "domain": "storage",
    },
    "TotalStorage": {
        "name": "Total Storage",
        "unit": "GB",
        "description": "Total storage capacity",
        "domain": "storage",
    },
    "AvailableRAM": {
        "name": "Available Memory",
        "unit": "GB",
        "description": "Available RAM",
        "domain": "storage",
    },
    "TotalRAM": {
        "name": "Total Memory",
        "unit": "GB",
        "description": "Total RAM",
        "domain": "storage",
    },
    # Derived/composite metrics
    "StorageUtilization": {
        "name": "Storage Used",
        "unit": "%",
        "description": "Percentage of storage in use",
        "domain": "storage",
    },
    "RAMPressure": {
        "name": "Memory Pressure",
        "unit": "%",
        "description": "Percentage of RAM in use",
        "domain": "storage",
    },
    "BatteryDrainPerHour": {
        "name": "Battery Drain Rate",
        "unit": "%/hour",
        "description": "Battery consumption rate per hour",
        "domain": "battery",
    },
    "ConnectionStabilityScore": {
        "name": "Connection Stability",
        "unit": "score",
        "description": "Overall connection reliability (0-1, higher=better)",
        "domain": "connectivity",
    },
}


# =============================================================================
# SERVICE DESCRIPTIONS (for troubleshooting)
# =============================================================================

SERVICE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "XSight Database SQL Server": {
        "purpose": "Data warehouse storing device telemetry (battery, connectivity, app usage)",
        "common_issues": "Connection timeout usually means network/firewall issues",
        "config_location": "DW_SQL_* environment variables",
    },
    "MobiControl SQL Server": {
        "purpose": "Core MDM database with device inventory, policies, compliance status",
        "common_issues": "'Host not configured' means missing database_host in config",
        "config_location": "MC_SQL_* environment variables",
    },
    "MobiControl API": {
        "purpose": "REST API for device management actions",
        "common_issues": "Requires ENABLE_MOBICONTROL=true environment variable",
        "config_location": "MOBICONTROL_* environment variables",
    },
    "LLM Service": {
        "purpose": "AI service for generating insights and explanations",
        "common_issues": "Local service (LM Studio) must be running with a model loaded",
        "config_location": "LLM_BASE_URL and LLM_MODEL_NAME environment variables",
    },
    "Redis": {
        "purpose": "Cache layer for performance optimization",
        "common_issues": "Usually auto-recovers; check if service is running",
        "config_location": "REDIS_URL environment variable",
    },
    "Qdrant": {
        "purpose": "Vector database for semantic search",
        "common_issues": "Check if Docker container is running",
        "config_location": "QDRANT_* environment variables",
    },
    "Backend Database": {
        "purpose": "PostgreSQL database for application state",
        "common_issues": "Check DATABASE_URL connection string",
        "config_location": "DATABASE_URL environment variable",
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def translate_metric(technical_name: str) -> str:
    """Convert technical metric name to human-readable name."""
    info = METRIC_TRANSLATIONS.get(technical_name, {})
    return info.get("name", technical_name)


def get_metric_info(technical_name: str) -> tuple[str, str, str]:
    """Get human name, unit, and description for a metric.

    Returns:
        Tuple of (human_name, unit, description)
    """
    info = METRIC_TRANSLATIONS.get(technical_name, {})
    return (
        info.get("name", technical_name),
        info.get("unit", ""),
        info.get("description", ""),
    )


def get_metric_domain(technical_name: str) -> str:
    """Get the domain category for a metric."""
    info = METRIC_TRANSLATIONS.get(technical_name, {})
    return info.get("domain", "unknown")


def format_metric_value(name: str, value: float) -> str:
    """Format metric value with appropriate unit and conversion."""
    info = METRIC_TRANSLATIONS.get(name, {})
    unit = info.get("unit", "")

    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    # Convert seconds to hours
    if unit == "hours" and ("Sec" in name or "Time" in name and "Sec" not in name):
        if "Sec" in name:
            return f"{value / 3600:.1f} hours"
        return f"{value:.1f} hours"

    # Convert KB to GB
    if unit == "GB" and "Kb" in name:
        return f"{value / 1e6:.2f} GB"

    # Convert bytes to MB
    if unit == "MB" and value > 1e6:
        return f"{value / 1e6:.1f} MB"
    elif unit == "MB":
        return f"{value:.1f} MB"

    # Format percentages
    if unit == "%":
        return f"{value:.1f}%"

    # Format counts
    if unit == "count":
        return f"{int(value)}"

    # Format dBm (signal strength)
    if unit == "dBm":
        return f"{value:.0f} dBm"

    # Default formatting
    if isinstance(value, float) and value == int(value):
        return f"{int(value)} {unit}".strip()
    return f"{value:.2f} {unit}".strip()


def get_severity_word(anomaly_score: float) -> str:
    """Convert anomaly score to human-readable severity level.

    Isolation Forest scores range from -1 (most anomalous) to 0 (normal).
    """
    if anomaly_score < -0.7:
        return "severe"
    elif anomaly_score < -0.5:
        return "moderate"
    elif anomaly_score < -0.3:
        return "mild"
    return "minimal"


def get_severity_emoji(anomaly_score: float) -> str:
    """Get emoji indicator for severity level."""
    if anomaly_score < -0.7:
        return "🔴"
    elif anomaly_score < -0.5:
        return "🟠"
    elif anomaly_score < -0.3:
        return "🟡"
    return "🟢"


def get_z_score_description(z: float) -> str:
    """Describe how unusual a z-score is in plain language."""
    abs_z = abs(z)
    direction = "higher" if z > 0 else "lower"

    if abs_z > 3:
        return f"significantly {direction} than normal"
    elif abs_z > 2:
        return f"notably {direction} than normal"
    elif abs_z > 1:
        return f"somewhat {direction} than normal"
    return "within normal range"


def get_z_score_severity(z: float) -> str:
    """Get severity classification for a z-score."""
    abs_z = abs(z)
    if abs_z > 3:
        return "significant"
    elif abs_z > 2:
        return "notable"
    elif abs_z > 1:
        return "minor"
    return "normal"


def get_health_status(anomaly_rate: float) -> tuple[str, str]:
    """Get health status and emoji based on anomaly rate.

    Returns:
        Tuple of (status_text, emoji)
    """
    if anomaly_rate < 0.05:
        return "Healthy", "✅"
    elif anomaly_rate < 0.15:
        return "Needs Attention", "⚠️"
    elif anomaly_rate < 0.30:
        return "Problematic", "🟠"
    return "Critical", "🔴"


def get_duration_interpretation(days: int) -> str:
    """Get interpretation guidance based on anomaly event duration."""
    if days <= 1:
        return "Single-day anomaly - could be temporary or random variation"
    elif days <= 3:
        return "Short event (2-3 days) - may be temporary (software update, unusual work period)"
    elif days <= 7:
        return "Extended event (4-7 days) - likely a real issue needing attention"
    return "Prolonged event (7+ days) - definite problem requiring intervention"


def build_metric_summary(metrics: list, top_n: int = 5, include_description: bool = True) -> str:
    """Build a human-readable summary of metrics sorted by z-score.

    Args:
        metrics: List of metric dicts with 'name', 'value', 'z_score' keys
        top_n: Number of top metrics to include
        include_description: Whether to include metric descriptions

    Returns:
        Formatted string summarizing the most unusual metrics
    """
    # Sort by absolute z-score
    sorted_metrics = sorted(metrics, key=lambda x: abs(x.get("z_score", 0)), reverse=True)[:top_n]

    lines = []
    for m in sorted_metrics:
        name = m.get("name", "")
        z = m.get("z_score", 0)
        value = m.get("value", 0)

        human_name, unit, desc = get_metric_info(name)
        formatted_value = format_metric_value(name, value)
        severity = get_z_score_description(z)

        if include_description and desc:
            lines.append(f"- **{human_name}**: {formatted_value} ({severity}) - {desc}")
        else:
            lines.append(f"- **{human_name}**: {formatted_value} ({severity})")

    return "\n".join(lines)


def get_common_root_causes() -> str:
    """Get reference text for common anomaly root causes."""
    return """Common anomaly patterns to consider:
- **Battery drain spike**: High battery drop + high app drain → runaway app or failing battery
- **Connectivity issues**: High disconnects + high offline time → poor signal area or network problems
- **Device abuse**: High drop count → physical mishandling, may need inspection
- **Storage pressure**: Low free storage → apps can't update, performance degrades
- **App instability**: High crashes + high ANR count → problematic app version or OS issue
- **Unusual usage**: Extreme app time or app switches → heavy work day or unauthorized use
- **Network degradation**: Poor signal strength + high drop rate → hardware issue or location problem"""


# =============================================================================
# SYSTEM PROMPT COMPONENTS
# =============================================================================

NO_THINKING_INSTRUCTION = """IMPORTANT: Output ONLY your final response. Do NOT include any internal reasoning, <think> tags, chain-of-thought, or preamble. Start directly with the requested output format."""

DOMAIN_CONTEXT = """This is an enterprise mobile device management (MDM) system monitoring Android/iOS/Windows devices used in warehouses, retail stores, and field operations. The system uses SOTI MobiControl for device management and an Isolation Forest ML model for anomaly detection."""
