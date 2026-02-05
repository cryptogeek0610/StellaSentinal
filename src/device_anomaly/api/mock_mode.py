"""Mock Mode Data Provider for SOTI Stella Sentinel.

This module provides comprehensive mock data generation for demo/testing purposes.
All mock data is consistent and internally coherent (device IDs match across endpoints, etc.).
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

# Seed for reproducibility
MOCK_SEED = 42
_rng = random.Random(MOCK_SEED)

# ============================================================================
# Mock Data Constants
# ============================================================================

MOCK_STORES = [
    {"id": "store-001", "name": "Downtown Flagship", "region": "Northeast"},
    {"id": "store-002", "name": "Westside Mall", "region": "West"},
    {"id": "store-003", "name": "Harbor Point", "region": "Southeast"},
    {"id": "store-004", "name": "Tech Plaza", "region": "West"},
    {"id": "store-005", "name": "Central Station", "region": "Midwest"},
    {"id": "store-006", "name": "Riverside Center", "region": "Northeast"},
    {"id": "store-007", "name": "Airport Terminal", "region": "Southeast"},
    {"id": "store-008", "name": "University District", "region": "Midwest"},
]

MOCK_DEVICE_MODELS = [
    "Samsung Galaxy Tab A8",
    "iPad Pro 11",
    "Zebra TC52",
    "Honeywell CT60",
    "Samsung Galaxy XCover 6",
    "Panasonic Toughbook N1",
]

MOCK_ANOMALY_TYPES = [
    "battery_drain",
    "storage_critical",
    "network_instability",
    "offline_extended",
    "data_spike",
]


# ============================================================================
# Mock Device Generation
# ============================================================================


def _generate_mock_devices(count: int = 50) -> list[dict[str, Any]]:
    """Generate a consistent set of mock devices."""
    devices = []
    for i in range(1, count + 1):
        store = MOCK_STORES[i % len(MOCK_STORES)]
        model = MOCK_DEVICE_MODELS[i % len(MOCK_DEVICE_MODELS)]

        # Determine status with realistic distribution
        status_roll = _rng.random()
        if status_roll < 0.75:
            status = "Active"
        elif status_roll < 0.90:
            status = "Idle"
        elif status_roll < 0.95:
            status = "Offline"
        else:
            status = "Charging"

        devices.append(
            {
                "device_id": i,
                "device_name": f"Device-{i:04d}",
                "device_model": model,
                "location": store["name"],
                "region": store["region"],
                "store_id": store["id"],
                "status": status,
                "battery": _rng.randint(15, 100),
                "is_charging": status == "Charging",
                "last_seen": (
                    datetime.now() - timedelta(minutes=_rng.randint(0, 1440))
                ).isoformat(),
                "os_version": f"Android {_rng.choice(['12', '13', '14'])}",
                "agent_version": f"15.{_rng.randint(1, 5)}.{_rng.randint(0, 9)}",
            }
        )

    return devices


# Cached mock devices for consistency
_MOCK_DEVICES = _generate_mock_devices(50)


def get_mock_device(device_id: int) -> dict[str, Any] | None:
    """Get a specific mock device by ID."""
    for device in _MOCK_DEVICES:
        if device["device_id"] == device_id:
            return device
    return None


def get_mock_devices() -> list[dict[str, Any]]:
    """Get all mock devices."""
    return _MOCK_DEVICES.copy()


# ============================================================================
# Mock Anomaly Generation
# ============================================================================


def _generate_mock_anomalies(count: int = 100) -> list[dict[str, Any]]:
    """Generate a consistent set of mock anomalies."""
    anomalies = []
    base_time = datetime.now()

    for i in range(1, count + 1):
        device = _rng.choice(_MOCK_DEVICES)
        anomaly_type = _rng.choice(MOCK_ANOMALY_TYPES)

        # Status distribution
        status_roll = _rng.random()
        if status_roll < 0.4:
            status = "new"
        elif status_roll < 0.7:
            status = "investigating"
        elif status_roll < 0.9:
            status = "resolved"
        else:
            status = "dismissed"

        # Generate realistic score
        score = round(_rng.uniform(0.65, 0.99), 3)

        # Time offset (spread over past 14 days)
        time_offset = timedelta(
            days=_rng.randint(0, 13), hours=_rng.randint(0, 23), minutes=_rng.randint(0, 59)
        )
        timestamp = base_time - time_offset

        anomaly = {
            "id": i,
            "device_id": device["device_id"],
            "timestamp": timestamp.isoformat(),
            "anomaly_score": score,
            "anomaly_label": -1,
            "status": status,
            "assigned_to": _rng.choice([None, "Admin", "Operator", "System"]),
            "total_battery_level_drop": _rng.randint(10, 80)
            if anomaly_type == "battery_drain"
            else _rng.randint(5, 20),
            "total_free_storage_kb": _rng.randint(50000, 500000)
            if anomaly_type == "storage_critical"
            else _rng.randint(1000000, 5000000),
            "download": _rng.randint(100000, 5000000)
            if anomaly_type == "data_spike"
            else _rng.randint(10000, 500000),
            "upload": _rng.randint(50000, 2000000)
            if anomaly_type == "data_spike"
            else _rng.randint(5000, 100000),
            "offline_time": _rng.randint(60, 480)
            if anomaly_type == "offline_extended"
            else _rng.randint(0, 30),
            "disconnect_count": _rng.randint(5, 25)
            if anomaly_type == "network_instability"
            else _rng.randint(0, 5),
            "wifi_signal_strength": _rng.randint(10, 40)
            if anomaly_type == "network_instability"
            else _rng.randint(50, 90),
            "connection_time": _rng.randint(30, 120),
            "feature_values_json": None,
            "created_at": timestamp.isoformat(),
            "updated_at": timestamp.isoformat(),
        }
        anomalies.append(anomaly)

    # Sort by timestamp descending
    anomalies.sort(key=lambda x: x["timestamp"], reverse=True)
    return anomalies


_MOCK_ANOMALIES = _generate_mock_anomalies(100)


# ============================================================================
# Public Mock Data Functions
# ============================================================================


def get_mock_dashboard_stats() -> dict[str, Any]:
    """Get mock dashboard KPI statistics."""
    today = datetime.now().date()
    today_anomalies = [
        a for a in _MOCK_ANOMALIES if datetime.fromisoformat(a["timestamp"]).date() == today
    ]

    # Count open cases
    open_cases = len([a for a in _MOCK_ANOMALIES if a.get("status") in ("open", "investigating")])

    return {
        "anomalies_today": len(today_anomalies) or _rng.randint(3, 12),
        "devices_monitored": len(_MOCK_DEVICES),
        "critical_issues": _rng.randint(1, 5),
        "resolved_today": _rng.randint(2, 8),
        "open_cases": open_cases or _rng.randint(5, 15),
    }


def get_mock_dashboard_trends(
    days: int = 7, start_date: datetime | None = None, end_date: datetime | None = None
) -> list[dict[str, Any]]:
    """Get mock anomaly trend data."""
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=days)

    trends = []
    current = start_date.date()
    end = end_date.date()

    while current <= end:
        # Generate realistic daily counts with some pattern
        base_count = 8
        day_of_week = current.weekday()

        # Higher on weekdays
        if day_of_week < 5:
            count = base_count + _rng.randint(2, 8)
        else:
            count = base_count + _rng.randint(-2, 3)

        trends.append(
            {
                "date": current.isoformat(),
                "anomaly_count": max(0, count),
            }
        )
        current += timedelta(days=1)

    return trends


def get_mock_connection_status() -> dict[str, Any]:
    """Get mock connection status (all connected)."""
    return {
        "backend_db": {
            "connected": True,
            "server": "postgres:5432",
            "error": None,
            "status": "connected",
        },
        "dw_sql": {
            "connected": True,
            "server": "mock-dw-server.local",
            "error": None,
            "status": "connected",
        },
        "mc_sql": {
            "connected": True,
            "server": "mock-mc-server.local",
            "error": None,
            "status": "connected",
        },
        "mobicontrol_api": {
            "connected": True,
            "server": "https://mock.mobicontrol.net/MobiControl",
            "error": None,
            "status": "connected",
        },
        "llm": {
            "connected": True,
            "server": "http://localhost:11434",
            "error": None,
            "status": "connected",
        },
        "redis": {
            "connected": True,
            "server": "redis://redis:6379",
            "error": None,
            "status": "connected",
        },
        "qdrant": {
            "connected": True,
            "server": "qdrant:6333",
            "error": None,
            "status": "connected",
        },
        "last_checked": datetime.now().isoformat(),
    }


def get_mock_anomalies(
    device_id: int | None = None,
    status: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> dict[str, Any]:
    """Get paginated mock anomalies."""
    filtered = _MOCK_ANOMALIES.copy()

    if device_id:
        filtered = [a for a in filtered if a["device_id"] == device_id]

    if status:
        filtered = [a for a in filtered if a["status"] == status]

    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size

    return {
        "anomalies": filtered[start:end],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


def get_mock_anomaly_detail(anomaly_id: int) -> dict[str, Any] | None:
    """Get mock anomaly detail with notes."""
    for anomaly in _MOCK_ANOMALIES:
        if anomaly["id"] == anomaly_id:
            detail = anomaly.copy()
            detail["notes"] = "Mock anomaly for demonstration purposes."
            detail["investigation_notes"] = [
                {
                    "id": 1,
                    "user": "System",
                    "note": "Anomaly detected by Isolation Forest model with high confidence.",
                    "action_type": "detection",
                    "created_at": detail["created_at"],
                },
                {
                    "id": 2,
                    "user": "Admin",
                    "note": "Reviewing device telemetry patterns.",
                    "action_type": "investigation",
                    "created_at": datetime.now().isoformat(),
                },
            ]
            return detail
    return None


def get_mock_device_detail(device_id: int) -> dict[str, Any] | None:
    """Get mock device detail with recent anomalies."""
    device = get_mock_device(device_id)
    if not device:
        return None

    device_anomalies = [a for a in _MOCK_ANOMALIES if a["device_id"] == device_id]

    return {
        **device,
        "anomaly_count": len(device_anomalies),
        "recent_anomalies": device_anomalies[:5],
    }


def get_mock_isolation_forest_stats(days: int = 30) -> dict[str, Any]:
    """Get mock Isolation Forest model statistics."""
    # Generate score distribution bins
    bins = []
    total_normal = 0
    total_anomalies = 0

    for i in range(10):
        bin_start = i * 0.1
        bin_end = (i + 1) * 0.1

        if bin_end <= 0.5:
            count = _rng.randint(100, 300)
            is_anomaly = False
            total_normal += count
        elif bin_end <= 0.7:
            count = _rng.randint(50, 150)
            is_anomaly = False
            total_normal += count
        else:
            count = _rng.randint(5, 30)
            is_anomaly = True
            total_anomalies += count

        bins.append(
            {
                "bin_start": round(bin_start, 1),
                "bin_end": round(bin_end, 1),
                "count": count,
                "is_anomaly": is_anomaly,
            }
        )

    total = total_normal + total_anomalies

    return {
        "config": {
            "n_estimators": 100,
            "contamination": 0.05,
            "random_state": 42,
            "scale_features": True,
            "min_variance": 0.01,
            "feature_count": 8,
            "model_type": "IsolationForest",
        },
        "score_distribution": {
            "bins": bins,
            "total_normal": total_normal,
            "total_anomalies": total_anomalies,
            "mean_score": 0.42,
            "median_score": 0.38,
            "min_score": 0.05,
            "max_score": 0.98,
            "std_score": 0.18,
        },
        "total_predictions": total,
        "anomaly_rate": round(total_anomalies / total, 4) if total > 0 else 0,
    }


def get_mock_baseline_suggestions(source: str | None = None) -> list[dict[str, Any]]:
    """Get mock baseline adjustment suggestions."""
    features = [
        ("TotalBatteryLevelDrop", "Global", "global", 15, 18, 17),
        ("OfflineTime", "Global", "global", 10, 14, 12),
        ("Download", "Store: Downtown Flagship", "store-001", 150000, 220000, 180000),
        ("WiFiSignalStrength", "Region: Northeast", "northeast", 65, 58, 60),
    ]

    suggestions = []
    for feature, level, group_key, baseline, observed, proposed in features:
        suggestions.append(
            {
                "level": level,
                "group_key": group_key,
                "feature": feature,
                "baseline_median": baseline,
                "observed_median": observed,
                "proposed_new_median": proposed,
                "rationale": f"Observed {feature} has drifted from baseline. "
                f"Recommend adjusting threshold to reduce false positives.",
            }
        )

    return suggestions


def get_mock_location_heatmap(attribute_name: str | None = None) -> dict[str, Any]:
    """Get mock location heatmap data."""
    locations = []

    for store in MOCK_STORES:
        device_count = _rng.randint(5, 15)
        active_count = int(device_count * _rng.uniform(0.6, 0.95))
        utilization = round(active_count / device_count * 100, 1)
        baseline = _rng.randint(70, 85)

        locations.append(
            {
                "id": store["id"],
                "name": store["name"],
                "utilization": utilization,
                "baseline": baseline,
                "deviceCount": device_count,
                "activeDeviceCount": active_count,
                "region": store["region"],
                "anomalyCount": _rng.randint(0, 5),
            }
        )

    return {
        "locations": locations,
        "attributeName": attribute_name or "Store",
        "totalLocations": len(locations),
        "totalDevices": sum(loc["deviceCount"] for loc in locations),
    }


def get_mock_custom_attributes() -> dict[str, Any]:
    """Get mock available custom attributes."""
    return {
        "custom_attributes": [
            "Store",
            "Region",
            "Department",
            "Zone",
            "Warehouse",
        ],
        "error": None,
    }
