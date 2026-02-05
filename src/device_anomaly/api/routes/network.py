"""API routes for Network Intelligence dashboard.

Provides WiFi AP quality, carrier performance, per-app data usage,
and network health analytics with DeviceGroup hierarchy filtering.
"""
from __future__ import annotations

import logging
import random
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from device_anomaly.api.dependencies import get_mock_mode, get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/network", tags=["network-intelligence"])


# ============================================================================
# Response Models
# ============================================================================


class APQualityResponse(BaseModel):
    """WiFi access point quality metrics."""

    ssid: str
    bssid: str | None = None
    avg_signal_dbm: float = Field(description="Average signal strength in dBm")
    drop_rate: float = Field(description="Connection drop rate (0-1)")
    device_count: int = Field(description="Number of devices using this AP")
    quality_score: float = Field(description="Quality score 0-100")
    location: str | None = None


class PerAppUsageResponse(BaseModel):
    """Per-application network usage."""

    app_name: str
    app_id: int | None = None
    data_download_mb: float
    data_upload_mb: float
    device_count: int
    is_background: bool = False


class CarrierStatsResponse(BaseModel):
    """Carrier performance statistics."""

    carrier_name: str
    device_count: int
    avg_signal: float
    avg_latency_ms: float | None = None
    reliability_score: float


class NetworkSummaryResponse(BaseModel):
    """Network intelligence summary."""

    tenant_id: str
    device_group_id: int | None = None
    device_group_name: str | None = None
    device_group_path: str | None = None
    total_devices: int = 0
    total_aps: int
    good_aps: int
    problematic_aps: int
    avg_signal_strength: float
    avg_drop_rate: float
    fleet_network_score: float
    wifi_vs_cellular_ratio: float
    devices_in_dead_zones: int
    recommendations: list[str] = []
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class APListResponse(BaseModel):
    """List of access points with quality metrics."""

    aps: list[APQualityResponse]
    total_count: int


class AppUsageListResponse(BaseModel):
    """List of per-app network usage."""

    apps: list[PerAppUsageResponse]
    total_download_mb: float
    total_upload_mb: float


class CarrierListResponse(BaseModel):
    """List of carrier statistics."""

    carriers: list[CarrierStatsResponse]


class DeviceGroupNode(BaseModel):
    """Device group node in hierarchy tree."""

    device_group_id: int
    group_name: str
    parent_device_group_id: int | None = None
    device_count: int
    full_path: str
    children: list[DeviceGroupNode] = []


class DeviceGroupHierarchyResponse(BaseModel):
    """Device group hierarchy response."""

    groups: list[DeviceGroupNode]
    total_groups: int


# ============================================================================
# Mock Data Generators
# ============================================================================


def _generate_mock_hierarchy() -> DeviceGroupHierarchyResponse:
    """Generate mock device group hierarchy."""
    # Create a realistic hierarchy structure
    groups = [
        DeviceGroupNode(
            device_group_id=1,
            group_name="All Devices",
            parent_device_group_id=None,
            device_count=1250,
            full_path="All Devices",
            children=[
                DeviceGroupNode(
                    device_group_id=2,
                    group_name="North America",
                    parent_device_group_id=1,
                    device_count=850,
                    full_path="All Devices > North America",
                    children=[
                        DeviceGroupNode(
                            device_group_id=4,
                            group_name="Warehouse East",
                            parent_device_group_id=2,
                            device_count=320,
                            full_path="All Devices > North America > Warehouse East",
                            children=[
                                DeviceGroupNode(
                                    device_group_id=8,
                                    group_name="Floor 1",
                                    parent_device_group_id=4,
                                    device_count=160,
                                    full_path="All Devices > North America > Warehouse East > Floor 1",
                                    children=[],
                                ),
                                DeviceGroupNode(
                                    device_group_id=9,
                                    group_name="Floor 2",
                                    parent_device_group_id=4,
                                    device_count=160,
                                    full_path="All Devices > North America > Warehouse East > Floor 2",
                                    children=[],
                                ),
                            ],
                        ),
                        DeviceGroupNode(
                            device_group_id=5,
                            group_name="Warehouse West",
                            parent_device_group_id=2,
                            device_count=280,
                            full_path="All Devices > North America > Warehouse West",
                            children=[
                                DeviceGroupNode(
                                    device_group_id=10,
                                    group_name="Loading Dock",
                                    parent_device_group_id=5,
                                    device_count=80,
                                    full_path="All Devices > North America > Warehouse West > Loading Dock",
                                    children=[],
                                ),
                                DeviceGroupNode(
                                    device_group_id=11,
                                    group_name="Storage Area",
                                    parent_device_group_id=5,
                                    device_count=200,
                                    full_path="All Devices > North America > Warehouse West > Storage Area",
                                    children=[],
                                ),
                            ],
                        ),
                        DeviceGroupNode(
                            device_group_id=6,
                            group_name="Distribution Center",
                            parent_device_group_id=2,
                            device_count=250,
                            full_path="All Devices > North America > Distribution Center",
                            children=[],
                        ),
                    ],
                ),
                DeviceGroupNode(
                    device_group_id=3,
                    group_name="Europe",
                    parent_device_group_id=1,
                    device_count=400,
                    full_path="All Devices > Europe",
                    children=[
                        DeviceGroupNode(
                            device_group_id=7,
                            group_name="UK Operations",
                            parent_device_group_id=3,
                            device_count=220,
                            full_path="All Devices > Europe > UK Operations",
                            children=[],
                        ),
                        DeviceGroupNode(
                            device_group_id=12,
                            group_name="Germany Warehouse",
                            parent_device_group_id=3,
                            device_count=180,
                            full_path="All Devices > Europe > Germany Warehouse",
                            children=[],
                        ),
                    ],
                ),
            ],
        ),
    ]
    return DeviceGroupHierarchyResponse(groups=groups, total_groups=12)


def _get_mock_group_info(device_group_id: int | None) -> dict[str, Any]:
    """Get mock group info for a specific group ID."""
    mock_groups = {
        1: {"name": "All Devices", "path": "All Devices", "device_count": 1250},
        2: {
            "name": "North America",
            "path": "All Devices > North America",
            "device_count": 850,
        },
        3: {"name": "Europe", "path": "All Devices > Europe", "device_count": 400},
        4: {
            "name": "Warehouse East",
            "path": "All Devices > North America > Warehouse East",
            "device_count": 320,
        },
        5: {
            "name": "Warehouse West",
            "path": "All Devices > North America > Warehouse West",
            "device_count": 280,
        },
        6: {
            "name": "Distribution Center",
            "path": "All Devices > North America > Distribution Center",
            "device_count": 250,
        },
        7: {
            "name": "UK Operations",
            "path": "All Devices > Europe > UK Operations",
            "device_count": 220,
        },
        8: {
            "name": "Floor 1",
            "path": "All Devices > North America > Warehouse East > Floor 1",
            "device_count": 160,
        },
        9: {
            "name": "Floor 2",
            "path": "All Devices > North America > Warehouse East > Floor 2",
            "device_count": 160,
        },
        10: {
            "name": "Loading Dock",
            "path": "All Devices > North America > Warehouse West > Loading Dock",
            "device_count": 80,
        },
        11: {
            "name": "Storage Area",
            "path": "All Devices > North America > Warehouse West > Storage Area",
            "device_count": 200,
        },
        12: {
            "name": "Germany Warehouse",
            "path": "All Devices > Europe > Germany Warehouse",
            "device_count": 180,
        },
    }
    return mock_groups.get(
        device_group_id, {"name": None, "path": None, "device_count": 1250}
    )


def _generate_mock_aps(
    count: int = 20, device_group_id: int | None = None
) -> list[APQualityResponse]:
    """Generate mock AP quality data."""
    random.seed(42 if device_group_id is None else device_group_id)

    # Vary locations based on device_group_id
    location_sets = {
        None: [
            "Main Office",
            "Warehouse A",
            "Warehouse B",
            "Loading Dock",
            "Break Room",
            "Server Room",
        ],
        4: ["East Floor 1", "East Floor 2", "East Receiving", "East Shipping"],
        5: ["West Loading Dock", "West Storage A", "West Storage B", "West Office"],
        8: ["Floor 1 - Zone A", "Floor 1 - Zone B", "Floor 1 - Zone C"],
        10: ["Dock Bay 1", "Dock Bay 2", "Dock Bay 3", "Dock Office"],
    }
    locations = location_sets.get(
        device_group_id,
        [
            "Main Office",
            "Warehouse A",
            "Warehouse B",
            "Loading Dock",
            "Break Room",
            "Server Room",
        ],
    )

    # Scale count based on group
    group_info = _get_mock_group_info(device_group_id)
    effective_count = min(count, max(5, group_info["device_count"] // 10))

    return [
        APQualityResponse(
            ssid=f"CORP-WiFi-{str(i + 1).zfill(2)}",
            bssid=f"AA:BB:CC:DD:EE:{str(i).zfill(2)}",
            avg_signal_dbm=-45 - random.random() * 40,
            drop_rate=random.random() * 0.1,
            device_count=5 + int(random.random() * 30),
            quality_score=60 + random.random() * 40,
            location=locations[i % len(locations)],
        )
        for i in range(effective_count)
    ]


def _generate_mock_apps(
    device_group_id: int | None = None,
) -> list[PerAppUsageResponse]:
    """Generate mock per-app usage data."""
    # Base data
    apps = [
        PerAppUsageResponse(
            app_name="MobiControl Agent",
            data_download_mb=1250,
            data_upload_mb=890,
            device_count=450,
            is_background=True,
        ),
        PerAppUsageResponse(
            app_name="Google Play Services",
            data_download_mb=980,
            data_upload_mb=120,
            device_count=445,
            is_background=True,
        ),
        PerAppUsageResponse(
            app_name="Chrome",
            data_download_mb=2100,
            data_upload_mb=180,
            device_count=320,
            is_background=False,
        ),
        PerAppUsageResponse(
            app_name="Outlook",
            data_download_mb=650,
            data_upload_mb=420,
            device_count=280,
            is_background=False,
        ),
        PerAppUsageResponse(
            app_name="Teams",
            data_download_mb=1800,
            data_upload_mb=1200,
            device_count=190,
            is_background=False,
        ),
        PerAppUsageResponse(
            app_name="Warehouse Scanner",
            data_download_mb=45,
            data_upload_mb=890,
            device_count=150,
            is_background=False,
        ),
        PerAppUsageResponse(
            app_name="Unknown App",
            data_download_mb=2500,
            data_upload_mb=1800,
            device_count=8,
            is_background=True,
        ),
        PerAppUsageResponse(
            app_name="Spotify",
            data_download_mb=3200,
            data_upload_mb=5,
            device_count=45,
            is_background=False,
        ),
    ]

    # Scale based on group
    if device_group_id is not None:
        group_info = _get_mock_group_info(device_group_id)
        scale = group_info["device_count"] / 1250
        for app in apps:
            app.data_download_mb = int(app.data_download_mb * scale)
            app.data_upload_mb = int(app.data_upload_mb * scale)
            app.device_count = max(1, int(app.device_count * scale))

    return apps


def _generate_mock_carriers(
    device_group_id: int | None = None,
) -> list[CarrierStatsResponse]:
    """Generate mock carrier statistics."""
    carriers = [
        CarrierStatsResponse(
            carrier_name="Verizon",
            device_count=180,
            avg_signal=-78,
            avg_latency_ms=45,
            reliability_score=92,
        ),
        CarrierStatsResponse(
            carrier_name="AT&T",
            device_count=145,
            avg_signal=-82,
            avg_latency_ms=52,
            reliability_score=88,
        ),
        CarrierStatsResponse(
            carrier_name="T-Mobile",
            device_count=98,
            avg_signal=-75,
            avg_latency_ms=38,
            reliability_score=85,
        ),
        CarrierStatsResponse(
            carrier_name="Sprint/T-Mobile",
            device_count=27,
            avg_signal=-88,
            avg_latency_ms=65,
            reliability_score=72,
        ),
    ]

    # Scale based on group
    if device_group_id is not None:
        group_info = _get_mock_group_info(device_group_id)
        scale = group_info["device_count"] / 1250
        for carrier in carriers:
            carrier.device_count = max(1, int(carrier.device_count * scale))

    return carriers


def _generate_mock_summary(
    tenant_id: str, device_group_id: int | None = None
) -> NetworkSummaryResponse:
    """Generate mock network summary."""
    group_info = _get_mock_group_info(device_group_id)

    # Scale metrics based on group size
    scale = group_info["device_count"] / 1250 if device_group_id else 1.0

    # Generate contextual recommendations based on group
    recommendations = []
    if device_group_id is None:
        recommendations = [
            "Consider relocating or boosting signal for 3 APs in warehouse section",
            "High data usage detected from unknown app on 8 devices - investigate",
            "Cellular fallback increasing - verify WiFi coverage in loading dock area",
        ]
    elif device_group_id in [4, 5, 8, 9, 10, 11]:  # Warehouse/floor level
        recommendations = [
            f"Signal quality in {group_info['name']} is below average - consider AP repositioning",
            f"3 devices in {group_info['name']} have high disconnect rates",
            "Network congestion detected during shift changes",
        ]
    else:  # Region level
        recommendations = [
            f"Compare performance across sites in {group_info['name']}",
            "Warehouse East showing 15% better WiFi coverage than Warehouse West",
            "Consider standardizing AP configurations across the region",
        ]

    return NetworkSummaryResponse(
        tenant_id=tenant_id,
        device_group_id=device_group_id,
        device_group_name=group_info["name"],
        device_group_path=group_info["path"],
        total_devices=group_info["device_count"],
        total_aps=max(5, int(156 * scale)),
        good_aps=max(4, int(128 * scale)),
        problematic_aps=max(1, int(28 * scale)),
        avg_signal_strength=-62 + random.uniform(-5, 5),
        avg_drop_rate=0.034 + random.uniform(-0.01, 0.02),
        fleet_network_score=78.5 + random.uniform(-10, 10),
        wifi_vs_cellular_ratio=0.73 + random.uniform(-0.1, 0.1),
        devices_in_dead_zones=max(0, int(12 * scale)),
        recommendations=recommendations,
    )


# ============================================================================
# API Endpoints
# ============================================================================


@router.get("/hierarchy", response_model=DeviceGroupHierarchyResponse)
async def get_device_group_hierarchy(
    tenant_id: str = Depends(get_tenant_id),
    mock_mode: bool = Depends(get_mock_mode),
):
    """Get DeviceGroup hierarchy tree for navigation."""
    if mock_mode:
        return _generate_mock_hierarchy()

    # Real implementation - try device_group_service first, fall back to network_data
    try:
        from device_anomaly.services.device_group_service import (
            build_hierarchy_tree,
            load_device_groups_with_path,
        )

        groups = load_device_groups_with_path()

        if groups:
            tree = build_hierarchy_tree(groups)

            # Convert to response model
            def convert_to_node(g: dict[str, Any]) -> DeviceGroupNode:
                return DeviceGroupNode(
                    device_group_id=g["device_group_id"],
                    group_name=g["group_name"],
                    parent_device_group_id=g["parent_device_group_id"],
                    device_count=g["device_count"],
                    full_path=g["full_path"],
                    children=[convert_to_node(c) for c in g.get("children", [])],
                )

            response_groups = [convert_to_node(g) for g in tree]
            return DeviceGroupHierarchyResponse(
                groups=response_groups, total_groups=len(groups)
            )

        # Fallback to network_data module (uses conf_DeviceGroup from XSight DW)
        logger.info("device_group_service returned empty, trying network_data fallback")
        from device_anomaly.data_access.network_data import build_group_hierarchy, get_device_groups

        groups = get_device_groups()
        if not groups:
            logger.warning("No device groups found in either source")
            return DeviceGroupHierarchyResponse(groups=[], total_groups=0)

        tree = build_group_hierarchy(groups)

        def convert_dict_to_node(g: dict) -> DeviceGroupNode:
            return DeviceGroupNode(
                device_group_id=g["device_group_id"],
                group_name=g["group_name"],
                parent_device_group_id=g.get("parent_device_group_id"),
                device_count=g.get("device_count", 0),
                full_path=g["full_path"],
                children=[convert_dict_to_node(c) for c in g.get("children", [])],
            )

        response_groups = [convert_dict_to_node(g) for g in tree]
        logger.info("Hierarchy loaded from network_data: %d groups", len(groups))
        return DeviceGroupHierarchyResponse(
            groups=response_groups, total_groups=len(groups)
        )

    except Exception as e:
        logger.error(f"Failed to load device group hierarchy: {e}")
        return DeviceGroupHierarchyResponse(groups=[], total_groups=0)


@router.get("/summary", response_model=NetworkSummaryResponse)
async def get_network_summary(
    tenant_id: str = Depends(get_tenant_id),
    mock_mode: bool = Depends(get_mock_mode),
    device_group_id: int | None = Query(
        default=None, description="Filter by DeviceGroup ID"
    ),
):
    """Get fleet network intelligence summary, optionally filtered by device group."""
    if mock_mode:
        return _generate_mock_summary(tenant_id, device_group_id)

    # Real implementation: fetch from XSight database
    try:
        from device_anomaly.data_access.network_data import get_wifi_summary

        summary = get_wifi_summary(days=7, device_group_id=device_group_id)

        total_devices = summary.get("total_devices", 0)
        total_aps = summary.get("total_aps", 0)
        avg_signal = summary.get("avg_signal_strength", -70.0)
        avg_drop_rate = summary.get("avg_drop_rate", 0.0)

        # Calculate quality metrics
        # Good APs: signal > -70 dBm (estimated as 70% of total for now)
        good_aps = int(total_aps * 0.7) if total_aps > 0 else 0
        problematic_aps = total_aps - good_aps

        # Fleet network score: combination of signal quality and drop rate
        # Signal score: -30 dBm = 100, -90 dBm = 0
        signal_score = max(0, min(100, (avg_signal + 90) / 60 * 100))
        drop_score = (1 - avg_drop_rate) * 100
        fleet_network_score = signal_score * 0.6 + drop_score * 0.4

        # Devices in dead zones: estimate based on signal distribution
        # This would need more detailed data; for now estimate 5% if signal is poor
        devices_in_dead_zones = int(total_devices * 0.05) if avg_signal < -75 else 0

        # Generate recommendations based on actual metrics
        recommendations = []
        if avg_signal < -75:
            recommendations.append(f"Average signal strength ({avg_signal:.0f} dBm) is below optimal - consider adding access points")
        if avg_drop_rate > 0.05:
            recommendations.append(f"High disconnect rate detected ({avg_drop_rate*100:.1f}%) - investigate network stability")
        if problematic_aps > 0:
            recommendations.append(f"{problematic_aps} access points have below-average performance")
        if not recommendations:
            recommendations.append("Network performance is within acceptable parameters")

        logger.info(
            "Network summary: %d devices, %d APs, avg signal %.1f dBm",
            total_devices, total_aps, avg_signal
        )

        return NetworkSummaryResponse(
            tenant_id=tenant_id,
            device_group_id=device_group_id,
            total_devices=total_devices,
            total_aps=total_aps,
            good_aps=good_aps,
            problematic_aps=problematic_aps,
            avg_signal_strength=avg_signal,
            avg_drop_rate=avg_drop_rate,
            fleet_network_score=round(fleet_network_score, 1),
            wifi_vs_cellular_ratio=0.8,  # Would need additional data source
            devices_in_dead_zones=devices_in_dead_zones,
            recommendations=recommendations,
            generated_at=datetime.now(UTC),
        )

    except Exception as e:
        logger.error(f"Error fetching network summary: {e}")
        return NetworkSummaryResponse(
            tenant_id=tenant_id,
            device_group_id=device_group_id,
            total_devices=0,
            total_aps=0,
            good_aps=0,
            problematic_aps=0,
            avg_signal_strength=0.0,
            avg_drop_rate=0.0,
            fleet_network_score=0.0,
            wifi_vs_cellular_ratio=0.0,
            devices_in_dead_zones=0,
            recommendations=[f"Error loading network data: {str(e)[:100]}"],
            generated_at=datetime.now(UTC),
        )


@router.get("/aps", response_model=APListResponse)
async def get_ap_quality(
    tenant_id: str = Depends(get_tenant_id),
    mock_mode: bool = Depends(get_mock_mode),
    limit: int = Query(default=50, le=500),
    min_device_count: int = Query(default=1, ge=0),
    device_group_id: int | None = Query(
        default=None, description="Filter by DeviceGroup ID"
    ),
):
    """Get WiFi access point quality metrics."""
    if mock_mode:
        aps = _generate_mock_aps(limit, device_group_id)
        return APListResponse(aps=aps, total_count=len(aps))

    # Real implementation: fetch from XSight database
    try:
        from device_anomaly.data_access.network_data import get_ap_quality_metrics

        ap_data = get_ap_quality_metrics(
            days=7,
            limit=limit,
            min_device_count=min_device_count,
            device_group_id=device_group_id
        )

        aps = [
            APQualityResponse(
                ssid=ap["ssid"],
                bssid=ap.get("bssid"),
                avg_signal_dbm=ap["avg_signal_dbm"],
                drop_rate=ap["drop_rate"],
                device_count=ap["device_count"],
                quality_score=ap["quality_score"],
                location=ap.get("location"),
            )
            for ap in ap_data
        ]

        logger.info("AP quality: returning %d access points", len(aps))
        return APListResponse(aps=aps, total_count=len(aps))

    except Exception as e:
        logger.error(f"Error fetching AP quality: {e}")
        return APListResponse(aps=[], total_count=0)


@router.get("/apps", response_model=AppUsageListResponse)
async def get_per_app_usage(
    tenant_id: str = Depends(get_tenant_id),
    mock_mode: bool = Depends(get_mock_mode),
    period_days: int = Query(default=7, ge=1, le=90),
    device_group_id: int | None = Query(
        default=None, description="Filter by DeviceGroup ID"
    ),
):
    """Get per-application network usage statistics."""
    if mock_mode:
        apps = _generate_mock_apps(device_group_id)
        return AppUsageListResponse(
            apps=apps,
            total_download_mb=sum(a.data_download_mb for a in apps),
            total_upload_mb=sum(a.data_upload_mb for a in apps),
        )

    # Real implementation: fetch from XSight database
    try:
        from device_anomaly.data_access.network_data import get_app_usage_metrics

        app_data = get_app_usage_metrics(days=period_days, device_group_id=device_group_id)

        apps = [
            PerAppUsageResponse(
                app_name=app["app_name"],
                app_id=app.get("app_id"),
                data_download_mb=app["data_download_mb"],
                data_upload_mb=app["data_upload_mb"],
                device_count=app["device_count"],
                is_background=app.get("is_background", False),
            )
            for app in app_data
        ]

        total_download = sum(a.data_download_mb for a in apps)
        total_upload = sum(a.data_upload_mb for a in apps)

        logger.info(
            "App usage: %d apps, %.1f MB download, %.1f MB upload",
            len(apps), total_download, total_upload
        )

        return AppUsageListResponse(
            apps=apps,
            total_download_mb=total_download,
            total_upload_mb=total_upload,
        )

    except Exception as e:
        logger.error(f"Error fetching app usage: {e}")
        return AppUsageListResponse(
            apps=[],
            total_download_mb=0.0,
            total_upload_mb=0.0,
        )


@router.get("/carriers", response_model=CarrierListResponse)
async def get_carrier_stats(
    tenant_id: str = Depends(get_tenant_id),
    mock_mode: bool = Depends(get_mock_mode),
    device_group_id: int | None = Query(
        default=None, description="Filter by DeviceGroup ID"
    ),
):
    """Get carrier performance statistics."""
    if mock_mode:
        return CarrierListResponse(carriers=_generate_mock_carriers(device_group_id))

    # Real implementation: fetch from XSight database
    try:
        from device_anomaly.data_access.network_data import get_carrier_metrics

        carrier_data = get_carrier_metrics(device_group_id=device_group_id)

        carriers = [
            CarrierStatsResponse(
                carrier_name=c["carrier_name"],
                device_count=c["device_count"],
                avg_signal=c["avg_signal"],
                avg_latency_ms=c.get("avg_latency_ms"),
                reliability_score=c["reliability_score"],
            )
            for c in carrier_data
        ]

        logger.info("Carrier stats: returning %d carriers", len(carriers))
        return CarrierListResponse(carriers=carriers)

    except Exception as e:
        logger.error(f"Error fetching carrier stats: {e}")
        return CarrierListResponse(carriers=[])


@router.get("/dead-zones")
async def get_dead_zones(
    tenant_id: str = Depends(get_tenant_id),
    mock_mode: bool = Depends(get_mock_mode),
    signal_threshold_dbm: int = Query(
        default=-85, description="Signal below this is considered dead zone"
    ),
    device_group_id: int | None = Query(
        default=None, description="Filter by DeviceGroup ID"
    ),
):
    """Get geographic dead zones based on signal quality."""
    if mock_mode:
        # Scale dead zones based on group
        group_info = _get_mock_group_info(device_group_id)
        scale = group_info["device_count"] / 1250 if device_group_id else 1.0
        dead_zone_count = max(0, int(2 * scale))

        return {
            "dead_zones": [
                {
                    "latitude": 40.7128 + i * 0.01,
                    "longitude": -74.0060 + i * 0.01,
                    "avg_signal_dbm": -92 + i * 2,
                    "device_count": max(1, int(5 * scale)),
                }
                for i in range(dead_zone_count)
            ],
            "total_count": dead_zone_count,
            "signal_threshold_dbm": signal_threshold_dbm,
            "device_group_id": device_group_id,
        }

    logger.info(
        "Dead zones requested for tenant %s, group %s", tenant_id, device_group_id
    )
    return {
        "dead_zones": [],
        "total_count": 0,
        "signal_threshold_dbm": signal_threshold_dbm,
        "device_group_id": device_group_id,
    }
