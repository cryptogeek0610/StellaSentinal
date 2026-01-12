"""API routes for location intelligence."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from device_anomaly.api.dependencies import get_mock_mode, get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/insights/location", tags=["location-intelligence"])


# ============================================================================
# Response Models
# ============================================================================

class GeoBoundsResponse(BaseModel):
    """Geographic bounding box."""
    min_lat: float
    max_lat: float
    min_long: float
    max_long: float


class HeatmapCellResponse(BaseModel):
    """Single cell in a WiFi heatmap grid."""
    lat: float
    long: float
    signal_strength: float
    reading_count: int
    is_dead_zone: bool = False
    access_point_id: Optional[str] = None


class WiFiHeatmapResponse(BaseModel):
    """Complete WiFi heatmap response."""
    tenant_id: str
    grid_cells: List[HeatmapCellResponse]
    bounds: Optional[GeoBoundsResponse]
    total_readings: int
    avg_signal_strength: float
    dead_zone_count: int
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DeadZoneResponse(BaseModel):
    """Dead zone location data."""
    zone_id: str
    lat: float
    long: float
    avg_signal: float
    affected_devices: int
    total_readings: int
    first_detected: Optional[datetime] = None
    last_detected: Optional[datetime] = None


class DeadZonesResponse(BaseModel):
    """List of dead zones."""
    tenant_id: str
    dead_zones: List[DeadZoneResponse]
    total_count: int
    recommendations: List[str]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MovementPointResponse(BaseModel):
    """Single point in device movement history."""
    timestamp: datetime
    lat: float
    long: float
    speed: float
    heading: float


class DeviceMovementResponse(BaseModel):
    """Device movement summary."""
    device_id: int
    movements: List[MovementPointResponse]
    total_distance_km: float
    avg_speed_kmh: float
    stationary_time_pct: float
    active_hours: List[int]


class DwellZoneResponse(BaseModel):
    """Location where devices spend significant time."""
    zone_id: str
    lat: float
    long: float
    avg_dwell_minutes: float
    device_count: int
    visit_count: int
    peak_hours: List[int]


class DwellTimeResponse(BaseModel):
    """Dwell time analysis response."""
    tenant_id: str
    dwell_zones: List[DwellZoneResponse]
    total_zones: int
    recommendations: List[str]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CoverageSummaryResponse(BaseModel):
    """WiFi coverage summary."""
    tenant_id: str
    total_readings: int
    avg_signal: float
    coverage_distribution: dict
    coverage_percentage: float
    recommendations: List[str]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# Mock Data Functions
# ============================================================================

def get_mock_wifi_heatmap(period_days: int) -> WiFiHeatmapResponse:
    """Generate mock WiFi heatmap data."""
    import random
    random.seed(42)

    # Generate a grid of cells around a central location
    base_lat, base_long = 37.7749, -122.4194  # San Francisco
    cells = []

    for i in range(-10, 11):
        for j in range(-10, 11):
            lat = base_lat + (i * 0.001)
            long = base_long + (j * 0.001)

            # Generate signal strength with some patterns
            distance = abs(i) + abs(j)
            base_signal = -45 - (distance * 2) + random.uniform(-5, 5)

            # Create some dead zones
            is_dead = (i == 5 and j == 5) or (i == -3 and j == -7)
            if is_dead:
                base_signal = -88 + random.uniform(-3, 3)

            cells.append(HeatmapCellResponse(
                lat=lat,
                long=long,
                signal_strength=base_signal,
                reading_count=random.randint(10, 100),
                is_dead_zone=base_signal < -85,
                access_point_id=f"AP_{(i + 10) % 5}_{(j + 10) % 5}" if not is_dead else None,
            ))

    dead_count = sum(1 for c in cells if c.is_dead_zone)

    return WiFiHeatmapResponse(
        tenant_id="default",
        grid_cells=cells,
        bounds=GeoBoundsResponse(
            min_lat=base_lat - 0.011,
            max_lat=base_lat + 0.011,
            min_long=base_long - 0.011,
            max_long=base_long + 0.011,
        ),
        total_readings=sum(c.reading_count for c in cells),
        avg_signal_strength=sum(c.signal_strength for c in cells) / len(cells),
        dead_zone_count=dead_count,
    )


def get_mock_dead_zones() -> DeadZonesResponse:
    """Generate mock dead zone data."""
    now = datetime.now(timezone.utc)
    zones = [
        DeadZoneResponse(
            zone_id="dz_0",
            lat=37.7754,
            long=-122.4189,
            avg_signal=-88.5,
            affected_devices=12,
            total_readings=156,
            first_detected=now - timedelta(days=14),
            last_detected=now - timedelta(hours=2),
        ),
        DeadZoneResponse(
            zone_id="dz_1",
            lat=37.7746,
            long=-122.4201,
            avg_signal=-86.2,
            affected_devices=8,
            total_readings=89,
            first_detected=now - timedelta(days=7),
            last_detected=now - timedelta(hours=5),
        ),
    ]

    return DeadZonesResponse(
        tenant_id="default",
        dead_zones=zones,
        total_count=len(zones),
        recommendations=[
            "Consider adding an access point near (37.775, -122.419) - 12 devices affected",
            "Investigate signal interference at (37.775, -122.420)",
            "Review AP placement in warehouse section B",
        ],
    )


def get_mock_device_movements(device_id: int) -> DeviceMovementResponse:
    """Generate mock device movement data."""
    import random
    random.seed(device_id)

    now = datetime.now(timezone.utc)
    base_lat, base_long = 37.7749, -122.4194
    movements = []

    for i in range(48):  # 48 hours of data
        ts = now - timedelta(hours=48 - i)
        lat = base_lat + random.uniform(-0.005, 0.005)
        long = base_long + random.uniform(-0.005, 0.005)
        speed = random.uniform(0, 15) if random.random() > 0.3 else 0

        movements.append(MovementPointResponse(
            timestamp=ts,
            lat=lat,
            long=long,
            speed=speed,
            heading=random.uniform(0, 360),
        ))

    return DeviceMovementResponse(
        device_id=device_id,
        movements=movements,
        total_distance_km=random.uniform(5, 25),
        avg_speed_kmh=random.uniform(3, 8),
        stationary_time_pct=random.uniform(30, 70),
        active_hours=[8, 9, 10, 11, 13, 14, 15, 16],
    )


def get_mock_dwell_time() -> DwellTimeResponse:
    """Generate mock dwell time data."""
    zones = [
        DwellZoneResponse(
            zone_id="dwell_0",
            lat=37.7751,
            long=-122.4190,
            avg_dwell_minutes=45.5,
            device_count=28,
            visit_count=156,
            peak_hours=[9, 10, 11, 14, 15],
        ),
        DwellZoneResponse(
            zone_id="dwell_1",
            lat=37.7745,
            long=-122.4198,
            avg_dwell_minutes=22.3,
            device_count=15,
            visit_count=89,
            peak_hours=[10, 11, 15, 16],
        ),
        DwellZoneResponse(
            zone_id="dwell_2",
            lat=37.7748,
            long=-122.4185,
            avg_dwell_minutes=15.8,
            device_count=42,
            visit_count=210,
            peak_hours=[8, 9, 17, 18],
        ),
    ]

    return DwellTimeResponse(
        tenant_id="default",
        dwell_zones=zones,
        total_zones=len(zones),
        recommendations=[
            "High dwell time at shipping dock (45 min avg) - consider process optimization",
            "Entry/exit zone shows normal dwell patterns",
        ],
    )


def get_mock_coverage_summary() -> CoverageSummaryResponse:
    """Generate mock coverage summary."""
    return CoverageSummaryResponse(
        tenant_id="default",
        total_readings=15680,
        avg_signal=-62.4,
        coverage_distribution={
            "excellent": 2340,
            "good": 5670,
            "fair": 4890,
            "poor": 2100,
            "dead": 680,
        },
        coverage_percentage=95.7,
        recommendations=[
            "Overall coverage is good at 95.7%",
            "2 dead zones identified - consider AP placement review",
            "Signal quality in warehouse section B could be improved",
        ],
    )


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/wifi-heatmap", response_model=WiFiHeatmapResponse)
def get_wifi_heatmap(
    period_days: int = Query(7, ge=1, le=30, description="Analysis period in days"),
    grid_size: float = Query(0.001, ge=0.0001, le=0.01, description="Grid cell size in degrees"),
    min_signal: int = Query(-100, ge=-120, le=-40, description="Minimum signal strength to include"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Get WiFi signal heatmap data.

    Returns a grid of cells with average signal strength for visualization.
    Grid size of 0.001 degrees is approximately 100 meters.
    """
    if mock_mode:
        return get_mock_wifi_heatmap(period_days)

    tenant_id = get_tenant_id()

    try:
        from device_anomaly.data_access.location_intelligence_loader import load_wifi_heatmap

        heatmap_data = load_wifi_heatmap(
            period_days=period_days,
            grid_size=grid_size,
            min_signal=min_signal,
        )

        cells = [
            HeatmapCellResponse(
                lat=c.lat,
                long=c.long,
                signal_strength=c.signal_strength,
                reading_count=c.reading_count,
                is_dead_zone=c.is_dead_zone,
                access_point_id=c.access_point_id,
            )
            for c in heatmap_data.grid_cells
        ]

        bounds = None
        if heatmap_data.bounds:
            bounds = GeoBoundsResponse(
                min_lat=heatmap_data.bounds.min_lat,
                max_lat=heatmap_data.bounds.max_lat,
                min_long=heatmap_data.bounds.min_long,
                max_long=heatmap_data.bounds.max_long,
            )

        return WiFiHeatmapResponse(
            tenant_id=tenant_id,
            grid_cells=cells,
            bounds=bounds,
            total_readings=heatmap_data.total_readings,
            avg_signal_strength=heatmap_data.avg_signal_strength,
            dead_zone_count=heatmap_data.dead_zone_count,
        )

    except Exception as e:
        logger.error(f"Failed to get WiFi heatmap: {e}")
        return WiFiHeatmapResponse(
            tenant_id=tenant_id,
            grid_cells=[],
            bounds=None,
            total_readings=0,
            avg_signal_strength=0,
            dead_zone_count=0,
        )


@router.get("/dead-zones", response_model=DeadZonesResponse)
def get_dead_zones(
    period_days: int = Query(7, ge=1, le=30, description="Analysis period in days"),
    signal_threshold: int = Query(-75, ge=-100, le=-50, description="Signal threshold for dead zone"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Identify areas with consistently poor WiFi signal.

    Dead zones are areas where signal strength is consistently below the threshold.
    """
    if mock_mode:
        return get_mock_dead_zones()

    tenant_id = get_tenant_id()

    try:
        from device_anomaly.data_access.location_intelligence_loader import detect_dead_zones

        dead_zones = detect_dead_zones(
            period_days=period_days,
            signal_threshold=signal_threshold,
        )

        zones = [
            DeadZoneResponse(
                zone_id=z.zone_id,
                lat=z.lat,
                long=z.long,
                avg_signal=z.avg_signal,
                affected_devices=z.affected_devices,
                total_readings=z.total_readings,
                first_detected=z.first_detected,
                last_detected=z.last_detected,
            )
            for z in dead_zones
        ]

        recommendations = []
        for z in zones[:3]:  # Top 3 recommendations
            recommendations.append(
                f"Consider adding AP near ({z.lat:.4f}, {z.long:.4f}) - {z.affected_devices} devices affected"
            )
        if not recommendations:
            recommendations.append("No significant dead zones detected")

        return DeadZonesResponse(
            tenant_id=tenant_id,
            dead_zones=zones,
            total_count=len(zones),
            recommendations=recommendations,
        )

    except Exception as e:
        logger.error(f"Failed to get dead zones: {e}")
        return DeadZonesResponse(
            tenant_id=tenant_id,
            dead_zones=[],
            total_count=0,
            recommendations=["Unable to analyze dead zones - check database connections"],
        )


@router.get("/device-movements/{device_id}", response_model=DeviceMovementResponse)
def get_device_movements(
    device_id: int,
    period_days: int = Query(7, ge=1, le=30, description="Analysis period in days"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Get GPS movement history for a specific device.

    Returns movement points with distance and speed statistics.
    """
    if mock_mode:
        return get_mock_device_movements(device_id)

    try:
        from device_anomaly.data_access.location_intelligence_loader import load_device_movements

        movement_data = load_device_movements(
            device_id=device_id,
            period_days=period_days,
        )

        movements = [
            MovementPointResponse(
                timestamp=m.timestamp,
                lat=m.lat,
                long=m.long,
                speed=m.speed,
                heading=m.heading,
            )
            for m in movement_data.movements
        ]

        return DeviceMovementResponse(
            device_id=device_id,
            movements=movements,
            total_distance_km=movement_data.total_distance_km,
            avg_speed_kmh=movement_data.avg_speed_kmh,
            stationary_time_pct=movement_data.stationary_time_pct,
            active_hours=movement_data.active_hours,
        )

    except Exception as e:
        logger.error(f"Failed to get device movements: {e}")
        return DeviceMovementResponse(
            device_id=device_id,
            movements=[],
            total_distance_km=0,
            avg_speed_kmh=0,
            stationary_time_pct=0,
            active_hours=[],
        )


@router.get("/dwell-time", response_model=DwellTimeResponse)
def get_dwell_time(
    period_days: int = Query(7, ge=1, le=30, description="Analysis period in days"),
    min_dwell_minutes: int = Query(5, ge=1, le=60, description="Minimum dwell time to consider"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Analyze where devices spend significant time.

    Identifies zones with high device dwell time for operational insights.
    """
    if mock_mode:
        return get_mock_dwell_time()

    tenant_id = get_tenant_id()

    try:
        from device_anomaly.data_access.location_intelligence_loader import load_dwell_time_analysis

        dwell_data = load_dwell_time_analysis(
            period_days=period_days,
            min_dwell_minutes=min_dwell_minutes,
        )

        zones = [
            DwellZoneResponse(
                zone_id=z.zone_id,
                lat=z.lat,
                long=z.long,
                avg_dwell_minutes=z.avg_dwell_minutes,
                device_count=z.device_count,
                visit_count=z.visit_count,
                peak_hours=z.peak_hours,
            )
            for z in dwell_data
        ]

        recommendations = []
        for z in zones[:2]:
            if z.avg_dwell_minutes > 30:
                recommendations.append(
                    f"High dwell time at ({z.lat:.4f}, {z.long:.4f}) - {z.avg_dwell_minutes:.0f} min avg"
                )

        return DwellTimeResponse(
            tenant_id=tenant_id,
            dwell_zones=zones,
            total_zones=len(zones),
            recommendations=recommendations if recommendations else ["Dwell patterns appear normal"],
        )

    except Exception as e:
        logger.error(f"Failed to get dwell time: {e}")
        return DwellTimeResponse(
            tenant_id=tenant_id,
            dwell_zones=[],
            total_zones=0,
            recommendations=["Unable to analyze dwell time - check database connections"],
        )


@router.get("/coverage-summary", response_model=CoverageSummaryResponse)
def get_coverage_summary(
    period_days: int = Query(7, ge=1, le=30, description="Analysis period in days"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Get summary statistics about WiFi coverage.

    Returns distribution of signal quality across the monitored area.
    """
    if mock_mode:
        return get_mock_coverage_summary()

    tenant_id = get_tenant_id()

    try:
        from device_anomaly.data_access.location_intelligence_loader import get_coverage_summary as get_coverage

        coverage = get_coverage(period_days=period_days)

        dist = coverage.get("coverage_distribution", {})
        total = sum(dist.values()) if dist else 0
        good_and_above = dist.get("excellent", 0) + dist.get("good", 0) + dist.get("fair", 0)
        coverage_pct = (good_and_above / total * 100) if total > 0 else 0

        recommendations = []
        if coverage_pct >= 95:
            recommendations.append(f"Overall coverage is excellent at {coverage_pct:.1f}%")
        elif coverage_pct >= 80:
            recommendations.append(f"Coverage is good at {coverage_pct:.1f}% - minor improvements possible")
        else:
            recommendations.append(f"Coverage at {coverage_pct:.1f}% needs improvement")

        dead_count = dist.get("dead", 0)
        if dead_count > 0:
            recommendations.append(f"{dead_count} readings in dead zones - review AP placement")

        return CoverageSummaryResponse(
            tenant_id=tenant_id,
            total_readings=coverage.get("total_readings", 0),
            avg_signal=coverage.get("avg_signal", 0),
            coverage_distribution=dist,
            coverage_percentage=coverage_pct,
            recommendations=recommendations,
        )

    except Exception as e:
        logger.error(f"Failed to get coverage summary: {e}")
        return CoverageSummaryResponse(
            tenant_id=tenant_id,
            total_readings=0,
            avg_signal=0,
            coverage_distribution={},
            coverage_percentage=0,
            recommendations=["Unable to analyze coverage - check database connections"],
        )
