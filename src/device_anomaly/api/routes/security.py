"""API routes for Security Posture dashboard.

Provides fleet security compliance, device risk assessment,
encryption status, security trend analytics, and PATH-based grouping.
"""

from __future__ import annotations

import logging
import random
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from device_anomaly.api.dependencies import get_mock_mode, get_tenant_id
from device_anomaly.services.path_builder import get_path_hierarchy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/security", tags=["security-posture"])


# ============================================================================
# Response Models
# ============================================================================


class DeviceSecurityResponse(BaseModel):
    """Individual device security status."""

    device_id: int
    device_name: str
    device_path: str | None = Field(default=None, description="Full hierarchical path")
    security_score: float = Field(description="Security score 0-100")
    is_encrypted: bool
    is_rooted: bool
    has_passcode: bool
    patch_age_days: int
    usb_debugging: bool
    developer_mode: bool
    risk_level: str = Field(description="low, medium, high, or critical")
    violations: list[str] = []


class ComplianceBreakdownResponse(BaseModel):
    """Compliance category breakdown."""

    category: str
    compliant: int
    non_compliant: int
    total: int
    compliance_pct: float


class SecurityTrendResponse(BaseModel):
    """Security trend data point."""

    date: str
    score: float
    compliant_pct: float


class SecuritySummaryResponse(BaseModel):
    """Fleet security posture summary."""

    tenant_id: str
    fleet_security_score: float = Field(description="Overall fleet security score 0-100")
    total_devices: int
    compliant_devices: int
    at_risk_devices: int
    critical_risk_devices: int
    encrypted_devices: int
    rooted_devices: int
    outdated_patch_devices: int
    usb_debugging_enabled: int
    developer_mode_enabled: int
    no_passcode_devices: int
    recommendations: list[str] = []
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DeviceListResponse(BaseModel):
    """List of devices with security status."""

    devices: list[DeviceSecurityResponse]
    total_count: int


class ComplianceListResponse(BaseModel):
    """Compliance breakdown by category."""

    categories: list[ComplianceBreakdownResponse]


class TrendListResponse(BaseModel):
    """Security trend data."""

    trends: list[SecurityTrendResponse]
    period_days: int


# ============================================================================
# PATH-Based Response Models
# ============================================================================


class PathNodeResponse(BaseModel):
    """A node in the PATH hierarchy tree."""

    path_id: str
    path_name: str
    full_path: str
    parent_path_id: str | None = None
    depth: int
    device_count: int
    security_score: float = 0.0
    compliant_count: int = 0
    at_risk_count: int = 0
    critical_count: int = 0
    children: list[PathNodeResponse] = []


class PathHierarchyResponse(BaseModel):
    """PATH hierarchy with security metrics."""

    tenant_id: str
    hierarchy: list[PathNodeResponse]
    total_paths: int
    total_devices: int
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PathSecuritySummaryResponse(BaseModel):
    """Security summary for a specific PATH."""

    path: str
    path_name: str
    device_count: int
    security_score: float
    compliant_count: int
    at_risk_count: int
    critical_count: int
    rooted_count: int
    unencrypted_count: int
    no_passcode_count: int
    outdated_patch_count: int
    usb_debugging_count: int
    developer_mode_count: int


class PathSecurityResponse(BaseModel):
    """Security summaries grouped by PATH."""

    tenant_id: str
    summaries: list[PathSecuritySummaryResponse]
    selected_path: str | None = None
    depth: int
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RiskClusterResponse(BaseModel):
    """A cluster of devices with similar security violations."""

    cluster_id: str
    cluster_name: str
    violation_type: str
    severity: str
    device_count: int
    avg_security_score: float
    affected_paths: list[str]
    first_detected: datetime | None = None
    sample_devices: list[DeviceSecurityResponse]
    recommendation: str


class RiskClustersResponse(BaseModel):
    """Collection of risk clusters."""

    tenant_id: str
    clusters: list[RiskClusterResponse]
    total_devices_affected: int
    coverage_percent: float
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PathComparisonItemResponse(BaseModel):
    """Comparison data for a single PATH."""

    path: str
    path_name: str
    security_score: float
    device_count: int
    compliant_pct: float
    rooted_count: int
    unencrypted_count: int
    outdated_patch_count: int
    delta_from_fleet: float


class PathComparisonResponse(BaseModel):
    """Comparison results across multiple PATHs."""

    tenant_id: str
    paths: list[PathComparisonItemResponse]
    fleet_average_score: float
    best_path: str | None = None
    worst_path: str | None = None
    insights: list[str] = []
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TemporalClusterResponse(BaseModel):
    """Devices with correlated security issues appearing around the same time."""

    cluster_id: str
    cluster_name: str
    issue_appeared_at: datetime
    device_count: int
    common_violations: list[str]
    affected_paths: list[str]
    correlation_insight: str
    sample_devices: list[DeviceSecurityResponse]


class TemporalClustersResponse(BaseModel):
    """Collection of temporal correlation clusters."""

    tenant_id: str
    clusters: list[TemporalClusterResponse]
    window_hours: int
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ============================================================================
# Mock Data Generators
# ============================================================================

MOCK_PATHS = [
    "North America / East Region / Store-NYC-001",
    "North America / East Region / Store-NYC-002",
    "North America / East Region / Warehouse-NYC",
    "North America / West Region / Store-LA-001",
    "North America / West Region / Store-LA-002",
    "North America / West Region / Distribution-LA",
    "Europe / UK / Store-London-001",
    "Europe / UK / Warehouse-Manchester",
    "Europe / Germany / Store-Berlin-001",
]


def _generate_mock_devices(
    count: int = 25, with_paths: bool = True
) -> list[DeviceSecurityResponse]:
    """Generate mock device security data with optional PATH assignment."""
    random.seed(42)
    devices = []
    for i in range(count):
        is_rooted = random.random() < 0.02
        is_encrypted = random.random() > 0.03
        has_passcode = random.random() > 0.01
        usb_debugging = random.random() < 0.05
        dev_mode = random.random() < 0.07
        patch_age = int(random.random() * 120)

        violations = []
        if is_rooted:
            violations.append("Device is rooted")
        if not is_encrypted:
            violations.append("Storage not encrypted")
        if not has_passcode:
            violations.append("No passcode set")
        if usb_debugging:
            violations.append("USB debugging enabled")
        if dev_mode:
            violations.append("Developer mode enabled")
        if patch_age > 60:
            violations.append(f"Security patch {patch_age} days old")

        score = 100.0
        if is_rooted:
            score -= 40
        if not is_encrypted:
            score -= 25
        if not has_passcode:
            score -= 15
        if usb_debugging:
            score -= 10
        if dev_mode:
            score -= 5
        if patch_age > 60:
            score -= min(20, patch_age / 3)

        risk_level = "low"
        if score < 50:
            risk_level = "critical"
        elif score < 70:
            risk_level = "high"
        elif score < 85:
            risk_level = "medium"

        device_path = MOCK_PATHS[i % len(MOCK_PATHS)] if with_paths else None

        devices.append(
            DeviceSecurityResponse(
                device_id=1000 + i,
                device_name=f"Device-{str(i + 1).zfill(3)}",
                device_path=device_path,
                security_score=max(0, score),
                is_encrypted=is_encrypted,
                is_rooted=is_rooted,
                has_passcode=has_passcode,
                patch_age_days=patch_age,
                usb_debugging=usb_debugging,
                developer_mode=dev_mode,
                risk_level=risk_level,
                violations=violations,
            )
        )
    return sorted(devices, key=lambda d: d.security_score)


def _generate_mock_compliance() -> list[ComplianceBreakdownResponse]:
    """Generate mock compliance breakdown."""
    data = [
        ("Encryption", 438, 12),
        ("Passcode", 445, 5),
        ("Patch Level", 405, 45),
        ("Root Status", 442, 8),
        ("Developer Mode", 419, 31),
        ("USB Debugging", 427, 23),
    ]
    return [
        ComplianceBreakdownResponse(
            category=cat,
            compliant=comp,
            non_compliant=non_comp,
            total=comp + non_comp,
            compliance_pct=round(comp / (comp + non_comp) * 100, 1),
        )
        for cat, comp, non_comp in data
    ]


def _generate_mock_trends(days: int = 30) -> list[SecurityTrendResponse]:
    """Generate mock security trends."""
    random.seed(42)
    today = datetime.now(UTC)
    return [
        SecurityTrendResponse(
            date=(today - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d"),
            score=65 + random.random() * 15 + (i * 0.2),
            compliant_pct=80 + random.random() * 10 + (i * 0.1),
        )
        for i in range(days)
    ]


def _generate_mock_summary(tenant_id: str) -> SecuritySummaryResponse:
    """Generate mock security summary."""
    return SecuritySummaryResponse(
        tenant_id=tenant_id,
        fleet_security_score=72.5,
        total_devices=450,
        compliant_devices=385,
        at_risk_devices=52,
        critical_risk_devices=13,
        encrypted_devices=438,
        rooted_devices=8,
        outdated_patch_devices=45,
        usb_debugging_enabled=23,
        developer_mode_enabled=31,
        no_passcode_devices=5,
        recommendations=[
            "Enforce encryption on 12 remaining unencrypted devices",
            "Investigate 8 rooted devices for potential security compromise",
            "Push security patches to 45 devices with outdated patches (>60 days)",
            "Disable developer mode on 31 devices in production environment",
            "Enforce passcode policy on 5 non-compliant devices",
        ],
    )


def _generate_mock_path_hierarchy(tenant_id: str) -> PathHierarchyResponse:
    """Generate mock PATH hierarchy with security metrics."""
    hierarchy = [
        PathNodeResponse(
            path_id="group-1",
            path_name="North America",
            full_path="North America",
            depth=0,
            device_count=300,
            security_score=74.2,
            compliant_count=258,
            at_risk_count=35,
            critical_count=7,
            children=[
                PathNodeResponse(
                    path_id="group-2",
                    path_name="East Region",
                    full_path="North America / East Region",
                    parent_path_id="group-1",
                    depth=1,
                    device_count=150,
                    security_score=78.5,
                    compliant_count=132,
                    at_risk_count=15,
                    critical_count=3,
                    children=[
                        PathNodeResponse(
                            path_id="group-3",
                            path_name="Store-NYC-001",
                            full_path="North America / East Region / Store-NYC-001",
                            parent_path_id="group-2",
                            depth=2,
                            device_count=45,
                            security_score=82.1,
                            compliant_count=41,
                            at_risk_count=4,
                            critical_count=0,
                        ),
                        PathNodeResponse(
                            path_id="group-4",
                            path_name="Store-NYC-002",
                            full_path="North America / East Region / Store-NYC-002",
                            parent_path_id="group-2",
                            depth=2,
                            device_count=38,
                            security_score=75.3,
                            compliant_count=32,
                            at_risk_count=5,
                            critical_count=1,
                        ),
                        PathNodeResponse(
                            path_id="group-5",
                            path_name="Warehouse-NYC",
                            full_path="North America / East Region / Warehouse-NYC",
                            parent_path_id="group-2",
                            depth=2,
                            device_count=67,
                            security_score=71.0,
                            compliant_count=54,
                            at_risk_count=11,
                            critical_count=2,
                        ),
                    ],
                ),
                PathNodeResponse(
                    path_id="group-6",
                    path_name="West Region",
                    full_path="North America / West Region",
                    parent_path_id="group-1",
                    depth=1,
                    device_count=150,
                    security_score=68.1,
                    compliant_count=118,
                    at_risk_count=25,
                    critical_count=7,
                    children=[
                        PathNodeResponse(
                            path_id="group-7",
                            path_name="Store-LA-001",
                            full_path="North America / West Region / Store-LA-001",
                            parent_path_id="group-6",
                            depth=2,
                            device_count=52,
                            security_score=71.2,
                            compliant_count=43,
                            at_risk_count=8,
                            critical_count=1,
                        ),
                        PathNodeResponse(
                            path_id="group-8",
                            path_name="Distribution-LA",
                            full_path="North America / West Region / Distribution-LA",
                            parent_path_id="group-6",
                            depth=2,
                            device_count=98,
                            security_score=62.8,
                            compliant_count=71,
                            at_risk_count=21,
                            critical_count=6,
                        ),
                    ],
                ),
            ],
        ),
        PathNodeResponse(
            path_id="group-10",
            path_name="Europe",
            full_path="Europe",
            depth=0,
            device_count=150,
            security_score=76.8,
            compliant_count=127,
            at_risk_count=17,
            critical_count=6,
            children=[
                PathNodeResponse(
                    path_id="group-11",
                    path_name="UK",
                    full_path="Europe / UK",
                    parent_path_id="group-10",
                    depth=1,
                    device_count=85,
                    security_score=79.2,
                    compliant_count=74,
                    at_risk_count=9,
                    critical_count=2,
                ),
                PathNodeResponse(
                    path_id="group-12",
                    path_name="Germany",
                    full_path="Europe / Germany",
                    parent_path_id="group-10",
                    depth=1,
                    device_count=65,
                    security_score=73.5,
                    compliant_count=53,
                    at_risk_count=8,
                    critical_count=4,
                ),
            ],
        ),
    ]

    return PathHierarchyResponse(
        tenant_id=tenant_id,
        hierarchy=hierarchy,
        total_paths=12,
        total_devices=450,
    )


def _generate_mock_risk_clusters(tenant_id: str) -> RiskClustersResponse:
    """Generate mock risk clusters."""
    devices = _generate_mock_devices(100)

    clusters = [
        RiskClusterResponse(
            cluster_id="cluster-rooted",
            cluster_name="Rooted Devices (8 devices)",
            violation_type="rooted",
            severity="critical",
            device_count=8,
            avg_security_score=35.2,
            affected_paths=[
                "North America / East Region / Warehouse-NYC",
                "North America / West Region / Distribution-LA",
            ],
            first_detected=datetime.now(UTC) - timedelta(days=7),
            sample_devices=[d for d in devices if d.is_rooted][:5],
            recommendation="Investigate for potential compromise and consider device wipe",
        ),
        RiskClusterResponse(
            cluster_id="cluster-unencrypted",
            cluster_name="Unencrypted Storage (12 devices)",
            violation_type="unencrypted",
            severity="critical",
            device_count=12,
            avg_security_score=52.1,
            affected_paths=[
                "North America / West Region / Distribution-LA",
                "Europe / Germany / Store-Berlin-001",
            ],
            first_detected=datetime.now(UTC) - timedelta(days=14),
            sample_devices=[d for d in devices if not d.is_encrypted][:5],
            recommendation="Enable encryption policy and verify compliance",
        ),
        RiskClusterResponse(
            cluster_id="cluster-patches",
            cluster_name="Outdated Patches >60 days (45 devices)",
            violation_type="outdated_patch",
            severity="high",
            device_count=45,
            avg_security_score=62.1,
            affected_paths=[
                "North America / East Region / Store-NYC-001",
                "North America / West Region / Store-LA-001",
                "Europe / UK / Warehouse-Manchester",
            ],
            first_detected=datetime.now(UTC) - timedelta(days=30),
            sample_devices=[d for d in devices if d.patch_age_days > 60][:5],
            recommendation="Schedule security patch deployment across affected locations",
        ),
        RiskClusterResponse(
            cluster_id="cluster-usb",
            cluster_name="USB Debugging Enabled (23 devices)",
            violation_type="usb_debugging",
            severity="medium",
            device_count=23,
            avg_security_score=78.5,
            affected_paths=[
                "North America / East Region / Store-NYC-001",
                "North America / West Region / Store-LA-001",
            ],
            first_detected=datetime.now(UTC) - timedelta(days=5),
            sample_devices=[d for d in devices if d.usb_debugging][:5],
            recommendation="Push policy to disable USB debugging on production devices",
        ),
        RiskClusterResponse(
            cluster_id="cluster-devmode",
            cluster_name="Developer Mode Enabled (31 devices)",
            violation_type="developer_mode",
            severity="medium",
            device_count=31,
            avg_security_score=81.2,
            affected_paths=[
                "North America / East Region / Warehouse-NYC",
                "Europe / Germany / Store-Berlin-001",
            ],
            first_detected=datetime.now(UTC) - timedelta(days=10),
            sample_devices=[d for d in devices if d.developer_mode][:5],
            recommendation="Disable developer mode on production devices",
        ),
    ]

    total_affected = sum(c.device_count for c in clusters)
    return RiskClustersResponse(
        tenant_id=tenant_id,
        clusters=clusters,
        total_devices_affected=total_affected,
        coverage_percent=round(total_affected / 450 * 100, 1),
    )


def _generate_mock_path_comparison(
    tenant_id: str,
    paths: list[str],
) -> PathComparisonResponse:
    """Generate mock path comparison data."""
    path_data = {
        "North America / East Region / Store-NYC-001": (82.1, 45, 91.1, 0, 1, 3),
        "North America / East Region / Store-NYC-002": (75.3, 38, 84.2, 1, 2, 5),
        "North America / East Region / Warehouse-NYC": (71.0, 67, 80.6, 2, 3, 8),
        "North America / West Region / Store-LA-001": (71.2, 52, 82.7, 1, 2, 6),
        "North America / West Region / Distribution-LA": (62.8, 98, 72.4, 4, 5, 15),
        "Europe / UK / Store-London-001": (79.2, 45, 88.9, 0, 1, 4),
        "Europe / UK / Warehouse-Manchester": (76.5, 40, 85.0, 1, 1, 5),
        "Europe / Germany / Store-Berlin-001": (73.5, 65, 81.5, 2, 2, 7),
    }

    fleet_avg = 72.5
    comparisons = []

    for path in paths:
        if path in path_data:
            score, count, compliant, rooted, unencrypted, outdated = path_data[path]
            path_name = path.split(" / ")[-1]
            comparisons.append(
                PathComparisonItemResponse(
                    path=path,
                    path_name=path_name,
                    security_score=score,
                    device_count=count,
                    compliant_pct=compliant,
                    rooted_count=rooted,
                    unencrypted_count=unencrypted,
                    outdated_patch_count=outdated,
                    delta_from_fleet=round(score - fleet_avg, 1),
                )
            )

    comparisons.sort(key=lambda c: c.security_score, reverse=True)

    insights = []
    if comparisons:
        best = comparisons[0]
        worst = comparisons[-1]
        if best.delta_from_fleet > 5:
            insights.append(
                f"{best.path_name} leads with {best.security_score} score "
                f"(+{best.delta_from_fleet} above fleet average)"
            )
        if worst.delta_from_fleet < -5:
            insights.append(
                f"{worst.path_name} needs attention with {worst.security_score} score "
                f"({worst.delta_from_fleet} below fleet average)"
            )

    return PathComparisonResponse(
        tenant_id=tenant_id,
        paths=comparisons,
        fleet_average_score=fleet_avg,
        best_path=comparisons[0].path if comparisons else None,
        worst_path=comparisons[-1].path if comparisons else None,
        insights=insights,
    )


def _generate_mock_temporal_clusters(
    tenant_id: str,
    window_hours: int,
) -> TemporalClustersResponse:
    """Generate mock temporal correlation clusters."""
    devices = _generate_mock_devices(50)

    clusters = [
        TemporalClusterResponse(
            cluster_id="temporal-1",
            cluster_name="Correlated Patch Issues (12 devices)",
            issue_appeared_at=datetime.now(UTC) - timedelta(hours=48),
            device_count=12,
            common_violations=["Security patch 75 days old", "Security patch 82 days old"],
            affected_paths=[
                "North America / West Region / Distribution-LA",
                "North America / West Region / Store-LA-001",
            ],
            correlation_insight=(
                "12 devices across 2 West Region locations developed patch issues simultaneously - "
                "likely a failed batch update deployment"
            ),
            sample_devices=[d for d in devices if d.patch_age_days > 60][:5],
        ),
        TemporalClusterResponse(
            cluster_id="temporal-2",
            cluster_name="USB Debugging Wave (8 devices)",
            issue_appeared_at=datetime.now(UTC) - timedelta(hours=24),
            device_count=8,
            common_violations=["USB debugging enabled"],
            affected_paths=["North America / East Region / Store-NYC-001"],
            correlation_insight=(
                "8 devices at Store-NYC-001 had USB debugging enabled within 24 hours - "
                "investigate if related to maintenance activity"
            ),
            sample_devices=[d for d in devices if d.usb_debugging][:5],
        ),
    ]

    return TemporalClustersResponse(
        tenant_id=tenant_id,
        clusters=clusters,
        window_hours=window_hours,
    )


# ============================================================================
# API Endpoints
# ============================================================================


@router.get("/summary", response_model=SecuritySummaryResponse)
async def get_security_summary(
    tenant_id: str = Depends(get_tenant_id),
    mock_mode: bool = Depends(get_mock_mode),
):
    """Get fleet security posture summary."""
    if mock_mode:
        return _generate_mock_summary(tenant_id)

    # Real implementation would load from:
    # - DevInfo security columns (IsEncrypted, IsRooted, HasPasscode, etc.)
    # - SecurityFeatureBuilder for composite scores
    logger.info(
        "Security summary requested for tenant %s (live mode, no data available)", tenant_id
    )
    # Return empty summary in live mode when no real data is available
    return SecuritySummaryResponse(
        tenant_id=tenant_id,
        fleet_security_score=100.0,
        total_devices=0,
        compliant_devices=0,
        at_risk_devices=0,
        critical_risk_devices=0,
        encrypted_devices=0,
        rooted_devices=0,
        outdated_patch_devices=0,
        usb_debugging_enabled=0,
        developer_mode_enabled=0,
        no_passcode_devices=0,
        recommendations=["No security data available - connect to data source"],
        generated_at=datetime.now(UTC),
    )


@router.get("/devices", response_model=DeviceListResponse)
async def get_device_security(
    tenant_id: str = Depends(get_tenant_id),
    mock_mode: bool = Depends(get_mock_mode),
    limit: int = Query(default=50, le=500),
    risk_level: str | None = Query(default=None, description="Filter by risk level"),
):
    """Get per-device security status."""
    if mock_mode:
        devices = _generate_mock_devices(limit)
        if risk_level:
            devices = [d for d in devices if d.risk_level == risk_level]
        return DeviceListResponse(devices=devices, total_count=len(devices))

    # Real implementation would query DevInfo with security columns, compute scores
    logger.info("Device security requested for tenant %s (live mode, no data available)", tenant_id)
    return DeviceListResponse(devices=[], total_count=0)


@router.get("/compliance", response_model=ComplianceListResponse)
async def get_compliance_breakdown(
    tenant_id: str = Depends(get_tenant_id),
    mock_mode: bool = Depends(get_mock_mode),
):
    """Get compliance breakdown by category."""
    if mock_mode:
        return ComplianceListResponse(categories=_generate_mock_compliance())

    # Real implementation would aggregate DevInfo security columns by category
    logger.info(
        "Compliance breakdown requested for tenant %s (live mode, no data available)", tenant_id
    )
    return ComplianceListResponse(categories=[])


@router.get("/trends", response_model=TrendListResponse)
async def get_security_trends(
    tenant_id: str = Depends(get_tenant_id),
    mock_mode: bool = Depends(get_mock_mode),
    period_days: int = Query(default=30, ge=7, le=90),
):
    """Get security score trends over time."""
    if mock_mode:
        return TrendListResponse(trends=_generate_mock_trends(period_days), period_days=period_days)

    # Real implementation would query historical security snapshots
    logger.info(
        "Security trends requested for tenant %s, period %d days (live mode, no data available)",
        tenant_id,
        period_days,
    )
    return TrendListResponse(trends=[], period_days=period_days)


@router.get("/at-risk")
async def get_at_risk_devices(
    tenant_id: str = Depends(get_tenant_id),
    mock_mode: bool = Depends(get_mock_mode),
    limit: int = Query(default=20, le=100),
):
    """Get devices with highest security risk."""
    if mock_mode:
        devices = _generate_mock_devices(limit)
        at_risk = [d for d in devices if d.risk_level in ("high", "critical")]
        return {
            "at_risk_devices": at_risk[:limit],
            "total_at_risk": len(at_risk),
            "critical_count": len([d for d in at_risk if d.risk_level == "critical"]),
        }

    # TODO: Query devices sorted by security score ascending
    logger.info("At-risk devices requested for tenant %s", tenant_id)
    return {"at_risk_devices": [], "total_at_risk": 0, "critical_count": 0}


# ============================================================================
# PATH-Based Endpoints
# ============================================================================


@router.get("/paths", response_model=PathHierarchyResponse)
async def get_security_path_hierarchy(
    tenant_id: str = Depends(get_tenant_id),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Get PATH hierarchy with security metrics.

    Returns a tree structure of device groups with aggregated security scores,
    compliant/at-risk counts for each node.
    """
    if mock_mode:
        return _generate_mock_path_hierarchy(tenant_id)

    # Try to load real hierarchy from database
    try:
        hierarchy = get_path_hierarchy()
        if hierarchy:
            # TODO: Enrich with real security metrics
            return PathHierarchyResponse(
                tenant_id=tenant_id,
                hierarchy=[PathNodeResponse(**node) for node in hierarchy],
                total_paths=len(hierarchy),
                total_devices=sum(n.get("device_count", 0) for n in hierarchy),
            )
    except Exception as e:
        logger.warning("Could not load path hierarchy: %s", e)

    # Return empty hierarchy in live mode
    return PathHierarchyResponse(
        tenant_id=tenant_id,
        hierarchy=[],
        total_paths=0,
        total_devices=0,
    )


@router.get("/by-path", response_model=PathSecurityResponse)
async def get_security_by_path(
    tenant_id: str = Depends(get_tenant_id),
    mock_mode: bool = Depends(get_mock_mode),
    path: str | None = Query(default=None, description="Filter to specific path"),
    depth: int = Query(default=2, ge=1, le=5, description="Hierarchy depth to aggregate"),
):
    """
    Get security summaries grouped by PATH.

    Aggregates security metrics at each PATH level for comparison.
    """
    if mock_mode:
        hierarchy = _generate_mock_path_hierarchy(tenant_id)
        # Flatten hierarchy to summaries
        summaries = []

        def flatten(nodes: list[PathNodeResponse], current_depth: int = 0):
            for node in nodes:
                if current_depth <= depth:
                    summaries.append(
                        PathSecuritySummaryResponse(
                            path=node.full_path,
                            path_name=node.path_name,
                            device_count=node.device_count,
                            security_score=node.security_score,
                            compliant_count=node.compliant_count,
                            at_risk_count=node.at_risk_count,
                            critical_count=node.critical_count,
                            rooted_count=0,  # Not tracked at node level in mock
                            unencrypted_count=0,
                            no_passcode_count=0,
                            outdated_patch_count=0,
                            usb_debugging_count=0,
                            developer_mode_count=0,
                        )
                    )
                    if node.children:
                        flatten(node.children, current_depth + 1)

        flatten(hierarchy.hierarchy)

        if path:
            summaries = [s for s in summaries if path in s.path]

        return PathSecurityResponse(
            tenant_id=tenant_id,
            summaries=summaries,
            selected_path=path,
            depth=depth,
        )

    # TODO: Implement real PATH grouping
    logger.info("Security by path requested for tenant %s", tenant_id)
    return PathSecurityResponse(
        tenant_id=tenant_id,
        summaries=[],
        selected_path=path,
        depth=depth,
    )


@router.get("/risk-clusters", response_model=RiskClustersResponse)
async def get_security_risk_clusters(
    tenant_id: str = Depends(get_tenant_id),
    mock_mode: bool = Depends(get_mock_mode),
    violation_types: str | None = Query(
        default=None, description="Comma-separated violation types to filter"
    ),
):
    """
    Get devices grouped by security violation type (risk clusters).

    Groups devices with similar security issues for targeted remediation.
    """
    if mock_mode:
        result = _generate_mock_risk_clusters(tenant_id)
        if violation_types:
            types = [t.strip() for t in violation_types.split(",")]
            result.clusters = [c for c in result.clusters if c.violation_type in types]
        return result

    # TODO: Use SecurityGrouper to compute real clusters
    logger.info("Risk clusters requested for tenant %s", tenant_id)
    return RiskClustersResponse(
        tenant_id=tenant_id,
        clusters=[],
        total_devices_affected=0,
        coverage_percent=0.0,
    )


@router.get("/path-comparison", response_model=PathComparisonResponse)
async def compare_security_by_path(
    tenant_id: str = Depends(get_tenant_id),
    mock_mode: bool = Depends(get_mock_mode),
    paths: str = Query(..., description="Comma-separated paths to compare"),
):
    """
    Compare security posture across multiple PATHs.

    Returns side-by-side comparison with insights about best/worst performers.
    """
    path_list = [p.strip() for p in paths.split(",") if p.strip()]

    if len(path_list) < 2:
        return PathComparisonResponse(
            tenant_id=tenant_id,
            paths=[],
            fleet_average_score=0,
            insights=["Please provide at least 2 paths to compare"],
        )

    if mock_mode:
        return _generate_mock_path_comparison(tenant_id, path_list)

    # TODO: Use SecurityGrouper to compute real comparison
    logger.info("Path comparison requested for tenant %s: %s", tenant_id, path_list)
    return PathComparisonResponse(
        tenant_id=tenant_id,
        paths=[],
        fleet_average_score=0,
        insights=[],
    )


@router.get("/temporal-clusters", response_model=TemporalClustersResponse)
async def get_temporal_security_clusters(
    tenant_id: str = Depends(get_tenant_id),
    mock_mode: bool = Depends(get_mock_mode),
    window_hours: int = Query(
        default=72, ge=24, le=168, description="Time window in hours for correlation"
    ),
):
    """
    Find devices with correlated security issues appearing around the same time.

    Identifies patterns that may indicate coordinated attacks, batch failures,
    or systematic issues.
    """
    if mock_mode:
        return _generate_mock_temporal_clusters(tenant_id, window_hours)

    # TODO: Use SecurityGrouper to find real temporal correlations
    logger.info(
        "Temporal clusters requested for tenant %s, window %d hours", tenant_id, window_hours
    )
    return TemporalClustersResponse(
        tenant_id=tenant_id,
        clusters=[],
        window_hours=window_hours,
    )
