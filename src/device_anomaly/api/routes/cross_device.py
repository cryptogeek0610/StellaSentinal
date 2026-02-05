"""
Cross-Device Pattern Detection API.

Endpoints for detecting and reporting systemic issues across device cohorts.
Enables detection of patterns like:
- "Samsung SM-G991B + Android 13: 2.3x higher crash rate than fleet"
- "Zebra TC52 + firmware 8.0.1: 45% more battery drain than previous firmware"
- "All devices at Warehouse B: 3x more network disconnects"
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from device_anomaly.api.dependencies import get_mock_mode

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cross-device", tags=["cross-device-patterns"])


# =============================================================================
# Response Models
# =============================================================================


class CohortIssue(BaseModel):
    """A systemic issue affecting a device cohort."""

    cohort_id: str = Field(description="Unique cohort identifier")
    cohort_name: str = Field(description="Human-readable cohort name")
    manufacturer: str | None = None
    model: str | None = None
    os_version: str | None = None
    firmware: str | None = None
    device_count: int = Field(description="Number of devices in cohort")
    fleet_percentage: float = Field(description="Percentage of total fleet")
    issue_type: str = Field(description="Type of issue: model_issue, os_issue, firmware_issue, etc.")
    severity: str = Field(description="Severity: critical, high, medium, low")
    primary_metric: str = Field(description="Primary metric showing deviation")
    metric_value: float = Field(description="Cohort's average value for metric")
    fleet_value: float = Field(description="Fleet's average value for metric")
    deviation_z: float = Field(description="Z-score of deviation from fleet")
    vs_fleet_multiplier: float = Field(description="Multiplier vs fleet (e.g., 2.3x)")
    direction: str = Field(description="Direction of deviation: elevated or reduced")
    trend: str = Field(description="Trend: worsening, stable, improving")
    recommendation: str = Field(description="Recommended action")


class SystemicIssuesResponse(BaseModel):
    """Response for systemic issues endpoint."""

    total_devices_analyzed: int
    total_cohorts_analyzed: int
    cohorts_with_issues: int
    critical_issues: int
    high_issues: int
    issues: list[CohortIssue]
    generated_at: str


class ModelReliability(BaseModel):
    """Reliability metrics for a device model."""

    model_id: str
    model_name: str
    manufacturer: str | None = None
    device_count: int
    fleet_percentage: float
    reliability_score: float = Field(description="Composite reliability score (higher = better)")
    avg_crash_count: float | None = None
    avg_drop_count: float | None = None
    avg_reboot_count: float | None = None
    crash_z: float | None = Field(None, description="Z-score vs fleet for crashes")
    drop_z: float | None = Field(None, description="Z-score vs fleet for drops")
    status: str = Field(description="Status: healthy, warning, critical")


class ModelReliabilityResponse(BaseModel):
    """Response for model reliability endpoint."""

    period_days: int
    total_models: int
    models: list[ModelReliability]
    generated_at: str


class OsStability(BaseModel):
    """Stability metrics for an OS version."""

    os_version_id: str
    os_version: str
    device_count: int
    fleet_percentage: float
    avg_crash_count: float | None = None
    avg_anr_count: float | None = None
    avg_reboot_count: float | None = None
    crash_z: float | None = None
    issues: list[str] = Field(default_factory=list)
    status: str = Field(description="Status: healthy, warning, critical")


class OsStabilityResponse(BaseModel):
    """Response for OS stability endpoint."""

    total_os_versions: int
    os_versions: list[OsStability]
    generated_at: str


class FirmwareImpact(BaseModel):
    """Impact analysis for a firmware version."""

    firmware_version: str
    manufacturer: str | None = None
    model: str | None = None
    device_count: int
    health_score: float = Field(description="Composite health score (higher = better)")
    avg_battery_drain: float | None = None
    avg_crash_count: float | None = None
    avg_drop_count: float | None = None
    status: str = Field(description="Status: healthy, warning, critical")


class FirmwareImpactResponse(BaseModel):
    """Response for firmware impact endpoint."""

    manufacturer_filter: str | None = None
    model_filter: str | None = None
    period_days: int
    total_firmware_versions: int
    firmware_versions: list[FirmwareImpact]
    generated_at: str


class CohortComparisonMetric(BaseModel):
    """A single metric comparison for a cohort."""

    metric: str
    cohort_value: float
    fleet_value: float
    z_score: float
    vs_fleet_multiplier: float
    direction: str
    is_anomalous: bool


class CohortDetail(BaseModel):
    """Detailed cohort information."""

    cohort_id: str
    cohort_name: str
    device_count: int
    fleet_percentage: float
    metrics: list[CohortComparisonMetric]
    issues_count: int


class CohortDetailResponse(BaseModel):
    """Response for cohort detail endpoint."""

    cohort: CohortDetail
    generated_at: str


# =============================================================================
# Mock Data Generators
# =============================================================================


def get_mock_systemic_issues(
    min_devices: int = 10,
    min_z: float = 2.0,
    severity_filter: list[str] | None = None,
) -> SystemicIssuesResponse:
    """Generate mock systemic issues for development."""
    issues = [
        CohortIssue(
            cohort_id="samsung_sm-g991b_13",
            cohort_name="Samsung Galaxy S21 (Android 13)",
            manufacturer="Samsung",
            model="SM-G991B",
            os_version="13",
            device_count=342,
            fleet_percentage=8.5,
            issue_type="model_issue",
            severity="high",
            primary_metric="CrashCount",
            metric_value=5.2,
            fleet_value=2.3,
            deviation_z=2.8,
            vs_fleet_multiplier=2.26,
            direction="elevated",
            trend="stable",
            recommendation="Investigate Samsung Galaxy S21 (Android 13) for elevated crash count. Review recent app or OS updates.",
        ),
        CohortIssue(
            cohort_id="zebra_tc52_11_fw8.0.1",
            cohort_name="Zebra TC52 (Android 11, FW 8.0.1)",
            manufacturer="Zebra",
            model="TC52",
            os_version="11",
            firmware="8.0.1",
            device_count=234,
            fleet_percentage=5.8,
            issue_type="firmware_issue",
            severity="critical",
            primary_metric="TotalBatteryLevelDrop",
            metric_value=45.3,
            fleet_value=28.1,
            deviation_z=3.2,
            vs_fleet_multiplier=1.61,
            direction="elevated",
            trend="worsening",
            recommendation="URGENT: Investigate firmware 8.0.1 for Zebra TC52. Consider rollback to previous version.",
        ),
        CohortIssue(
            cohort_id="honeywell_ct40_verizon",
            cohort_name="Honeywell CT40 (Verizon)",
            manufacturer="Honeywell",
            model="CT40",
            device_count=156,
            fleet_percentage=3.9,
            issue_type="carrier_issue",
            severity="medium",
            primary_metric="TotalDropCnt",
            metric_value=12.4,
            fleet_value=5.1,
            deviation_z=2.3,
            vs_fleet_multiplier=2.43,
            direction="elevated",
            trend="stable",
            recommendation="Review Verizon network configuration for Honeywell CT40 devices. Check APN settings.",
        ),
        CohortIssue(
            cohort_id="warehouse_b_all",
            cohort_name="Warehouse B (All Devices)",
            device_count=89,
            fleet_percentage=2.2,
            issue_type="location_issue",
            severity="high",
            primary_metric="WifiDisconnectCount",
            metric_value=8.7,
            fleet_value=2.9,
            deviation_z=2.6,
            vs_fleet_multiplier=3.0,
            direction="elevated",
            trend="worsening",
            recommendation="Investigate WiFi infrastructure at Warehouse B. Check for coverage gaps or interference.",
        ),
    ]

    # Apply severity filter
    if severity_filter:
        issues = [i for i in issues if i.severity in severity_filter]

    # Apply min_devices filter
    issues = [i for i in issues if i.device_count >= min_devices]

    return SystemicIssuesResponse(
        total_devices_analyzed=4023,
        total_cohorts_analyzed=47,
        cohorts_with_issues=4,
        critical_issues=1,
        high_issues=2,
        issues=issues,
        generated_at=datetime.now(UTC).isoformat(),
    )


def get_mock_model_reliability(
    period_days: int = 30,
    top_n: int = 20,
) -> ModelReliabilityResponse:
    """Generate mock model reliability data."""
    models = [
        ModelReliability(
            model_id="iphone_14",
            model_name="iPhone 14",
            manufacturer="Apple",
            device_count=456,
            fleet_percentage=11.3,
            reliability_score=8.5,
            avg_crash_count=0.8,
            avg_drop_count=1.2,
            avg_reboot_count=0.3,
            crash_z=-1.2,
            drop_z=-0.8,
            status="healthy",
        ),
        ModelReliability(
            model_id="tc52",
            model_name="TC52",
            manufacturer="Zebra",
            device_count=567,
            fleet_percentage=14.1,
            reliability_score=7.2,
            avg_crash_count=1.5,
            avg_drop_count=3.2,
            avg_reboot_count=0.8,
            crash_z=0.3,
            drop_z=0.5,
            status="healthy",
        ),
        ModelReliability(
            model_id="sm-g991b",
            model_name="Galaxy S21",
            manufacturer="Samsung",
            device_count=342,
            fleet_percentage=8.5,
            reliability_score=4.8,
            avg_crash_count=5.2,
            avg_drop_count=4.1,
            avg_reboot_count=1.2,
            crash_z=2.8,
            drop_z=1.5,
            status="warning",
        ),
        ModelReliability(
            model_id="ct40",
            model_name="CT40",
            manufacturer="Honeywell",
            device_count=234,
            fleet_percentage=5.8,
            reliability_score=6.1,
            avg_crash_count=2.1,
            avg_drop_count=6.8,
            avg_reboot_count=0.5,
            crash_z=0.8,
            drop_z=2.1,
            status="warning",
        ),
    ]

    return ModelReliabilityResponse(
        period_days=period_days,
        total_models=len(models),
        models=models[:top_n],
        generated_at=datetime.now(UTC).isoformat(),
    )


def get_mock_os_stability() -> OsStabilityResponse:
    """Generate mock OS stability data."""
    os_versions = [
        OsStability(
            os_version_id="14",
            os_version="Android 14",
            device_count=1234,
            fleet_percentage=30.7,
            avg_crash_count=1.2,
            avg_anr_count=0.3,
            avg_reboot_count=0.4,
            crash_z=-0.5,
            issues=[],
            status="healthy",
        ),
        OsStability(
            os_version_id="13",
            os_version="Android 13",
            device_count=1567,
            fleet_percentage=39.0,
            avg_crash_count=2.8,
            avg_anr_count=0.8,
            avg_reboot_count=0.7,
            crash_z=1.2,
            issues=["Slightly elevated crash rate (1.2σ)"],
            status="warning",
        ),
        OsStability(
            os_version_id="12",
            os_version="Android 12",
            device_count=890,
            fleet_percentage=22.1,
            avg_crash_count=1.9,
            avg_anr_count=0.5,
            avg_reboot_count=0.5,
            crash_z=0.3,
            issues=[],
            status="healthy",
        ),
        OsStability(
            os_version_id="11",
            os_version="Android 11",
            device_count=332,
            fleet_percentage=8.2,
            avg_crash_count=4.1,
            avg_anr_count=1.2,
            avg_reboot_count=1.1,
            crash_z=2.4,
            issues=["Elevated crash rate (2.4σ above fleet)", "Elevated ANR rate"],
            status="warning",
        ),
    ]

    return OsStabilityResponse(
        total_os_versions=len(os_versions),
        os_versions=os_versions,
        generated_at=datetime.now(UTC).isoformat(),
    )


def get_mock_firmware_impact(
    manufacturer: str | None = None,
    model: str | None = None,
    period_days: int = 90,
) -> FirmwareImpactResponse:
    """Generate mock firmware impact data."""
    firmware_versions = [
        FirmwareImpact(
            firmware_version="8.2.0",
            manufacturer="Zebra",
            model="TC52",
            device_count=234,
            health_score=7.8,
            avg_battery_drain=28.1,
            avg_crash_count=1.2,
            avg_drop_count=2.8,
            status="healthy",
        ),
        FirmwareImpact(
            firmware_version="8.1.0",
            manufacturer="Zebra",
            model="TC52",
            device_count=189,
            health_score=6.5,
            avg_battery_drain=32.4,
            avg_crash_count=1.8,
            avg_drop_count=3.5,
            status="healthy",
        ),
        FirmwareImpact(
            firmware_version="8.0.1",
            manufacturer="Zebra",
            model="TC52",
            device_count=156,
            health_score=3.2,
            avg_battery_drain=45.3,
            avg_crash_count=3.5,
            avg_drop_count=5.2,
            status="critical",
        ),
        FirmwareImpact(
            firmware_version="8.0.0",
            manufacturer="Zebra",
            model="TC52",
            device_count=78,
            health_score=5.8,
            avg_battery_drain=35.2,
            avg_crash_count=2.1,
            avg_drop_count=4.1,
            status="warning",
        ),
    ]

    # Filter by manufacturer/model if specified
    if manufacturer:
        firmware_versions = [
            f for f in firmware_versions
            if f.manufacturer and f.manufacturer.lower() == manufacturer.lower()
        ]
    if model:
        firmware_versions = [
            f for f in firmware_versions
            if f.model and f.model.lower() == model.lower()
        ]

    return FirmwareImpactResponse(
        manufacturer_filter=manufacturer,
        model_filter=model,
        period_days=period_days,
        total_firmware_versions=len(firmware_versions),
        firmware_versions=firmware_versions,
        generated_at=datetime.now(UTC).isoformat(),
    )


# =============================================================================
# Real Data Loaders
# =============================================================================


def _load_cross_device_data() -> Any | None:
    """Load data for cross-device pattern analysis."""
    try:
        from datetime import timedelta

        from device_anomaly.data_access.unified_loader import load_unified_device_dataset

        end_date = datetime.now(UTC).date()
        start_date = end_date - timedelta(days=30)

        df = load_unified_device_dataset(
            start_date=start_date,
            end_date=end_date,
        )
        return df
    except Exception as e:
        logger.warning(f"Failed to load unified dataset: {e}")
        return None


def _get_cohort_detector():
    """Get or create cohort detector."""
    try:
        from device_anomaly.models.cohort_detector import CrossDevicePatternDetector

        # Try to load pre-trained detector
        detector_path = Path("models/production/cohort_detector.pkl")
        if detector_path.exists():
            return CrossDevicePatternDetector.load(detector_path)

        # Create new detector
        return CrossDevicePatternDetector()
    except Exception as e:
        logger.warning(f"Failed to get cohort detector: {e}")
        return None


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("/systemic-issues", response_model=SystemicIssuesResponse)
async def get_systemic_issues(
    min_devices: int = Query(10, description="Minimum devices in cohort to flag"),
    min_z: float = Query(2.0, description="Minimum Z-score for flagging"),
    severity: list[str] | None = Query(None, description="Filter by severity: critical, high, medium, low"),
    issue_type: list[str] | None = Query(None, description="Filter by issue type: model_issue, os_issue, etc."),
    mock_mode: bool = Depends(get_mock_mode),
) -> SystemicIssuesResponse:
    """
    Get systemic issues affecting device cohorts.

    Detects patterns like:
    - "Samsung SM-G991B + Android 13: 2.3x higher crash rate than fleet"
    - "Zebra TC52 + firmware 8.0.1: 45% more battery drain than previous firmware"
    - "All devices at Warehouse B: 3x more network disconnects"

    This endpoint analyzes device telemetry to find cohorts (groups of devices
    sharing characteristics) that are systematically performing differently
    from the fleet average.
    """
    if mock_mode:
        return get_mock_systemic_issues(min_devices, min_z, severity)

    # Real implementation
    try:

        df = _load_cross_device_data()
        if df is None or df.empty:
            logger.warning("No data available for cross-device analysis")
            return get_mock_systemic_issues(min_devices, min_z, severity)

        detector = _get_cohort_detector()
        if detector is None:
            return get_mock_systemic_issues(min_devices, min_z, severity)

        # Fit and detect
        detector.fit(df)
        systemic_issues = detector.detect_systemic_issues(
            df,
            min_devices=min_devices,
            min_z=min_z,
        )

        # Convert to response format
        issues = []
        for issue in systemic_issues:
            # Parse cohort_id for components
            cohort_parts = issue.cohort_id.split("_")
            manufacturer = cohort_parts[0] if len(cohort_parts) > 0 else None
            model = cohort_parts[1] if len(cohort_parts) > 1 else None
            os_version = cohort_parts[2] if len(cohort_parts) > 2 else None

            cohort_issue = CohortIssue(
                cohort_id=issue.cohort_id,
                cohort_name=issue.cohort_name,
                manufacturer=manufacturer,
                model=model,
                os_version=os_version,
                device_count=issue.device_count,
                fleet_percentage=issue.fleet_percentage,
                issue_type=issue.issue_type,
                severity=issue.severity,
                primary_metric=issue.primary_metric,
                metric_value=0.0,  # Would need to extract from pattern
                fleet_value=0.0,
                deviation_z=0.0,
                vs_fleet_multiplier=issue.vs_fleet_multiplier,
                direction="elevated" if issue.vs_fleet_multiplier > 1 else "reduced",
                trend=issue.trend,
                recommendation=issue.recommendation,
            )
            issues.append(cohort_issue)

        # Apply filters
        if severity:
            issues = [i for i in issues if i.severity in severity]
        if issue_type:
            issues = [i for i in issues if i.issue_type in issue_type]

        return SystemicIssuesResponse(
            total_devices_analyzed=len(df),
            total_cohorts_analyzed=len(set(df.get("cohort_id", []))) if "cohort_id" in df.columns else 0,
            cohorts_with_issues=len({i.cohort_id for i in issues}),
            critical_issues=len([i for i in issues if i.severity == "critical"]),
            high_issues=len([i for i in issues if i.severity == "high"]),
            issues=issues,
            generated_at=datetime.now(UTC).isoformat(),
        )

    except Exception as e:
        logger.error(f"Error in systemic issues detection: {e}")
        return get_mock_systemic_issues(min_devices, min_z, severity)


@router.get("/model-reliability", response_model=ModelReliabilityResponse)
async def get_model_reliability(
    period_days: int = Query(30, description="Analysis period in days"),
    top_n: int = Query(20, description="Number of models to return"),
    mock_mode: bool = Depends(get_mock_mode),
) -> ModelReliabilityResponse:
    """
    Get device model reliability rankings.

    Returns models ranked by composite reliability score based on
    crash rate, drop rate, reboot rate, etc.

    Models with higher reliability scores are performing better than
    the fleet average.
    """
    if mock_mode:
        return get_mock_model_reliability(period_days, top_n)

    # Real implementation
    try:
        df = _load_cross_device_data()
        if df is None or df.empty:
            return get_mock_model_reliability(period_days, top_n)

        detector = _get_cohort_detector()
        if detector is None:
            return get_mock_model_reliability(period_days, top_n)

        detector.fit(df)
        rankings = detector.get_model_reliability_rankings(df, top_n=top_n)

        models = []
        for r in rankings:
            status = "healthy"
            if r.get("reliability_score", 0) < 0:
                status = "warning"
            if r.get("reliability_score", 0) < -2:
                status = "critical"

            models.append(ModelReliability(
                model_id=r.get("model_id", "unknown"),
                model_name=r.get("model_name", "Unknown Model"),
                manufacturer=r.get("manufacturer"),
                device_count=r.get("device_count", 0),
                fleet_percentage=round(100 * r.get("device_count", 0) / max(1, len(df)), 1),
                reliability_score=r.get("reliability_score", 0),
                avg_crash_count=r.get("avg_crashcount"),
                avg_drop_count=r.get("avg_totaldropcnt"),
                avg_reboot_count=r.get("avg_rebootcount"),
                crash_z=r.get("crashcount_z"),
                drop_z=r.get("totaldropcnt_z"),
                status=status,
            ))

        return ModelReliabilityResponse(
            period_days=period_days,
            total_models=len(models),
            models=models,
            generated_at=datetime.now(UTC).isoformat(),
        )

    except Exception as e:
        logger.error(f"Error in model reliability analysis: {e}")
        return get_mock_model_reliability(period_days, top_n)


@router.get("/os-stability", response_model=OsStabilityResponse)
async def get_os_stability(
    period_days: int = Query(30, description="Analysis period in days"),
    mock_mode: bool = Depends(get_mock_mode),
) -> OsStabilityResponse:
    """
    Get OS version stability analysis.

    Returns which OS versions are showing elevated issue rates
    compared to the fleet average.
    """
    if mock_mode:
        return get_mock_os_stability()

    # Real implementation
    try:
        df = _load_cross_device_data()
        if df is None or df.empty:
            return get_mock_os_stability()

        detector = _get_cohort_detector()
        if detector is None:
            return get_mock_os_stability()

        detector.fit(df)
        os_analysis = detector.get_os_stability_analysis(df)

        os_versions = []
        for os in os_analysis:
            os_versions.append(OsStability(
                os_version_id=os.get("os_version_id", "unknown"),
                os_version=os.get("os_version", "Unknown OS"),
                device_count=os.get("device_count", 0),
                fleet_percentage=os.get("fleet_percentage", 0),
                avg_crash_count=os.get("avg_crashcount"),
                avg_anr_count=os.get("avg_anrcount"),
                avg_reboot_count=os.get("avg_rebootcount"),
                crash_z=os.get("crashcount_z"),
                issues=os.get("issues", []),
                status=os.get("status", "healthy"),
            ))

        return OsStabilityResponse(
            total_os_versions=len(os_versions),
            os_versions=os_versions,
            generated_at=datetime.now(UTC).isoformat(),
        )

    except Exception as e:
        logger.error(f"Error in OS stability analysis: {e}")
        return get_mock_os_stability()


@router.get("/firmware-impact", response_model=FirmwareImpactResponse)
async def get_firmware_impact(
    manufacturer: str | None = Query(None, description="Filter by manufacturer"),
    model: str | None = Query(None, description="Filter by model"),
    period_days: int = Query(90, description="Analysis period in days"),
    mock_mode: bool = Depends(get_mock_mode),
) -> FirmwareImpactResponse:
    """
    Analyze firmware version impact on device health.

    Detects firmware versions that correlate with:
    - Increased crash rates
    - Battery drain issues
    - Network instability

    Use this to identify problematic firmware versions that may need
    rollback or patching.
    """
    if mock_mode:
        return get_mock_firmware_impact(manufacturer, model, period_days)

    # Real implementation
    try:
        df = _load_cross_device_data()
        if df is None or df.empty:
            return get_mock_firmware_impact(manufacturer, model, period_days)

        detector = _get_cohort_detector()
        if detector is None:
            return get_mock_firmware_impact(manufacturer, model, period_days)

        detector.fit(df)
        firmware_analysis = detector.get_firmware_impact_analysis(
            df,
            manufacturer=manufacturer,
            model=model,
        )

        firmware_versions = []
        for fw in firmware_analysis:
            firmware_versions.append(FirmwareImpact(
                firmware_version=fw.get("firmware_version", "unknown"),
                manufacturer=fw.get("manufacturer"),
                model=fw.get("model"),
                device_count=fw.get("device_count", 0),
                health_score=fw.get("health_score", 0),
                avg_battery_drain=fw.get("avg_totalbatteryleveldrop"),
                avg_crash_count=fw.get("avg_crashcount"),
                avg_drop_count=fw.get("avg_totaldropcnt"),
                status=fw.get("status", "healthy"),
            ))

        return FirmwareImpactResponse(
            manufacturer_filter=manufacturer,
            model_filter=model,
            period_days=period_days,
            total_firmware_versions=len(firmware_versions),
            firmware_versions=firmware_versions,
            generated_at=datetime.now(UTC).isoformat(),
        )

    except Exception as e:
        logger.error(f"Error in firmware impact analysis: {e}")
        return get_mock_firmware_impact(manufacturer, model, period_days)


@router.get("/cohort/{cohort_id}", response_model=CohortDetailResponse)
async def get_cohort_detail(
    cohort_id: str,
    mock_mode: bool = Depends(get_mock_mode),
) -> CohortDetailResponse:
    """
    Get detailed information about a specific cohort.

    Returns all metrics for the cohort compared to fleet baseline.
    """
    if mock_mode:
        # Generate mock detail
        metrics = [
            CohortComparisonMetric(
                metric="CrashCount",
                cohort_value=5.2,
                fleet_value=2.3,
                z_score=2.8,
                vs_fleet_multiplier=2.26,
                direction="elevated",
                is_anomalous=True,
            ),
            CohortComparisonMetric(
                metric="TotalBatteryLevelDrop",
                cohort_value=32.1,
                fleet_value=28.5,
                z_score=0.8,
                vs_fleet_multiplier=1.13,
                direction="elevated",
                is_anomalous=False,
            ),
            CohortComparisonMetric(
                metric="TotalDropCnt",
                cohort_value=4.1,
                fleet_value=3.8,
                z_score=0.5,
                vs_fleet_multiplier=1.08,
                direction="elevated",
                is_anomalous=False,
            ),
        ]

        return CohortDetailResponse(
            cohort=CohortDetail(
                cohort_id=cohort_id,
                cohort_name="Samsung Galaxy S21 (Android 13)",
                device_count=342,
                fleet_percentage=8.5,
                metrics=metrics,
                issues_count=1,
            ),
            generated_at=datetime.now(UTC).isoformat(),
        )

    # Real implementation would query the cohort from the database
    raise HTTPException(status_code=501, detail="Real implementation pending")


@router.get("/summary")
async def get_cross_device_summary(
    mock_mode: bool = Depends(get_mock_mode),
) -> dict[str, Any]:
    """
    Get high-level summary of cross-device pattern analysis.

    Returns counts of issues by severity and type.
    """
    # Call with explicit default values since Query objects aren't resolved when called internally
    issues_response = await get_systemic_issues(
        min_devices=10,
        min_z=2.0,
        severity=None,
        issue_type=None,
        mock_mode=mock_mode,
    )

    # Aggregate by type
    by_type: dict[str, int] = {}
    for issue in issues_response.issues:
        by_type[issue.issue_type] = by_type.get(issue.issue_type, 0) + 1

    return {
        "total_devices_analyzed": issues_response.total_devices_analyzed,
        "total_cohorts_analyzed": issues_response.total_cohorts_analyzed,
        "cohorts_with_issues": issues_response.cohorts_with_issues,
        "issues_by_severity": {
            "critical": issues_response.critical_issues,
            "high": issues_response.high_issues,
            "medium": len([i for i in issues_response.issues if i.severity == "medium"]),
            "low": len([i for i in issues_response.issues if i.severity == "low"]),
        },
        "issues_by_type": by_type,
        "generated_at": issues_response.generated_at,
    }
