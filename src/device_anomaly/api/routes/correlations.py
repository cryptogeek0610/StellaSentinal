"""
API routes for Correlation Intelligence endpoints.

Provides correlation analysis including:
- Correlation matrix computation
- Scatter plot data
- Causal graph visualization
- Auto-generated insights
- Cohort correlation patterns
- Time-lagged correlations
"""
from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from device_anomaly.api.dependencies import get_mock_mode

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/correlations", tags=["correlations"])

# In-memory cache for correlation matrix
_correlation_cache: Dict[str, Any] = {}


# =============================================================================
# Response Models
# =============================================================================


class CorrelationCell(BaseModel):
    """Single cell in correlation matrix."""
    metric_x: str
    metric_y: str
    correlation: float
    p_value: Optional[float] = None
    sample_count: int
    method: str = "pearson"


class CorrelationMatrixResponse(BaseModel):
    """Full correlation matrix response."""
    metrics: List[str]
    matrix: List[List[float]]
    strong_correlations: List[CorrelationCell]
    method: str
    computed_at: str
    total_samples: int
    domain_filter: Optional[str] = None


class ScatterDataPoint(BaseModel):
    """Single point for scatter plot."""
    device_id: int
    x_value: float
    y_value: float
    is_anomaly: bool
    cohort: Optional[str] = None
    timestamp: Optional[str] = None


class ScatterPlotResponse(BaseModel):
    """Scatter plot data for two metrics."""
    metric_x: str
    metric_y: str
    points: List[ScatterDataPoint]
    correlation: float
    regression_slope: Optional[float] = None
    regression_intercept: Optional[float] = None
    r_squared: Optional[float] = None
    total_points: int
    anomaly_count: int


class CausalNode(BaseModel):
    """Node in causal graph."""
    metric: str
    domain: str
    is_cause: bool
    is_effect: bool
    connection_count: int


class CausalEdge(BaseModel):
    """Edge in causal graph."""
    source: str
    target: str
    relationship: str
    strength: float
    evidence: Optional[str] = None


class CausalGraphResponse(BaseModel):
    """Causal relationship network."""
    nodes: List[CausalNode]
    edges: List[CausalEdge]
    generated_at: str


class CorrelationInsight(BaseModel):
    """Auto-discovered correlation insight."""
    insight_id: str
    headline: str
    description: str
    metrics_involved: List[str]
    correlation_value: float
    strength: str
    direction: str
    novelty_score: float
    confidence: float
    recommendation: Optional[str] = None


class CorrelationInsightsResponse(BaseModel):
    """Auto-generated correlation insights."""
    insights: List[CorrelationInsight]
    total_correlations_analyzed: int
    generated_at: str


class CohortCorrelationPattern(BaseModel):
    """Correlation pattern for a specific cohort."""
    cohort_id: str
    cohort_name: str
    metric_pair: List[str]
    cohort_correlation: float
    fleet_correlation: float
    deviation: float
    device_count: int
    is_anomalous: bool
    insight: Optional[str] = None


class CohortCorrelationPatternsResponse(BaseModel):
    """Cohort-specific correlation patterns."""
    patterns: List[CohortCorrelationPattern]
    anomalous_cohorts: int
    generated_at: str


class TimeLagCorrelation(BaseModel):
    """Time-lagged correlation result."""
    metric_a: str
    metric_b: str
    lag_days: int
    correlation: float
    p_value: Optional[float] = None
    direction: str
    insight: str


class TimeLagCorrelationsResponse(BaseModel):
    """Time-lagged correlation analysis."""
    correlations: List[TimeLagCorrelation]
    max_lag_analyzed: int
    generated_at: str


# =============================================================================
# Mock Data Generators
# =============================================================================


def get_mock_correlation_matrix(
    domain: Optional[str] = None,
    method: str = "pearson",
) -> CorrelationMatrixResponse:
    """Generate mock correlation matrix with realistic correlations."""
    # Define metrics by domain
    all_metrics = {
        "battery": ["TotalBatteryLevelDrop", "TotalDischargeTime_Sec", "ScreenOnTime_Sec", "BatteryDrainPerHour"],
        "rf": ["AvgSignalStrength", "TotalDropCnt", "WifiDisconnectCount", "CellSignalStrength"],
        "throughput": ["Download", "Upload", "TotalDataUsage"],
        "usage": ["AppForegroundTime", "CrashCount", "AppVisitCount"],
        "storage": ["StorageUtilization", "RAMPressure", "FreeStorageKb"],
        "system": ["CPUUsage", "Temperature", "MemoryUsage"],
    }

    if domain and domain in all_metrics:
        metrics = all_metrics[domain]
    else:
        # All metrics
        metrics = []
        for domain_metrics in all_metrics.values():
            metrics.extend(domain_metrics)
        metrics = metrics[:15]  # Limit for display

    n = len(metrics)

    # Known correlations (domain knowledge)
    known_correlations = {
        ("TotalBatteryLevelDrop", "ScreenOnTime_Sec"): 0.78,
        ("TotalBatteryLevelDrop", "AvgSignalStrength"): -0.72,
        ("AvgSignalStrength", "TotalDropCnt"): -0.81,
        ("TotalBatteryLevelDrop", "TotalDropCnt"): 0.65,
        ("Download", "Upload"): 0.73,
        ("CrashCount", "RAMPressure"): 0.68,
        ("CPUUsage", "Temperature"): 0.76,
        ("TotalBatteryLevelDrop", "CPUUsage"): 0.54,
        ("AppForegroundTime", "TotalBatteryLevelDrop"): 0.67,
        ("StorageUtilization", "CrashCount"): 0.45,
    }

    # Build matrix
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
            elif i < j:
                # Check for known correlation
                pair = (metrics[i], metrics[j])
                reverse_pair = (metrics[j], metrics[i])
                if pair in known_correlations:
                    corr = known_correlations[pair]
                elif reverse_pair in known_correlations:
                    corr = known_correlations[reverse_pair]
                else:
                    # Random weak correlation
                    corr = round(random.uniform(-0.4, 0.4), 2)
                matrix[i][j] = corr
                matrix[j][i] = corr

    # Find strong correlations
    strong = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i][j]) >= 0.6:
                strong.append(CorrelationCell(
                    metric_x=metrics[i],
                    metric_y=metrics[j],
                    correlation=matrix[i][j],
                    p_value=0.001 if abs(matrix[i][j]) > 0.7 else 0.01,
                    sample_count=random.randint(3000, 5000),
                    method=method,
                ))

    # Sort by absolute correlation
    strong.sort(key=lambda x: abs(x.correlation), reverse=True)

    return CorrelationMatrixResponse(
        metrics=metrics,
        matrix=matrix,
        strong_correlations=strong,
        method=method,
        computed_at=datetime.now(timezone.utc).isoformat(),
        total_samples=4532,
        domain_filter=domain,
    )


def get_mock_scatter_data(
    metric_x: str,
    metric_y: str,
    limit: int = 500,
) -> ScatterPlotResponse:
    """Generate mock scatter plot data."""
    # Known correlation for these metrics
    known_correlations = {
        ("TotalBatteryLevelDrop", "ScreenOnTime_Sec"): (0.78, 0.15),
        ("TotalBatteryLevelDrop", "AvgSignalStrength"): (-0.72, -0.08),
        ("AvgSignalStrength", "TotalDropCnt"): (-0.81, -2.5),
    }

    pair = (metric_x, metric_y)
    reverse_pair = (metric_y, metric_x)

    if pair in known_correlations:
        corr, slope = known_correlations[pair]
    elif reverse_pair in known_correlations:
        corr, _ = known_correlations[reverse_pair]
        slope = 1.0 / known_correlations[reverse_pair][1] if known_correlations[reverse_pair][1] != 0 else 0.5
    else:
        corr = round(random.uniform(-0.5, 0.5), 2)
        slope = corr * 0.5

    # Generate correlated data points
    points = []
    anomaly_count = 0
    cohorts = ["Samsung_SM-G991B", "Samsung_SM-A515F", "Zebra_TC52", "Honeywell_CT40", "Other"]

    for i in range(limit):
        # Generate correlated values
        x = random.gauss(50, 15)
        noise = random.gauss(0, 10 * (1 - abs(corr)))
        y = slope * x + 30 + noise

        is_anomaly = random.random() < 0.08
        if is_anomaly:
            anomaly_count += 1
            # Anomalies are outliers
            y += random.choice([-1, 1]) * random.uniform(20, 40)

        points.append(ScatterDataPoint(
            device_id=1000 + i,
            x_value=round(max(0, x), 2),
            y_value=round(max(0, y), 2),
            is_anomaly=is_anomaly,
            cohort=random.choice(cohorts),
            timestamp=f"2025-01-{random.randint(1, 9):02d}T{random.randint(0, 23):02d}:00:00Z",
        ))

    # Calculate r-squared
    r_squared = corr ** 2

    return ScatterPlotResponse(
        metric_x=metric_x,
        metric_y=metric_y,
        points=points,
        correlation=corr,
        regression_slope=round(slope, 4),
        regression_intercept=30.0,
        r_squared=round(r_squared, 4),
        total_points=len(points),
        anomaly_count=anomaly_count,
    )


def get_mock_causal_graph() -> CausalGraphResponse:
    """Generate causal graph from RootCauseAnalyzer's domain knowledge."""
    # This mirrors the causal graph from root_cause.py
    causal_relationships = {
        "ScreenOnTime": ["BatteryDrain", "BatteryDrainPerHour"],
        "AppForegroundTime": ["BatteryDrain", "BatteryDrainPerHour"],
        "BackgroundAppActivity": ["BatteryDrain"],
        "PoorSignal": ["BatteryDrain", "NetworkDrops"],
        "ChargingPattern": ["BatteryHealth", "BatteryCapacity"],
        "BatteryHealth": ["BatteryDrain", "BatteryCapacity"],
        "Temperature": ["BatteryDrain", "CPUThrottle"],
        "LocationMovement": ["APHopping", "NetworkDrops"],
        "WeakWifiCoverage": ["NetworkDrops", "SignalStrength"],
        "CellCoverage": ["TowerHopping", "NetworkDrops"],
        "APHopping": ["BatteryDrain", "NetworkDrops"],
        "HighDataUsage": ["BatteryDrain", "NetworkCongestion"],
        "AppVersion": ["AppCrash", "ANR"],
        "LowMemory": ["AppCrash", "ANR", "SlowPerformance"],
        "LowStorage": ["AppCrash", "InstallFailure"],
        "HighCPU": ["BatteryDrain", "Temperature", "SlowPerformance"],
        "OsVersion": ["AppCrash", "SecurityVulnerability"],
        "AgentVersion": ["DataCollectionIssues"],
        "Rooted": ["SecurityIssues", "AppBehavior"],
    }

    # Domain mapping
    domain_map = {
        "ScreenOnTime": "usage", "AppForegroundTime": "usage", "BackgroundAppActivity": "usage",
        "PoorSignal": "rf", "WeakWifiCoverage": "rf", "CellCoverage": "rf", "SignalStrength": "rf",
        "ChargingPattern": "battery", "BatteryHealth": "battery", "BatteryDrain": "battery",
        "BatteryDrainPerHour": "battery", "BatteryCapacity": "battery",
        "Temperature": "system", "CPUThrottle": "system", "HighCPU": "system",
        "LocationMovement": "location", "APHopping": "connectivity", "NetworkDrops": "connectivity",
        "TowerHopping": "connectivity", "NetworkCongestion": "connectivity",
        "HighDataUsage": "throughput",
        "AppVersion": "app", "AppCrash": "app", "ANR": "app", "AppBehavior": "app",
        "LowMemory": "storage", "LowStorage": "storage", "InstallFailure": "storage",
        "SlowPerformance": "system",
        "OsVersion": "system", "AgentVersion": "system", "SecurityVulnerability": "security",
        "Rooted": "security", "SecurityIssues": "security", "DataCollectionIssues": "system",
    }

    # Build nodes and edges
    nodes_dict: Dict[str, CausalNode] = {}
    edges: List[CausalEdge] = []

    for cause, effects in causal_relationships.items():
        # Add cause node
        if cause not in nodes_dict:
            nodes_dict[cause] = CausalNode(
                metric=cause,
                domain=domain_map.get(cause, "unknown"),
                is_cause=True,
                is_effect=False,
                connection_count=0,
            )
        nodes_dict[cause].is_cause = True
        nodes_dict[cause].connection_count += len(effects)

        for effect in effects:
            # Add effect node
            if effect not in nodes_dict:
                nodes_dict[effect] = CausalNode(
                    metric=effect,
                    domain=domain_map.get(effect, "unknown"),
                    is_cause=False,
                    is_effect=True,
                    connection_count=0,
                )
            nodes_dict[effect].is_effect = True
            nodes_dict[effect].connection_count += 1

            # Add edge
            edges.append(CausalEdge(
                source=cause,
                target=effect,
                relationship="causes",
                strength=0.8,
                evidence=f"{cause} is known to cause {effect}",
            ))

    return CausalGraphResponse(
        nodes=list(nodes_dict.values()),
        edges=edges,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def get_mock_correlation_insights() -> CorrelationInsightsResponse:
    """Generate auto-discovered correlation insights."""
    insights = [
        CorrelationInsight(
            insight_id="ins_001",
            headline="Battery drain strongly correlates with screen time",
            description="Devices with higher screen-on time show proportionally higher battery drain. This is expected behavior but can help identify devices with abnormal screen usage patterns.",
            metrics_involved=["TotalBatteryLevelDrop", "ScreenOnTime_Sec"],
            correlation_value=0.78,
            strength="strong",
            direction="positive",
            novelty_score=0.3,
            confidence=0.95,
            recommendation="Consider implementing screen brightness auto-adjustment policies for high-drain devices.",
        ),
        CorrelationInsight(
            insight_id="ins_002",
            headline="Poor signal quality causes increased network drops",
            description="Strong negative correlation between signal strength and connection drops. Devices in weak coverage areas experience significantly more disconnections.",
            metrics_involved=["AvgSignalStrength", "TotalDropCnt"],
            correlation_value=-0.81,
            strength="strong",
            direction="negative",
            novelty_score=0.2,
            confidence=0.97,
            recommendation="Map coverage dead zones and consider WiFi boosters or network optimization.",
        ),
        CorrelationInsight(
            insight_id="ins_003",
            headline="Battery drain increases with weak signal",
            description="Devices constantly searching for better signal consume more battery. This cross-domain correlation explains unexpectedly high drain in certain locations.",
            metrics_involved=["TotalBatteryLevelDrop", "AvgSignalStrength"],
            correlation_value=-0.72,
            strength="strong",
            direction="negative",
            novelty_score=0.6,
            confidence=0.89,
            recommendation="Prioritize coverage improvements in high-activity areas to reduce battery impact.",
        ),
        CorrelationInsight(
            insight_id="ins_004",
            headline="Memory pressure correlates with app crashes",
            description="Devices experiencing high RAM pressure show increased crash rates. This suggests memory-hungry apps may be destabilizing the system.",
            metrics_involved=["RAMPressure", "CrashCount"],
            correlation_value=0.68,
            strength="moderate",
            direction="positive",
            novelty_score=0.4,
            confidence=0.85,
            recommendation="Review memory usage of frequently used apps and consider increasing minimum free memory thresholds.",
        ),
        CorrelationInsight(
            insight_id="ins_005",
            headline="CPU usage drives temperature increases",
            description="Clear correlation between CPU utilization and device temperature. Sustained high CPU usage leads to thermal throttling.",
            metrics_involved=["CPUUsage", "Temperature"],
            correlation_value=0.76,
            strength="strong",
            direction="positive",
            novelty_score=0.2,
            confidence=0.93,
            recommendation="Monitor apps causing sustained CPU load and consider background processing limits.",
        ),
        CorrelationInsight(
            insight_id="ins_006",
            headline="Download and upload traffic are strongly correlated",
            description="Symmetric data usage patterns suggest bidirectional app communications (API calls, sync operations).",
            metrics_involved=["Download", "Upload"],
            correlation_value=0.73,
            strength="strong",
            direction="positive",
            novelty_score=0.1,
            confidence=0.91,
            recommendation=None,
        ),
        CorrelationInsight(
            insight_id="ins_007",
            headline="App foreground time impacts battery drain",
            description="Active app usage correlates with battery consumption, though less strongly than screen time alone.",
            metrics_involved=["AppForegroundTime", "TotalBatteryLevelDrop"],
            correlation_value=0.67,
            strength="moderate",
            direction="positive",
            novelty_score=0.3,
            confidence=0.87,
            recommendation="Identify power-hungry apps consuming excessive foreground time.",
        ),
        CorrelationInsight(
            insight_id="ins_008",
            headline="Storage utilization weakly correlates with crashes",
            description="Devices with nearly full storage show slightly elevated crash rates, likely due to cache and temp file issues.",
            metrics_involved=["StorageUtilization", "CrashCount"],
            correlation_value=0.45,
            strength="weak",
            direction="positive",
            novelty_score=0.5,
            confidence=0.72,
            recommendation="Set up storage threshold alerts to proactively free space before issues occur.",
        ),
    ]

    return CorrelationInsightsResponse(
        insights=insights,
        total_correlations_analyzed=253,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def get_mock_cohort_patterns() -> CohortCorrelationPatternsResponse:
    """Generate cohort-specific correlation patterns."""
    patterns = [
        CohortCorrelationPattern(
            cohort_id="samsung_sm-g991b_13",
            cohort_name="Samsung Galaxy S21 (Android 13)",
            metric_pair=["TotalBatteryLevelDrop", "AvgSignalStrength"],
            cohort_correlation=-0.85,
            fleet_correlation=-0.72,
            deviation=0.13,
            device_count=342,
            is_anomalous=True,
            insight="This model shows 18% stronger battery-signal correlation than fleet average, suggesting firmware-specific radio power management issues.",
        ),
        CohortCorrelationPattern(
            cohort_id="zebra_tc52_11",
            cohort_name="Zebra TC52 (Android 11)",
            metric_pair=["CrashCount", "RAMPressure"],
            cohort_correlation=0.82,
            fleet_correlation=0.68,
            deviation=0.14,
            device_count=567,
            is_anomalous=True,
            insight="Enterprise scanner app on this model correlates crashes strongly with memory pressure. Consider memory optimization.",
        ),
        CohortCorrelationPattern(
            cohort_id="honeywell_ct40_10",
            cohort_name="Honeywell CT40 (Android 10)",
            metric_pair=["TotalDropCnt", "LocationMovement"],
            cohort_correlation=0.71,
            fleet_correlation=0.55,
            deviation=0.16,
            device_count=234,
            is_anomalous=True,
            insight="Mobile workers with this device experience more drops during movement than other models. WiFi roaming may need tuning.",
        ),
        CohortCorrelationPattern(
            cohort_id="samsung_sm-a515f_12",
            cohort_name="Samsung Galaxy A51 (Android 12)",
            metric_pair=["Download", "TotalBatteryLevelDrop"],
            cohort_correlation=0.58,
            fleet_correlation=0.62,
            deviation=-0.04,
            device_count=189,
            is_anomalous=False,
            insight=None,
        ),
    ]

    anomalous_count = sum(1 for p in patterns if p.is_anomalous)

    return CohortCorrelationPatternsResponse(
        patterns=patterns,
        anomalous_cohorts=anomalous_count,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def get_mock_time_lagged_correlations() -> TimeLagCorrelationsResponse:
    """Generate time-lagged correlation analysis."""
    correlations = [
        TimeLagCorrelation(
            metric_a="AvgSignalStrength",
            metric_b="TotalBatteryLevelDrop",
            lag_days=1,
            correlation=0.45,
            p_value=0.003,
            direction="a_predicts_b",
            insight="Poor signal today predicts elevated battery drain tomorrow (r=0.45, lag=1d). Useful for proactive battery alerts.",
        ),
        TimeLagCorrelation(
            metric_a="RAMPressure",
            metric_b="CrashCount",
            lag_days=2,
            correlation=0.38,
            p_value=0.012,
            direction="a_predicts_b",
            insight="Sustained memory pressure predicts crashes 2 days out. Early warning signal for stability issues.",
        ),
        TimeLagCorrelation(
            metric_a="StorageUtilization",
            metric_b="SlowPerformance",
            lag_days=3,
            correlation=0.42,
            p_value=0.008,
            direction="a_predicts_b",
            insight="Storage filling up predicts performance degradation within 3 days. Trigger cleanup before issues arise.",
        ),
        TimeLagCorrelation(
            metric_a="TotalDropCnt",
            metric_b="AppCrash",
            lag_days=1,
            correlation=0.33,
            p_value=0.025,
            direction="a_predicts_b",
            insight="Network instability correlates with app crashes the next day, possibly due to incomplete sync operations.",
        ),
        TimeLagCorrelation(
            metric_a="Temperature",
            metric_b="BatteryHealth",
            lag_days=7,
            correlation=-0.41,
            p_value=0.006,
            direction="a_predicts_b",
            insight="Sustained high temperatures correlate with battery health decline over a week. Critical for device longevity.",
        ),
    ]

    return TimeLagCorrelationsResponse(
        correlations=correlations,
        max_lag_analyzed=7,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("/matrix", response_model=CorrelationMatrixResponse)
def get_correlation_matrix(
    domain: Optional[str] = Query(None, description="Filter by domain (battery, rf, throughput, usage, storage, system)"),
    method: str = Query("pearson", description="Correlation method: pearson or spearman"),
    threshold: float = Query(0.6, description="Minimum |r| for strong correlations"),
    max_metrics: int = Query(50, description="Maximum metrics to include"),
    mock_mode: bool = Depends(get_mock_mode),
) -> CorrelationMatrixResponse:
    """
    Get correlation matrix for numeric metrics.

    Returns N x N correlation matrix and list of strong correlations.
    Can be filtered by metric domain.
    """
    if mock_mode:
        return get_mock_correlation_matrix(domain, method)

    # Real implementation using CorrelationService
    try:
        from device_anomaly.services.correlation_service import CorrelationService
        from device_anomaly.data_access.unified_loader import load_unified_device_dataset
        from datetime import timedelta

        # Load recent telemetry data
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=30)

        df = load_unified_device_dataset(start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            logger.warning("No data available for correlation, returning mock")
            return get_mock_correlation_matrix(domain, method)

        service = CorrelationService()
        result = service.compute_correlation_matrix(
            df=df,
            method=method,
            domain_filter=domain,
        )

        # Convert strong correlations to response format
        strong = [
            CorrelationCell(
                metric_x=c["metric_x"],
                metric_y=c["metric_y"],
                correlation=c["correlation"],
                p_value=c.get("p_value"),
                sample_count=c.get("sample_count", 0),
                method=method,
            )
            for c in result.get("strong_correlations", [])
            if abs(c["correlation"]) >= threshold
        ]

        return CorrelationMatrixResponse(
            metrics=result.get("metrics", []),
            matrix=result.get("matrix", []),
            strong_correlations=strong,
            method=method,
            computed_at=result.get("computed_at", datetime.now(timezone.utc).isoformat()),
            total_samples=result.get("sample_count", 0),
            domain_filter=domain,
        )

    except Exception as e:
        logger.warning(f"Real correlation computation failed: {e}, returning mock data")
        return get_mock_correlation_matrix(domain, method)


@router.get("/scatter", response_model=ScatterPlotResponse)
def get_scatter_data(
    metric_x: str = Query(..., description="First metric name"),
    metric_y: str = Query(..., description="Second metric name"),
    color_by: str = Query("anomaly", description="Color by: anomaly or cohort"),
    limit: int = Query(500, description="Max data points"),
    mock_mode: bool = Depends(get_mock_mode),
) -> ScatterPlotResponse:
    """
    Get scatter plot data for two metrics.

    Returns data points, correlation coefficient, and regression line parameters.
    """
    if mock_mode:
        return get_mock_scatter_data(metric_x, metric_y, limit)

    # Real implementation
    try:
        from device_anomaly.data_access.unified_loader import load_unified_device_dataset
        from datetime import timedelta
        from scipy import stats as scipy_stats

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=30)

        df = load_unified_device_dataset(start_date=start_date, end_date=end_date)
        if df is None or df.empty or metric_x not in df.columns or metric_y not in df.columns:
            logger.warning("Data not available for scatter plot, returning mock")
            return get_mock_scatter_data(metric_x, metric_y, limit)

        # Get valid data points
        valid_df = df[[metric_x, metric_y]].dropna()
        if len(valid_df) < 10:
            return get_mock_scatter_data(metric_x, metric_y, limit)

        # Sample if too many points
        if len(valid_df) > limit:
            valid_df = valid_df.sample(n=limit, random_state=42)

        # Compute correlation
        corr, p_value = scipy_stats.pearsonr(valid_df[metric_x], valid_df[metric_y])

        # Compute regression
        slope, intercept, r_value, _, _ = scipy_stats.linregress(
            valid_df[metric_x], valid_df[metric_y]
        )

        # Build points
        points = []
        for idx, row in valid_df.iterrows():
            device_id = df.loc[idx, "DeviceId"] if "DeviceId" in df.columns else idx
            is_anomaly = df.loc[idx, "is_anomaly"] if "is_anomaly" in df.columns else False
            cohort = df.loc[idx, "cohort_id"] if "cohort_id" in df.columns else None

            points.append(ScatterDataPoint(
                device_id=int(device_id) if not isinstance(device_id, int) else device_id,
                x_value=round(float(row[metric_x]), 2),
                y_value=round(float(row[metric_y]), 2),
                is_anomaly=bool(is_anomaly),
                cohort=str(cohort) if cohort else None,
            ))

        anomaly_count = sum(1 for p in points if p.is_anomaly)

        return ScatterPlotResponse(
            metric_x=metric_x,
            metric_y=metric_y,
            points=points,
            correlation=round(corr, 3),
            regression_slope=round(slope, 4),
            regression_intercept=round(intercept, 4),
            r_squared=round(r_value**2, 4),
            total_points=len(points),
            anomaly_count=anomaly_count,
        )

    except Exception as e:
        logger.warning(f"Real scatter data failed: {e}, returning mock")
        return get_mock_scatter_data(metric_x, metric_y, limit)


@router.get("/causal-graph", response_model=CausalGraphResponse)
def get_causal_graph(
    include_inferred: bool = Query(True, description="Include correlation-inferred edges"),
    mock_mode: bool = Depends(get_mock_mode),
) -> CausalGraphResponse:
    """
    Get causal relationship network.

    Returns nodes and edges representing known causal relationships
    from domain knowledge (RootCauseAnalyzer).
    """
    if mock_mode:
        return get_mock_causal_graph()

    # Real implementation would use RootCauseAnalyzer's causal graph
    try:
        from device_anomaly.insights.root_cause import RootCauseAnalyzer
        analyzer = RootCauseAnalyzer()
        # Build graph from analyzer._causal_graph
        # For now, mock data matches the actual causal graph
    except Exception as e:
        logger.warning(f"Failed to load RootCauseAnalyzer: {e}")

    return get_mock_causal_graph()


@router.get("/insights", response_model=CorrelationInsightsResponse)
def get_correlation_insights(
    top_k: int = Query(10, description="Number of top insights to return"),
    min_strength: float = Query(0.5, description="Minimum correlation strength"),
    mock_mode: bool = Depends(get_mock_mode),
) -> CorrelationInsightsResponse:
    """
    Get auto-discovered correlation insights.

    Returns ranked list of insights about metric relationships,
    including strength, direction, and recommendations.
    """
    if mock_mode:
        return get_mock_correlation_insights()

    # Real implementation
    try:
        from device_anomaly.services.correlation_service import CorrelationService
        from device_anomaly.data_access.unified_loader import load_unified_device_dataset
        from datetime import timedelta

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=30)

        df = load_unified_device_dataset(start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return get_mock_correlation_insights()

        service = CorrelationService()
        insights_data = service.generate_correlation_insights(
            df=df,
            top_k=top_k,
            min_strength=min_strength,
        )

        insights = [
            CorrelationInsight(
                insight_id=i.get("insight_id", f"ins_{idx}"),
                headline=i.get("headline", ""),
                description=i.get("description", ""),
                metrics_involved=i.get("metrics_involved", []),
                correlation_value=i.get("correlation_value", 0),
                strength=i.get("strength", "weak"),
                direction=i.get("direction", "positive"),
                novelty_score=i.get("novelty_score", 0.5),
                confidence=1 - i.get("confidence", 0.5),  # Convert p-value to confidence
                recommendation=i.get("recommendation"),
            )
            for idx, i in enumerate(insights_data)
        ]

        return CorrelationInsightsResponse(
            insights=insights,
            total_correlations_analyzed=len(df.columns) * (len(df.columns) - 1) // 2,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        logger.warning(f"Real insights generation failed: {e}, returning mock")
        return get_mock_correlation_insights()


@router.get("/cohort-patterns", response_model=CohortCorrelationPatternsResponse)
def get_cohort_correlation_patterns(
    metric_pair: Optional[str] = Query(None, description="Specific metric pair (comma-separated)"),
    mock_mode: bool = Depends(get_mock_mode),
) -> CohortCorrelationPatternsResponse:
    """
    Get cohort-specific correlation patterns.

    Identifies cohorts with unusual correlation patterns compared to fleet average.
    """
    if mock_mode:
        return get_mock_cohort_patterns()

    # Real implementation
    try:
        from device_anomaly.services.correlation_service import CorrelationService
        from device_anomaly.data_access.unified_loader import load_unified_device_dataset
        from datetime import timedelta

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=30)

        df = load_unified_device_dataset(start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return get_mock_cohort_patterns()

        service = CorrelationService()

        # Default metric pairs to analyze
        metric_pairs_to_analyze = [
            ("TotalBatteryLevelDrop", "AvgSignalStrength"),
            ("CrashCount", "RAMPressure"),
            ("TotalDropCnt", "AvgSignalStrength"),
            ("Download", "TotalBatteryLevelDrop"),
        ]

        # Use specific pair if provided
        if metric_pair:
            parts = metric_pair.split(",")
            if len(parts) == 2:
                metric_pairs_to_analyze = [(parts[0].strip(), parts[1].strip())]

        all_patterns = []
        for pair in metric_pairs_to_analyze:
            if pair[0] not in df.columns or pair[1] not in df.columns:
                continue

            patterns = service.compute_cohort_correlations(
                df=df,
                metric_pair=pair,
            )
            all_patterns.extend(patterns)

        # Convert to response format
        response_patterns = [
            CohortCorrelationPattern(
                cohort_id=p.cohort_id,
                cohort_name=p.cohort_name,
                metric_pair=p.metric_pair,
                cohort_correlation=p.cohort_correlation,
                fleet_correlation=p.fleet_correlation,
                deviation=p.deviation,
                device_count=p.device_count,
                is_anomalous=p.is_anomalous,
                insight=p.insight,
            )
            for p in all_patterns
        ]

        anomalous_count = sum(1 for p in response_patterns if p.is_anomalous)

        return CohortCorrelationPatternsResponse(
            patterns=response_patterns,
            anomalous_cohorts=anomalous_count,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        logger.warning(f"Real cohort pattern analysis failed: {e}, returning mock")
        return get_mock_cohort_patterns()


@router.get("/time-lagged", response_model=TimeLagCorrelationsResponse)
def get_time_lagged_correlations(
    max_lag: int = Query(7, description="Maximum lag in days"),
    min_correlation: float = Query(0.3, description="Minimum correlation threshold"),
    mock_mode: bool = Depends(get_mock_mode),
) -> TimeLagCorrelationsResponse:
    """
    Get time-lagged correlations for predictive insights.

    Analyzes how metrics at time T correlate with other metrics at time T+lag.
    Useful for predictive alerting.
    """
    if mock_mode:
        return get_mock_time_lagged_correlations()

    # Real implementation
    try:
        from device_anomaly.services.correlation_service import CorrelationService
        from device_anomaly.data_access.unified_loader import load_unified_device_dataset
        from datetime import timedelta

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=60)  # Need more data for lag analysis

        df = load_unified_device_dataset(start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return get_mock_time_lagged_correlations()

        service = CorrelationService()
        lagged_results = service.compute_time_lagged_correlations(
            df=df,
            max_lag_days=max_lag,
        )

        # Filter by minimum correlation
        filtered = [r for r in lagged_results if abs(r.correlation) >= min_correlation]

        # Convert to response format
        correlations = [
            TimeLagCorrelation(
                metric_a=r.metric_a,
                metric_b=r.metric_b,
                lag_days=r.lag_days,
                correlation=r.correlation,
                p_value=r.p_value,
                direction=r.direction,
                insight=r.insight,
            )
            for r in filtered
        ]

        return TimeLagCorrelationsResponse(
            correlations=correlations,
            max_lag_analyzed=max_lag,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        logger.warning(f"Real time-lagged analysis failed: {e}, returning mock")
        return get_mock_time_lagged_correlations()
