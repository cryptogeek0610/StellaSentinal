"""
Mock data generators for Correlation Intelligence endpoints.

Used when mock_mode is enabled or as fallback when real data is unavailable.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Dict, List, Optional

from device_anomaly.api.schemas.correlations import (
    CausalEdge,
    CausalGraphResponse,
    CausalNode,
    CohortCorrelationPattern,
    CohortCorrelationPatternsResponse,
    CorrelationCell,
    CorrelationInsight,
    CorrelationInsightsResponse,
    CorrelationMatrixResponse,
    FilterStats,
    ScatterAnomalyExplanation,
    ScatterDataPoint,
    ScatterPlotResponse,
    TimeLagCorrelation,
    TimeLagCorrelationsResponse,
)


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

    # Build p-value matrix (mock p-values based on correlation strength)
    p_values = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                p_values[i][j] = 0.0  # Self-correlation has p=0
            else:
                # Mock p-value: stronger correlations have lower p-values
                abs_corr = abs(matrix[i][j])
                if abs_corr > 0.7:
                    p_values[i][j] = round(random.uniform(0.0001, 0.001), 4)
                elif abs_corr > 0.5:
                    p_values[i][j] = round(random.uniform(0.001, 0.01), 4)
                elif abs_corr > 0.3:
                    p_values[i][j] = round(random.uniform(0.01, 0.05), 4)
                else:
                    p_values[i][j] = round(random.uniform(0.05, 0.5), 4)

    # Find strong correlations
    strong = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i][j]) >= 0.6:
                p_val = p_values[i][j]
                strong.append(
                    CorrelationCell(
                        metric_x=metrics[i],
                        metric_y=metrics[j],
                        correlation=matrix[i][j],
                        p_value=p_val,
                        sample_count=random.randint(3000, 5000),
                        method=method,
                        is_significant=p_val < 0.05,
                    )
                )

    # Sort by absolute correlation
    strong.sort(key=lambda x: abs(x.correlation), reverse=True)

    # Mock filter stats
    total_input = n + random.randint(5, 15)  # Pretend we had more metrics initially
    filter_stats = FilterStats(
        total_input=total_input,
        low_variance=random.randint(1, 3),
        low_cardinality=random.randint(1, 2),
        high_null=random.randint(2, 5),
        passed=n,
    )

    return CorrelationMatrixResponse(
        metrics=metrics,
        matrix=matrix,
        p_values=p_values,
        strong_correlations=strong,
        method=method,
        computed_at=datetime.now(timezone.utc).isoformat(),
        total_samples=4532,
        domain_filter=domain,
        date_range={"start": "2025-12-20", "end": "2026-01-19"},
        filter_stats=filter_stats,
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

        points.append(
            ScatterDataPoint(
                device_id=1000 + i,
                x_value=round(max(0, x), 2),
                y_value=round(max(0, y), 2),
                is_anomaly=is_anomaly,
                cohort=random.choice(cohorts),
                timestamp=f"2025-01-{random.randint(1, 9):02d}T{random.randint(0, 23):02d}:00:00Z",
            )
        )

    # Calculate r-squared
    r_squared = corr**2

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
        "ScreenOnTime": "usage",
        "AppForegroundTime": "usage",
        "BackgroundAppActivity": "usage",
        "PoorSignal": "rf",
        "WeakWifiCoverage": "rf",
        "CellCoverage": "rf",
        "SignalStrength": "rf",
        "ChargingPattern": "battery",
        "BatteryHealth": "battery",
        "BatteryDrain": "battery",
        "BatteryDrainPerHour": "battery",
        "BatteryCapacity": "battery",
        "Temperature": "system",
        "CPUThrottle": "system",
        "HighCPU": "system",
        "LocationMovement": "location",
        "APHopping": "connectivity",
        "NetworkDrops": "connectivity",
        "TowerHopping": "connectivity",
        "NetworkCongestion": "connectivity",
        "HighDataUsage": "throughput",
        "AppVersion": "app",
        "AppCrash": "app",
        "ANR": "app",
        "AppBehavior": "app",
        "LowMemory": "storage",
        "LowStorage": "storage",
        "InstallFailure": "storage",
        "SlowPerformance": "system",
        "OsVersion": "system",
        "AgentVersion": "system",
        "SecurityVulnerability": "security",
        "Rooted": "security",
        "SecurityIssues": "security",
        "DataCollectionIssues": "system",
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
            edges.append(
                CausalEdge(
                    source=cause,
                    target=effect,
                    relationship="causes",
                    strength=0.8,
                    evidence=f"{cause} is known to cause {effect}",
                )
            )

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


def get_mock_scatter_explanation(
    device_id: int,
    metric_x: str,
    metric_y: str,
    x_value: float,
    y_value: float,
) -> ScatterAnomalyExplanation:
    """Generate a mock scatter anomaly explanation for demo/testing."""
    return ScatterAnomalyExplanation(
        explanation=(
            f"Device {device_id} shows unusual behavior in the {metric_x} vs {metric_y} relationship. "
            f"The values ({x_value:.2f}, {y_value:.2f}) deviate significantly from the expected regression pattern, "
            "suggesting this device may be experiencing conditions that differ from the fleet norm."
        ),
        what_happened=(
            f"This device reported {metric_x}={x_value:.2f} and {metric_y}={y_value:.2f}, "
            "which places it outside the expected correlation pattern. "
            "The combination of these two metrics suggests abnormal device behavior."
        ),
        key_concerns=[
            f"{metric_x} value ({x_value:.2f}) is unusually high/low relative to {metric_y}",
            "Device deviates from the fleet's typical correlation pattern",
            "Possible hardware stress, environmental factors, or usage anomaly",
        ],
        likely_explanation=(
            "This anomaly could be caused by: (1) abnormal usage patterns such as intensive workloads, "
            "(2) environmental conditions affecting device performance, "
            "or (3) early signs of hardware degradation affecting one or both metrics."
        ),
        suggested_action=(
            "Review the device's recent activity logs and compare with similar devices in the same cohort. "
            "If the pattern persists, consider investigating environmental factors or scheduling preventive maintenance."
        ),
    )
