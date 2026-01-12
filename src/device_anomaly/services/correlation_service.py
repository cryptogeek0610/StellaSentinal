"""
Real Correlation Computation Service.

Computes actual correlations from telemetry data stored in the database,
replacing mock implementations with real statistical analysis.

Features:
- Pearson and Spearman correlations
- Time-lagged cross-correlations (predictive insights)
- Cohort-specific correlation patterns
- Result caching for performance
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CorrelationResult:
    """Result of correlation computation between two metrics."""

    metric_a: str
    metric_b: str
    correlation: float
    p_value: float
    sample_count: int
    method: str  # "pearson" or "spearman"
    confidence_interval: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    is_significant: bool = False


@dataclass
class TimeLaggedCorrelation:
    """Time-lagged correlation result for predictive insights."""

    metric_a: str
    metric_b: str
    lag_days: int
    correlation: float
    p_value: float
    direction: str  # "a_leads_b" or "b_leads_a"
    insight: str
    is_significant: bool = False


@dataclass
class CohortCorrelationPattern:
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


# =============================================================================
# Domain Knowledge
# =============================================================================

# Known strong correlations from domain knowledge (used as reference)
KNOWN_CORRELATIONS: Dict[Tuple[str, str], float] = {
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

# Metric domains for filtering
METRIC_DOMAINS: Dict[str, List[str]] = {
    "battery": [
        "TotalBatteryLevelDrop",
        "TotalDischargeTime_Sec",
        "ScreenOnTime_Sec",
        "BatteryDrainPerHour",
        "BatteryHealth",
    ],
    "rf": [
        "AvgSignalStrength",
        "TotalDropCnt",
        "WifiDisconnectCount",
        "CellSignalStrength",
    ],
    "throughput": [
        "Download",
        "Upload",
        "TotalDataUsage",
    ],
    "usage": [
        "AppForegroundTime",
        "CrashCount",
        "AppVisitCount",
        "ANRCount",
    ],
    "storage": [
        "StorageUtilization",
        "RAMPressure",
        "FreeStorageKb",
    ],
    "system": [
        "CPUUsage",
        "Temperature",
        "MemoryUsage",
        "RebootCount",
    ],
}


# =============================================================================
# Correlation Service
# =============================================================================


class CorrelationService:
    """
    Service for computing real correlations from telemetry data.

    Features:
    - Pearson and Spearman correlations
    - Time-lagged cross-correlations
    - Cohort-specific correlations
    - Caching (15 min TTL)
    """

    def __init__(
        self,
        cache_ttl_minutes: int = 15,
        min_samples: int = 30,
        significance_level: float = 0.05,
    ):
        """
        Initialize the correlation service.

        Args:
            cache_ttl_minutes: Cache time-to-live in minutes
            min_samples: Minimum samples required for correlation
            significance_level: P-value threshold for significance
        """
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.min_samples = min_samples
        self.significance_level = significance_level
        self._cache: Dict[str, Tuple[Any, datetime]] = {}

    def _get_cache_key(self, *args: Any) -> str:
        """Generate cache key from arguments."""
        key_str = str(args)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now(timezone.utc) - timestamp < self.cache_ttl:
                return value
            del self._cache[key]
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self._cache[key] = (value, datetime.now(timezone.utc))

    def compute_correlation_matrix(
        self,
        df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        method: str = "pearson",
        domain_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute correlation matrix from telemetry data.

        Args:
            df: DataFrame with telemetry data
            metrics: List of metrics to analyze (auto-detected if None)
            method: Correlation method ("pearson" or "spearman")
            domain_filter: Filter to specific domain

        Returns:
            Dictionary with correlation matrix and strong correlations
        """
        if df.empty:
            return {
                "error": "No data available for correlation computation",
                "metrics": [],
                "matrix": [],
                "strong_correlations": [],
            }

        # Determine metrics to analyze
        if metrics is None:
            if domain_filter and domain_filter in METRIC_DOMAINS:
                metrics = METRIC_DOMAINS[domain_filter]
            else:
                # Get all numeric columns
                metrics = [
                    col
                    for col in df.columns
                    if np.issubdtype(df[col].dtype, np.number)
                    and col not in {"DeviceId", "Timestamp", "CollectedDate"}
                ]

        # Filter to available metrics
        available_metrics = [m for m in metrics if m in df.columns]
        if len(available_metrics) < 2:
            return {
                "error": "Insufficient metrics for correlation",
                "metrics": available_metrics,
                "matrix": [],
                "strong_correlations": [],
            }

        # Limit to reasonable number
        available_metrics = available_metrics[:50]

        # Compute correlation matrix
        numeric_df = df[available_metrics].apply(pd.to_numeric, errors="coerce")

        if method == "spearman":
            corr_matrix = numeric_df.corr(method="spearman")
        else:
            corr_matrix = numeric_df.corr(method="pearson")

        # Compute p-values
        p_value_matrix = self._compute_p_values(numeric_df, method)

        # Find strong correlations
        strong_correlations = self._extract_strong_correlations(
            corr_matrix, p_value_matrix, numeric_df, threshold=0.5
        )

        return {
            "metrics": available_metrics,
            "matrix": corr_matrix.fillna(0).values.tolist(),
            "p_values": p_value_matrix.fillna(1).values.tolist(),
            "strong_correlations": strong_correlations,
            "method": method,
            "sample_count": len(df),
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "domain_filter": domain_filter,
        }

    def _compute_p_values(
        self,
        df: pd.DataFrame,
        method: str = "pearson",
    ) -> pd.DataFrame:
        """Compute p-value matrix for correlations."""
        cols = df.columns
        n = len(cols)
        p_values = np.ones((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                x = df[cols[i]].dropna()
                y = df[cols[j]].dropna()

                # Align indices
                common_idx = x.index.intersection(y.index)
                if len(common_idx) < self.min_samples:
                    continue

                x_aligned = x.loc[common_idx]
                y_aligned = y.loc[common_idx]

                try:
                    if method == "spearman":
                        _, p_val = stats.spearmanr(x_aligned, y_aligned)
                    else:
                        _, p_val = stats.pearsonr(x_aligned, y_aligned)

                    p_values[i, j] = p_val
                    p_values[j, i] = p_val
                except Exception:
                    pass

        return pd.DataFrame(p_values, index=cols, columns=cols)

    def _extract_strong_correlations(
        self,
        corr_matrix: pd.DataFrame,
        p_value_matrix: pd.DataFrame,
        df: pd.DataFrame,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Extract strong correlations from matrix."""
        strong = []
        metrics = corr_matrix.columns.tolist()

        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                corr = corr_matrix.iloc[i, j]
                p_val = p_value_matrix.iloc[i, j]

                if abs(corr) >= threshold and not np.isnan(corr):
                    # Count valid samples
                    x = df[metrics[i]].dropna()
                    y = df[metrics[j]].dropna()
                    common_idx = x.index.intersection(y.index)

                    strong.append({
                        "metric_x": metrics[i],
                        "metric_y": metrics[j],
                        "correlation": round(float(corr), 3),
                        "p_value": round(float(p_val), 4) if not np.isnan(p_val) else None,
                        "sample_count": len(common_idx),
                        "is_significant": p_val < self.significance_level if not np.isnan(p_val) else False,
                    })

        # Sort by absolute correlation
        strong.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return strong

    def compute_time_lagged_correlations(
        self,
        df: pd.DataFrame,
        metric_pairs: Optional[List[Tuple[str, str]]] = None,
        max_lag_days: int = 7,
    ) -> List[TimeLaggedCorrelation]:
        """
        Compute time-lagged correlations for predictive insights.

        Determines if metric A at time T predicts metric B at time T+lag.

        Args:
            df: DataFrame with time-series telemetry (must have timestamp)
            metric_pairs: Pairs of metrics to analyze
            max_lag_days: Maximum lag to test

        Returns:
            List of significant time-lagged correlations
        """
        if df.empty:
            return []

        # Default metric pairs to analyze
        if metric_pairs is None:
            metric_pairs = [
                ("AvgSignalStrength", "TotalBatteryLevelDrop"),
                ("RAMPressure", "CrashCount"),
                ("StorageUtilization", "CrashCount"),
                ("TotalDropCnt", "CrashCount"),
                ("Temperature", "BatteryHealth"),
                ("CPUUsage", "CrashCount"),
            ]

        results: List[TimeLaggedCorrelation] = []

        for metric_a, metric_b in metric_pairs:
            if metric_a not in df.columns or metric_b not in df.columns:
                continue

            best_result = self._find_best_lag(df, metric_a, metric_b, max_lag_days)
            if best_result is not None:
                results.append(best_result)

        # Sort by absolute correlation
        results.sort(key=lambda x: abs(x.correlation), reverse=True)
        return results

    def _find_best_lag(
        self,
        df: pd.DataFrame,
        metric_a: str,
        metric_b: str,
        max_lag_days: int,
    ) -> Optional[TimeLaggedCorrelation]:
        """Find the best lag for correlation between two metrics."""
        # Need time-ordered data
        if "Timestamp" not in df.columns and "CollectedDate" not in df.columns:
            return None

        time_col = "Timestamp" if "Timestamp" in df.columns else "CollectedDate"

        # Group by device and date for daily aggregation
        df_sorted = df.sort_values([time_col])

        best_lag = 0
        best_corr = 0.0
        best_p = 1.0

        # Test different lags
        for lag in range(1, max_lag_days + 1):
            # Shift metric_b by lag days
            series_a = df_sorted[metric_a].dropna()
            series_b = df_sorted[metric_b].shift(-lag).dropna()

            # Align
            common_idx = series_a.index.intersection(series_b.index)
            if len(common_idx) < self.min_samples:
                continue

            a_aligned = series_a.loc[common_idx]
            b_aligned = series_b.loc[common_idx]

            try:
                corr, p_value = stats.pearsonr(a_aligned, b_aligned)

                if abs(corr) > abs(best_corr) and p_value < self.significance_level:
                    best_corr = corr
                    best_lag = lag
                    best_p = p_value
            except Exception:
                continue

        # Minimum threshold for reporting
        if abs(best_corr) < 0.3:
            return None

        insight = self._generate_lag_insight(metric_a, metric_b, best_lag, best_corr)

        return TimeLaggedCorrelation(
            metric_a=metric_a,
            metric_b=metric_b,
            lag_days=best_lag,
            correlation=round(best_corr, 3),
            p_value=round(best_p, 4),
            direction="a_leads_b" if best_lag > 0 else "b_leads_a",
            insight=insight,
            is_significant=best_p < self.significance_level,
        )

    def _generate_lag_insight(
        self,
        metric_a: str,
        metric_b: str,
        lag_days: int,
        correlation: float,
    ) -> str:
        """Generate human-readable insight for lagged correlation."""
        direction = "higher" if correlation > 0 else "lower"
        metric_a_readable = metric_a.replace("_", " ").lower()
        metric_b_readable = metric_b.replace("_", " ").lower()

        if correlation > 0:
            return (
                f"Higher {metric_a_readable} today predicts higher {metric_b_readable} "
                f"in {lag_days} day(s) (r={correlation:.2f}). "
                "Useful for proactive alerting."
            )
        else:
            return (
                f"Lower {metric_a_readable} today predicts higher {metric_b_readable} "
                f"in {lag_days} day(s) (r={correlation:.2f}). "
                "Consider monitoring inverse patterns."
            )

    def compute_cohort_correlations(
        self,
        df: pd.DataFrame,
        metric_pair: Tuple[str, str],
        cohort_column: str = "cohort_id",
    ) -> List[CohortCorrelationPattern]:
        """
        Compute correlations per cohort and compare to fleet.

        Identifies cohorts with unusual correlation patterns.

        Args:
            df: DataFrame with telemetry and cohort info
            metric_pair: Tuple of two metrics to correlate
            cohort_column: Column containing cohort IDs

        Returns:
            List of cohort correlation patterns
        """
        metric_a, metric_b = metric_pair

        if df.empty or metric_a not in df.columns or metric_b not in df.columns:
            return []

        if cohort_column not in df.columns:
            # Try to build cohort_id
            from device_anomaly.features.cohort_stats import build_cohort_id

            cohort_id = build_cohort_id(df)
            if cohort_id is None:
                return []
            df = df.copy()
            df[cohort_column] = cohort_id

        # Fleet-wide correlation
        x_fleet = df[metric_a].dropna()
        y_fleet = df[metric_b].dropna()
        common_fleet = x_fleet.index.intersection(y_fleet.index)

        if len(common_fleet) < self.min_samples:
            return []

        fleet_corr, _ = stats.pearsonr(
            x_fleet.loc[common_fleet],
            y_fleet.loc[common_fleet],
        )

        # Per-cohort correlations
        patterns: List[CohortCorrelationPattern] = []

        for cohort_id, grp in df.groupby(cohort_column):
            if len(grp) < self.min_samples:
                continue

            x = grp[metric_a].dropna()
            y = grp[metric_b].dropna()
            common_idx = x.index.intersection(y.index)

            if len(common_idx) < 20:
                continue

            try:
                cohort_corr, p_value = stats.pearsonr(
                    x.loc[common_idx],
                    y.loc[common_idx],
                )
            except Exception:
                continue

            deviation = cohort_corr - fleet_corr
            is_anomalous = abs(deviation) > 0.15  # 0.15 difference is notable

            # Generate insight if anomalous
            insight = None
            if is_anomalous:
                direction = "stronger" if abs(cohort_corr) > abs(fleet_corr) else "weaker"
                insight = (
                    f"This cohort shows {abs(deviation):.0%} {direction} correlation "
                    f"between {metric_a} and {metric_b} than fleet average."
                )

            patterns.append(CohortCorrelationPattern(
                cohort_id=str(cohort_id),
                cohort_name=str(cohort_id),  # Would ideally look up human-readable name
                metric_pair=[metric_a, metric_b],
                cohort_correlation=round(cohort_corr, 3),
                fleet_correlation=round(fleet_corr, 3),
                deviation=round(deviation, 3),
                device_count=len(grp),
                is_anomalous=is_anomalous,
                insight=insight,
            ))

        # Sort by deviation magnitude
        patterns.sort(key=lambda x: abs(x.deviation), reverse=True)
        return patterns

    def generate_correlation_insights(
        self,
        df: pd.DataFrame,
        top_k: int = 10,
        min_strength: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Generate auto-discovered correlation insights.

        Args:
            df: DataFrame with telemetry data
            top_k: Number of top insights to return
            min_strength: Minimum correlation strength

        Returns:
            List of insight dictionaries
        """
        # Compute correlation matrix
        result = self.compute_correlation_matrix(df)
        strong_correlations = result.get("strong_correlations", [])

        insights = []
        for i, corr in enumerate(strong_correlations[:top_k]):
            if abs(corr["correlation"]) < min_strength:
                continue

            insight = self._generate_insight(corr)
            insights.append(insight)

        return insights

    def _generate_insight(self, corr: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insight from correlation result."""
        metric_x = corr["metric_x"]
        metric_y = corr["metric_y"]
        correlation = corr["correlation"]

        # Determine strength
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.5:
            strength = "moderate"
        else:
            strength = "weak"

        direction = "positive" if correlation > 0 else "negative"

        # Generate headline
        metric_x_readable = metric_x.replace("_", " ")
        metric_y_readable = metric_y.replace("_", " ")

        if correlation > 0:
            headline = f"{metric_x_readable} correlates with {metric_y_readable}"
        else:
            headline = f"{metric_x_readable} inversely correlates with {metric_y_readable}"

        # Generate description
        description = self._get_insight_description(metric_x, metric_y, correlation)

        # Generate recommendation
        recommendation = self._get_insight_recommendation(metric_x, metric_y, correlation)

        return {
            "insight_id": f"ins_{hashlib.md5(f'{metric_x}_{metric_y}'.encode()).hexdigest()[:6]}",
            "headline": headline,
            "description": description,
            "metrics_involved": [metric_x, metric_y],
            "correlation_value": correlation,
            "strength": strength,
            "direction": direction,
            "novelty_score": 0.5,  # Would compute based on domain knowledge
            "confidence": corr.get("p_value", 0.5),
            "recommendation": recommendation,
        }

    def _get_insight_description(
        self,
        metric_x: str,
        metric_y: str,
        correlation: float,
    ) -> str:
        """Generate description for correlation insight."""
        # Domain-specific descriptions
        descriptions = {
            ("TotalBatteryLevelDrop", "ScreenOnTime_Sec"): (
                "Devices with higher screen-on time show proportionally higher battery drain. "
                "This is expected behavior but can help identify devices with abnormal screen usage patterns."
            ),
            ("AvgSignalStrength", "TotalDropCnt"): (
                "Strong negative correlation between signal strength and connection drops. "
                "Devices in weak coverage areas experience significantly more disconnections."
            ),
            ("TotalBatteryLevelDrop", "AvgSignalStrength"): (
                "Devices constantly searching for better signal consume more battery. "
                "This cross-domain correlation explains unexpectedly high drain in certain locations."
            ),
            ("CrashCount", "RAMPressure"): (
                "Devices experiencing high RAM pressure show increased crash rates. "
                "This suggests memory-hungry apps may be destabilizing the system."
            ),
        }

        key = (metric_x, metric_y)
        reverse_key = (metric_y, metric_x)

        if key in descriptions:
            return descriptions[key]
        if reverse_key in descriptions:
            return descriptions[reverse_key]

        # Generic description
        direction = "higher" if correlation > 0 else "lower"
        return (
            f"Devices with higher {metric_x.lower().replace('_', ' ')} tend to show "
            f"{direction} {metric_y.lower().replace('_', ' ')}. "
            f"Correlation strength: {abs(correlation):.2f}."
        )

    def _get_insight_recommendation(
        self,
        metric_x: str,
        metric_y: str,
        correlation: float,
    ) -> Optional[str]:
        """Generate recommendation for correlation insight."""
        recommendations = {
            ("TotalBatteryLevelDrop", "ScreenOnTime_Sec"): (
                "Consider implementing screen brightness auto-adjustment policies for high-drain devices."
            ),
            ("AvgSignalStrength", "TotalDropCnt"): (
                "Map coverage dead zones and consider WiFi boosters or network optimization."
            ),
            ("TotalBatteryLevelDrop", "AvgSignalStrength"): (
                "Prioritize coverage improvements in high-activity areas to reduce battery impact."
            ),
            ("CrashCount", "RAMPressure"): (
                "Review memory usage of frequently used apps and consider increasing minimum free memory thresholds."
            ),
            ("StorageUtilization", "CrashCount"): (
                "Set up storage threshold alerts to proactively free space before issues occur."
            ),
        }

        key = (metric_x, metric_y)
        reverse_key = (metric_y, metric_x)

        if key in recommendations:
            return recommendations[key]
        if reverse_key in recommendations:
            return recommendations[reverse_key]

        return None


# =============================================================================
# Convenience Functions
# =============================================================================


def compute_correlations_from_dataframe(
    df: pd.DataFrame,
    domain: Optional[str] = None,
    method: str = "pearson",
) -> Dict[str, Any]:
    """
    Convenience function to compute correlations from a DataFrame.

    Args:
        df: DataFrame with telemetry data
        domain: Optional domain filter
        method: Correlation method

    Returns:
        Correlation matrix result
    """
    service = CorrelationService()
    return service.compute_correlation_matrix(df, domain_filter=domain, method=method)


def get_time_lagged_insights(
    df: pd.DataFrame,
    max_lag_days: int = 7,
) -> List[TimeLaggedCorrelation]:
    """
    Get time-lagged correlation insights.

    Args:
        df: DataFrame with time-series data
        max_lag_days: Maximum lag to test

    Returns:
        List of significant time-lagged correlations
    """
    service = CorrelationService()
    return service.compute_time_lagged_correlations(df, max_lag_days=max_lag_days)
