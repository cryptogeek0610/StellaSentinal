"""
Cross-Device Pattern Detection.

Detects patterns affecting entire device segments:
- Model-wide firmware bugs (e.g., "Samsung SM-G991B + Android 13: 2.3x higher crash rate")
- OS version stability issues
- Manufacturer quality patterns
- Location-based infrastructure problems
- Carrier-specific network issues

This module enables the detection of systemic issues that affect entire cohorts
of devices, rather than individual device anomalies.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CohortPattern:
    """A detected cross-device pattern affecting a cohort."""

    pattern_id: str
    pattern_type: (
        str  # "model_issue", "os_issue", "firmware_issue", "location_issue", "carrier_issue"
    )
    cohort_definition: dict[
        str, str
    ]  # {"manufacturer": "Samsung", "model": "SM-G991B", "os": "13"}
    cohort_name: str  # Human-readable: "Samsung Galaxy S21 (Android 13)"
    affected_devices: int
    fleet_percentage: float
    primary_metric: str
    metric_value: float  # Cohort's average value
    fleet_value: float  # Fleet's average value
    deviation_z: float  # Z-score of cohort vs fleet
    vs_fleet_multiplier: float  # e.g., 2.3x higher
    p_value: float
    effect_size: float  # Cohen's d
    confidence: float
    severity: str  # "critical", "high", "medium", "low"
    direction: str  # "elevated" or "reduced"
    evidence: list[dict[str, Any]] = field(default_factory=list)
    recommendation: str = ""
    detected_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class CohortBaseline:
    """Fleet-wide baseline statistics for a metric."""

    metric: str
    mean: float
    std: float
    median: float
    p25: float
    p75: float
    p5: float
    p95: float
    count: int


@dataclass
class SystemicIssue:
    """A systemic issue affecting multiple devices."""

    issue_id: str
    issue_type: str
    cohort_id: str
    cohort_name: str
    device_count: int
    fleet_percentage: float
    primary_metric: str
    vs_fleet_multiplier: float
    severity: str
    trend: str  # "worsening", "stable", "improving"
    first_detected: str
    recommendation: str
    related_patterns: list[str] = field(default_factory=list)


# =============================================================================
# Cohort Dimension Configuration
# =============================================================================

# Default dimensions to analyze for patterns
DEFAULT_COHORT_DIMENSIONS = [
    "ManufacturerId",
    "ModelId",
    "OsVersionId",
    "FirmwareVersion",
    "Carrier",
    "CarrierCode",
    "LocationRegion",
    "SiteId",
    "BusinessUnit",
]

# Key metrics to analyze for cohort patterns
DEFAULT_PATTERN_METRICS = [
    # Battery metrics
    "TotalBatteryLevelDrop",
    "BatteryDrainPerHour",
    "TotalDischargeTime_Sec",
    # Connectivity metrics
    "TotalDropCnt",
    "AvgSignalStrength",
    "WifiDisconnectCount",
    # Stability metrics
    "CrashCount",
    "ANRCount",
    "RebootCount",
    # Performance metrics
    "RAMPressure",
    "StorageUtilization",
    "CPUUsage",
    # Data usage
    "TotalDownload",
    "TotalUpload",
]

# Metrics where HIGH is bad (most metrics)
HIGH_IS_BAD_METRICS = {
    "TotalBatteryLevelDrop",
    "BatteryDrainPerHour",
    "TotalDropCnt",
    "WifiDisconnectCount",
    "CrashCount",
    "ANRCount",
    "RebootCount",
    "RAMPressure",
    "StorageUtilization",
    "CPUUsage",
}

# Metrics where LOW is bad
LOW_IS_BAD_METRICS = {
    "AvgSignalStrength",
    "FreeStorageKb",
    "AvailableRAM",
    "BatteryHealth",
}


# =============================================================================
# Cross-Device Pattern Detector
# =============================================================================


class CrossDevicePatternDetector:
    """
    Multi-dimensional cohort pattern detector.

    Detects patterns across multiple segmentation dimensions:
    1. Manufacturer-level (all Samsung devices)
    2. Model-level (all TC52 devices)
    3. OS Version-level (all Android 13 devices)
    4. Manufacturer + OS (Samsung + Android 13)
    5. Manufacturer + Model + OS (Samsung SM-G991B + Android 13)
    6. Location-based (all devices at Warehouse A)
    7. Carrier-based (all Verizon devices)

    Uses statistical tests to determine if observed differences are
    significant vs random variation.
    """

    def __init__(
        self,
        cohort_dimensions: list[str] | None = None,
        pattern_metrics: list[str] | None = None,
        significance_level: float = 0.05,
        min_effect_size: float = 0.5,  # Cohen's d threshold
        min_sample_size: int = 20,
        min_fleet_percentage: float = 1.0,  # Only flag if >1% of fleet
        z_threshold: float = 2.0,  # Cohort is anomalous if Z > 2
    ):
        """
        Initialize the detector.

        Args:
            cohort_dimensions: Columns to use for cohort segmentation
            pattern_metrics: Metrics to analyze for patterns
            significance_level: P-value threshold for statistical significance
            min_effect_size: Minimum Cohen's d for practical significance
            min_sample_size: Minimum devices in a cohort to analyze
            min_fleet_percentage: Minimum percentage of fleet to flag
            z_threshold: Z-score threshold for flagging anomalous cohorts
        """
        self.cohort_dimensions = cohort_dimensions or DEFAULT_COHORT_DIMENSIONS
        self.pattern_metrics = pattern_metrics or DEFAULT_PATTERN_METRICS
        self.significance_level = significance_level
        self.min_effect_size = min_effect_size
        self.min_sample_size = min_sample_size
        self.min_fleet_percentage = min_fleet_percentage
        self.z_threshold = z_threshold

        # Learned baselines from fit()
        self.fleet_baselines: dict[str, CohortBaseline] = {}
        self.fleet_size: int = 0
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> CrossDevicePatternDetector:
        """
        Learn fleet-wide baselines from training data.

        Args:
            df: DataFrame with device telemetry and cohort columns

        Returns:
            self for method chaining
        """
        if df.empty:
            logger.warning("Cannot fit on empty DataFrame")
            return self

        self.fleet_size = len(df)

        # Compute baseline statistics for each metric
        for metric in self.pattern_metrics:
            if metric not in df.columns:
                continue

            series = pd.to_numeric(df[metric], errors="coerce").dropna()
            if len(series) < self.min_sample_size:
                continue

            self.fleet_baselines[metric] = CohortBaseline(
                metric=metric,
                mean=float(series.mean()),
                std=float(series.std()) if series.std() > 0 else 1e-6,
                median=float(series.median()),
                p25=float(series.quantile(0.25)),
                p75=float(series.quantile(0.75)),
                p5=float(series.quantile(0.05)),
                p95=float(series.quantile(0.95)),
                count=len(series),
            )

        self._fitted = True
        logger.info(
            f"CrossDevicePatternDetector fitted on {self.fleet_size} devices, "
            f"{len(self.fleet_baselines)} metrics"
        )
        return self

    def detect_patterns(
        self,
        df: pd.DataFrame,
        metrics: list[str] | None = None,
    ) -> list[CohortPattern]:
        """
        Detect statistically significant cross-device patterns.

        For each cohort dimension combination:
        1. Group devices by cohort
        2. Compute cohort aggregate metrics
        3. Compare to fleet baseline using statistical tests
        4. Flag cohorts with significant deviations

        Args:
            df: DataFrame with device telemetry and cohort columns
            metrics: Optional list of metrics to analyze (uses defaults if None)

        Returns:
            List of detected CohortPattern objects
        """
        if df.empty:
            return []

        # Use fitted baselines or compute from current data
        if not self._fitted:
            logger.info("Detector not fitted, computing baselines from current data")
            self.fit(df)

        metrics = metrics or self.pattern_metrics
        patterns: list[CohortPattern] = []

        # Generate all cohort dimension combinations to analyze
        dimension_combos = self._get_dimension_combinations(df)

        for dims in dimension_combos:
            # Skip if any dimension not in dataframe
            if not all(d in df.columns for d in dims):
                continue

            # Group and analyze
            cohort_patterns = self._analyze_cohort_dimension(df, dims, metrics)
            patterns.extend(cohort_patterns)

        # Deduplicate and rank patterns
        patterns = self._deduplicate_patterns(patterns)
        patterns = self._rank_patterns(patterns)

        logger.info(f"Detected {len(patterns)} cross-device patterns")
        return patterns

    def detect_systemic_issues(
        self,
        df: pd.DataFrame,
        min_devices: int = 10,
        min_z: float = 2.0,
    ) -> list[SystemicIssue]:
        """
        Detect systemic issues affecting device cohorts.

        This is a higher-level wrapper that returns issues in a format
        suitable for the API.

        Args:
            df: DataFrame with device telemetry
            min_devices: Minimum devices for a cohort to be considered
            min_z: Minimum Z-score to flag as issue

        Returns:
            List of SystemicIssue objects
        """
        # Detect patterns
        patterns = self.detect_patterns(df)

        # Filter to significant issues
        issues: list[SystemicIssue] = []

        for pattern in patterns:
            if pattern.affected_devices < min_devices:
                continue
            if abs(pattern.deviation_z) < min_z:
                continue

            issue_id = hashlib.md5(
                f"{pattern.cohort_name}_{pattern.primary_metric}".encode()
            ).hexdigest()[:12]

            issue = SystemicIssue(
                issue_id=issue_id,
                issue_type=pattern.pattern_type,
                cohort_id=pattern.pattern_id,
                cohort_name=pattern.cohort_name,
                device_count=pattern.affected_devices,
                fleet_percentage=pattern.fleet_percentage,
                primary_metric=pattern.primary_metric,
                vs_fleet_multiplier=pattern.vs_fleet_multiplier,
                severity=pattern.severity,
                trend="stable",  # TODO: Compute trend from historical data
                first_detected=pattern.detected_at,
                recommendation=pattern.recommendation,
            )
            issues.append(issue)

        # Sort by severity and device count
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        issues.sort(key=lambda x: (severity_order.get(x.severity, 4), -x.device_count))

        return issues

    def get_model_reliability_rankings(
        self,
        df: pd.DataFrame,
        top_n: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Get device model reliability rankings.

        Returns models ranked by composite reliability score based on
        crash rate, drop rate, reboot rate, etc.

        Args:
            df: DataFrame with device telemetry
            top_n: Number of models to return

        Returns:
            List of model reliability rankings
        """
        if "ModelId" not in df.columns:
            return []

        # Metrics that indicate reliability issues (higher = worse)
        reliability_metrics = ["CrashCount", "TotalDropCnt", "RebootCount", "ANRCount"]
        available_metrics = [m for m in reliability_metrics if m in df.columns]

        if not available_metrics:
            return []

        # Group by model
        model_stats = []
        for model_id, grp in df.groupby("ModelId"):
            if len(grp) < self.min_sample_size:
                continue

            # Get model name if available
            model_name = str(model_id)
            if "ModelName" in grp.columns:
                model_name = grp["ModelName"].iloc[0] or str(model_id)

            # Compute reliability metrics
            stats_dict = {
                "model_id": str(model_id),
                "model_name": model_name,
                "device_count": len(grp),
            }

            # Add manufacturer if available
            if "ManufacturerId" in grp.columns:
                stats_dict["manufacturer"] = str(grp["ManufacturerId"].iloc[0])

            # Compute average for each metric
            total_score = 0
            for metric in available_metrics:
                if metric in grp.columns:
                    series = pd.to_numeric(grp[metric], errors="coerce").dropna()
                    if len(series) > 0:
                        avg = float(series.mean())
                        stats_dict[f"avg_{metric.lower()}"] = round(avg, 2)

                        # Compare to fleet baseline
                        if metric in self.fleet_baselines:
                            baseline = self.fleet_baselines[metric]
                            if baseline.std > 0:
                                z = (avg - baseline.mean) / baseline.std
                                stats_dict[f"{metric.lower()}_z"] = round(z, 2)
                                total_score += z

            stats_dict["reliability_score"] = round(-total_score, 2)  # Higher = better
            model_stats.append(stats_dict)

        # Sort by reliability score (higher = better)
        model_stats.sort(key=lambda x: x.get("reliability_score", 0), reverse=True)

        return model_stats[:top_n]

    def get_os_stability_analysis(
        self,
        df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Analyze OS version stability.

        Returns which OS versions are showing elevated issue rates.

        Args:
            df: DataFrame with device telemetry

        Returns:
            List of OS version stability analysis
        """
        if "OsVersionId" not in df.columns:
            return []

        stability_metrics = ["CrashCount", "ANRCount", "RebootCount", "TotalDropCnt"]
        available_metrics = [m for m in stability_metrics if m in df.columns]

        if not available_metrics:
            return []

        # Group by OS version
        os_stats = []
        for os_version, grp in df.groupby("OsVersionId"):
            if len(grp) < self.min_sample_size:
                continue

            os_name = str(os_version)
            if "OsVersionName" in grp.columns:
                os_name = grp["OsVersionName"].iloc[0] or str(os_version)

            stats_dict = {
                "os_version_id": str(os_version),
                "os_version": os_name,
                "device_count": len(grp),
                "fleet_percentage": round(100 * len(grp) / len(df), 1),
            }

            # Compute stability metrics
            issues_detected = []
            for metric in available_metrics:
                if metric in grp.columns:
                    series = pd.to_numeric(grp[metric], errors="coerce").dropna()
                    if len(series) > 0:
                        avg = float(series.mean())
                        stats_dict[f"avg_{metric.lower()}"] = round(avg, 2)

                        # Compare to fleet
                        if metric in self.fleet_baselines:
                            baseline = self.fleet_baselines[metric]
                            if baseline.std > 0:
                                z = (avg - baseline.mean) / baseline.std
                                stats_dict[f"{metric.lower()}_z"] = round(z, 2)

                                if z > self.z_threshold:
                                    issues_detected.append(
                                        f"{metric}: {avg:.1f} ({z:.1f}σ above fleet)"
                                    )

            stats_dict["issues"] = issues_detected
            stats_dict["status"] = "warning" if issues_detected else "healthy"

            os_stats.append(stats_dict)

        # Sort by number of issues (descending), then by device count
        os_stats.sort(key=lambda x: (-len(x.get("issues", [])), -x["device_count"]))

        return os_stats

    def get_firmware_impact_analysis(
        self,
        df: pd.DataFrame,
        manufacturer: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Analyze firmware version impact on device health.

        Args:
            df: DataFrame with device telemetry
            manufacturer: Optional filter by manufacturer
            model: Optional filter by model

        Returns:
            List of firmware impact analysis
        """
        if "FirmwareVersion" not in df.columns:
            return []

        # Apply filters
        filtered_df = df.copy()
        if manufacturer and "ManufacturerId" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["ManufacturerId"].astype(str) == str(manufacturer)
            ]
        if model and "ModelId" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["ModelId"].astype(str) == str(model)]

        if filtered_df.empty:
            return []

        health_metrics = [
            "TotalBatteryLevelDrop",
            "CrashCount",
            "TotalDropCnt",
            "RAMPressure",
        ]
        available_metrics = [m for m in health_metrics if m in filtered_df.columns]

        if not available_metrics:
            return []

        # Group by firmware version
        firmware_stats = []
        for firmware, grp in filtered_df.groupby("FirmwareVersion"):
            if len(grp) < 5:  # Lower threshold for firmware analysis
                continue

            firmware_str = str(firmware) if firmware else "Unknown"

            stats_dict = {
                "firmware_version": firmware_str,
                "device_count": len(grp),
            }

            # Add context
            if "ManufacturerId" in grp.columns:
                stats_dict["manufacturer"] = str(grp["ManufacturerId"].iloc[0])
            if "ModelId" in grp.columns:
                stats_dict["model"] = str(grp["ModelId"].iloc[0])

            # Compute health metrics
            health_score = 0
            for metric in available_metrics:
                if metric in grp.columns:
                    series = pd.to_numeric(grp[metric], errors="coerce").dropna()
                    if len(series) > 0:
                        avg = float(series.mean())
                        stats_dict[f"avg_{metric.lower()}"] = round(avg, 2)

                        # Compare to fleet baseline
                        if metric in self.fleet_baselines:
                            baseline = self.fleet_baselines[metric]
                            if baseline.std > 0:
                                z = (avg - baseline.mean) / baseline.std
                                # For bad metrics, positive Z is bad
                                if metric in HIGH_IS_BAD_METRICS:
                                    health_score -= z
                                else:
                                    health_score += z

            stats_dict["health_score"] = round(health_score, 2)
            stats_dict["status"] = (
                "healthy" if health_score > 0 else "warning" if health_score > -1 else "critical"
            )

            firmware_stats.append(stats_dict)

        # Sort by health score (higher = better)
        firmware_stats.sort(key=lambda x: x.get("health_score", 0), reverse=True)

        return firmware_stats

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _get_dimension_combinations(
        self,
        df: pd.DataFrame,
    ) -> list[list[str]]:
        """Generate meaningful dimension combinations to analyze."""
        available_dims = [d for d in self.cohort_dimensions if d in df.columns]

        if not available_dims:
            return []

        combos: list[list[str]] = []

        # Single dimensions (most general)
        combos.extend([[d] for d in available_dims])

        # Two-dimension combinations (most useful for cross-device patterns)
        from itertools import combinations

        for combo in combinations(available_dims, 2):
            combos.append(list(combo))

        # Three-dimension combinations (most specific)
        # Only use important combinations to avoid explosion
        important_3d = [
            ["ManufacturerId", "ModelId", "OsVersionId"],
            ["ManufacturerId", "ModelId", "FirmwareVersion"],
            ["ManufacturerId", "OsVersionId", "Carrier"],
        ]
        for combo in important_3d:
            if all(d in available_dims for d in combo):
                combos.append(combo)

        return combos

    def _analyze_cohort_dimension(
        self,
        df: pd.DataFrame,
        dimensions: list[str],
        metrics: list[str],
    ) -> list[CohortPattern]:
        """Analyze patterns for a specific dimension combination."""
        patterns: list[CohortPattern] = []

        # Group by dimension(s)
        grouped = df.groupby(dimensions)

        for cohort_key, cohort_df in grouped:
            # Ensure cohort_key is a tuple
            if not isinstance(cohort_key, tuple):
                cohort_key = (cohort_key,)

            # Skip small cohorts
            if len(cohort_df) < self.min_sample_size:
                continue

            # Skip if below minimum fleet percentage
            fleet_pct = 100 * len(cohort_df) / max(1, len(df))
            if fleet_pct < self.min_fleet_percentage:
                continue

            # Build cohort definition
            cohort_def = dict(zip(dimensions, [str(k) for k in cohort_key], strict=False))
            cohort_name = self._build_cohort_name(cohort_def, cohort_df)
            cohort_id = "_".join(str(k) for k in cohort_key)

            # Analyze each metric
            for metric in metrics:
                if metric not in cohort_df.columns or metric not in self.fleet_baselines:
                    continue

                pattern = self._analyze_metric_for_cohort(
                    cohort_df=cohort_df,
                    metric=metric,
                    cohort_id=cohort_id,
                    cohort_name=cohort_name,
                    cohort_def=cohort_def,
                    dimensions=dimensions,
                    fleet_size=len(df),
                )

                if pattern is not None:
                    patterns.append(pattern)

        return patterns

    def _analyze_metric_for_cohort(
        self,
        cohort_df: pd.DataFrame,
        metric: str,
        cohort_id: str,
        cohort_name: str,
        cohort_def: dict[str, str],
        dimensions: list[str],
        fleet_size: int,
    ) -> CohortPattern | None:
        """Analyze a specific metric for a cohort."""
        baseline = self.fleet_baselines[metric]

        # Get cohort metric values
        cohort_series = pd.to_numeric(cohort_df[metric], errors="coerce").dropna()
        if len(cohort_series) < 5:
            return None

        cohort_mean = float(cohort_series.mean())
        cohort_std = float(cohort_series.std()) if cohort_series.std() > 0 else 1e-6

        # Compute Z-score (cohort mean vs fleet mean)
        z_score = (cohort_mean - baseline.mean) / baseline.std if baseline.std > 0 else 0

        # Skip if Z-score not significant
        if abs(z_score) < self.z_threshold:
            return None

        # Statistical significance test (t-test)
        # We compare cohort mean to fleet mean
        # Using one-sample t-test since we have the population mean
        try:
            t_stat, p_value = stats.ttest_1samp(cohort_series, baseline.mean)
        except Exception:
            p_value = 1.0

        if p_value > self.significance_level:
            return None

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((cohort_std**2 + baseline.std**2) / 2)
        effect_size = abs(cohort_mean - baseline.mean) / max(pooled_std, 1e-6)

        if effect_size < self.min_effect_size:
            return None

        # Compute multiplier
        if baseline.mean != 0:
            multiplier = cohort_mean / baseline.mean
        else:
            multiplier = 1.0 if cohort_mean == 0 else float("inf")

        # Determine direction and severity
        is_high_bad = metric in HIGH_IS_BAD_METRICS
        is_low_bad = metric in LOW_IS_BAD_METRICS

        if is_high_bad:
            direction = "elevated" if z_score > 0 else "reduced"
            is_bad = z_score > 0
        elif is_low_bad:
            direction = "elevated" if z_score > 0 else "reduced"
            is_bad = z_score < 0
        else:
            direction = "elevated" if z_score > 0 else "reduced"
            is_bad = abs(z_score) > self.z_threshold

        if not is_bad:
            return None  # Skip beneficial deviations

        severity = self._compute_severity(abs(z_score), effect_size, is_bad)

        # Determine pattern type
        pattern_type = self._infer_pattern_type(dimensions, cohort_def)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            pattern_type=pattern_type,
            cohort_name=cohort_name,
            metric=metric,
            direction=direction,
            multiplier=multiplier,
            severity=severity,
        )

        # Create pattern ID
        pattern_id = hashlib.md5(f"{cohort_id}_{metric}".encode()).hexdigest()[:12]

        return CohortPattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            cohort_definition=cohort_def,
            cohort_name=cohort_name,
            affected_devices=len(cohort_df),
            fleet_percentage=round(100 * len(cohort_df) / max(1, fleet_size), 1),
            primary_metric=metric,
            metric_value=round(cohort_mean, 2),
            fleet_value=round(baseline.mean, 2),
            deviation_z=round(z_score, 2),
            vs_fleet_multiplier=round(multiplier, 2),
            p_value=round(p_value, 4),
            effect_size=round(effect_size, 2),
            confidence=round(1 - p_value, 2),
            severity=severity,
            direction=direction,
            recommendation=recommendation,
        )

    def _build_cohort_name(
        self,
        cohort_def: dict[str, str],
        cohort_df: pd.DataFrame,
    ) -> str:
        """Build human-readable cohort name."""
        parts = []

        # Try to get human-readable names from data
        name_columns = {
            "ManufacturerId": "ManufacturerName",
            "ModelId": "ModelName",
            "OsVersionId": "OsVersionName",
        }

        for dim, value in cohort_def.items():
            name_col = name_columns.get(dim)
            if name_col and name_col in cohort_df.columns:
                name_value = cohort_df[name_col].iloc[0]
                if name_value and str(name_value) != "nan":
                    parts.append(str(name_value))
                    continue

            # Fall back to ID
            if dim == "OsVersionId":
                parts.append(f"Android {value}")
            elif dim == "FirmwareVersion":
                parts.append(f"FW {value}")
            elif dim in ("Carrier", "CarrierCode"):
                parts.append(f"{value}")
            elif dim in ("LocationRegion", "SiteId"):
                parts.append(f"Site {value}")
            else:
                parts.append(str(value))

        return " ".join(parts) if parts else "Unknown Cohort"

    def _infer_pattern_type(
        self,
        dimensions: list[str],
        cohort_def: dict[str, str],
    ) -> str:
        """Infer the pattern type from dimensions."""
        if "FirmwareVersion" in dimensions:
            return "firmware_issue"
        if "ModelId" in dimensions or "ManufacturerId" in dimensions:
            return "model_issue"
        if "OsVersionId" in dimensions:
            return "os_issue"
        if any(d in dimensions for d in ("LocationRegion", "SiteId", "BusinessUnit")):
            return "location_issue"
        if any(d in dimensions for d in ("Carrier", "CarrierCode")):
            return "carrier_issue"
        return "unknown_issue"

    def _compute_severity(
        self,
        z_score: float,
        effect_size: float,
        is_bad: bool,
    ) -> str:
        """Compute severity based on Z-score and effect size."""
        if not is_bad:
            return "low"

        # Use combination of Z-score and effect size
        combined = (abs(z_score) + effect_size) / 2

        if combined >= 3.0:
            return "critical"
        elif combined >= 2.0:
            return "high"
        elif combined >= 1.0:
            return "medium"
        return "low"

    def _generate_recommendation(
        self,
        pattern_type: str,
        cohort_name: str,
        metric: str,
        direction: str,
        multiplier: float,
        severity: str,
    ) -> str:
        """Generate actionable recommendation for pattern."""
        metric_lower = metric.lower().replace("_", " ")
        multiplier_str = f"{multiplier:.1f}x" if multiplier < 100 else "significantly"

        recommendations = {
            "firmware_issue": f"Investigate firmware for {cohort_name}. Consider rollback or patch.",
            "model_issue": f"Review {cohort_name} hardware/driver configuration. Check for known issues.",
            "os_issue": f"Check OS compatibility for {cohort_name}. Review system requirements.",
            "location_issue": f"Investigate infrastructure at {cohort_name}. Check network coverage and environment.",
            "carrier_issue": f"Review carrier configuration for {cohort_name}. Check APN settings and coverage.",
        }

        base_rec = recommendations.get(
            pattern_type,
            f"Investigate {cohort_name} for {metric_lower} issues.",
        )

        if severity == "critical":
            return f"URGENT: {base_rec} {direction.title()} {metric_lower} ({multiplier_str} vs fleet)."
        return f"{base_rec} {direction.title()} {metric_lower} ({multiplier_str} vs fleet)."

    def _deduplicate_patterns(
        self,
        patterns: list[CohortPattern],
    ) -> list[CohortPattern]:
        """Remove duplicate patterns, keeping the most specific."""
        # Group by cohort_id and metric
        seen: dict[str, CohortPattern] = {}

        for pattern in patterns:
            key = f"{pattern.primary_metric}"

            # Keep the pattern with more specific cohort definition
            if key not in seen:
                seen[key] = pattern
            else:
                existing = seen[key]
                # Prefer more specific (more dimensions)
                if (
                    len(pattern.cohort_definition) > len(existing.cohort_definition)
                    or pattern.severity in ("critical", "high")
                    and existing.severity not in ("critical", "high")
                ):
                    seen[key] = pattern

        return list(seen.values())

    def _rank_patterns(
        self,
        patterns: list[CohortPattern],
    ) -> list[CohortPattern]:
        """Rank patterns by impact (severity × affected devices)."""
        severity_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}

        def impact_score(p: CohortPattern) -> float:
            severity_weight = severity_weights.get(p.severity, 1)
            return severity_weight * p.affected_devices * abs(p.deviation_z)

        patterns.sort(key=impact_score, reverse=True)
        return patterns

    # =========================================================================
    # Serialization
    # =========================================================================

    def save(self, path: Path) -> None:
        """Save detector to file."""
        import pickle

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "fleet_baselines": self.fleet_baselines,
                    "fleet_size": self.fleet_size,
                    "config": {
                        "cohort_dimensions": self.cohort_dimensions,
                        "pattern_metrics": self.pattern_metrics,
                        "significance_level": self.significance_level,
                        "min_effect_size": self.min_effect_size,
                        "min_sample_size": self.min_sample_size,
                        "min_fleet_percentage": self.min_fleet_percentage,
                        "z_threshold": self.z_threshold,
                    },
                    "fitted": self._fitted,
                },
                f,
            )
        logger.info(f"Saved CrossDevicePatternDetector to {path}")

    @classmethod
    def load(cls, path: Path) -> CrossDevicePatternDetector:
        """Load detector from file."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        detector = cls(**data["config"])
        detector.fleet_baselines = data["fleet_baselines"]
        detector.fleet_size = data["fleet_size"]
        detector._fitted = data["fitted"]

        logger.info(f"Loaded CrossDevicePatternDetector from {path}")
        return detector


# =============================================================================
# Convenience Functions
# =============================================================================


def detect_cross_device_patterns(
    df: pd.DataFrame,
    min_devices: int = 20,
    z_threshold: float = 2.0,
) -> list[CohortPattern]:
    """
    Convenience function to detect cross-device patterns.

    Args:
        df: DataFrame with device telemetry
        min_devices: Minimum devices per cohort
        z_threshold: Z-score threshold for flagging

    Returns:
        List of detected patterns
    """
    detector = CrossDevicePatternDetector(
        min_sample_size=min_devices,
        z_threshold=z_threshold,
    )
    detector.fit(df)
    return detector.detect_patterns(df)


def get_systemic_issues(
    df: pd.DataFrame,
    min_devices: int = 10,
    min_z: float = 2.0,
) -> list[SystemicIssue]:
    """
    Convenience function to get systemic issues.

    Args:
        df: DataFrame with device telemetry
        min_devices: Minimum devices per cohort
        min_z: Minimum Z-score to flag

    Returns:
        List of systemic issues
    """
    detector = CrossDevicePatternDetector(
        min_sample_size=min_devices,
        z_threshold=min_z,
    )
    detector.fit(df)
    return detector.detect_systemic_issues(df, min_devices=min_devices, min_z=min_z)
