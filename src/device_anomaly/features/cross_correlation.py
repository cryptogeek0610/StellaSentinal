"""
Cross-Correlation Feature Engineering Module.

Implements advanced cross-domain correlation features that capture
known relationships between device attributes and behavior patterns:

1. OEM Version + Signal: Firmware versions with known WiFi bugs
2. Model + Battery Drain: Device models with defective batteries
3. OS Version + Crash Rate: OS updates causing app instability
4. Location + Signal: Geographic signal quality patterns

These features help distinguish between:
- Device-specific anomalies (something wrong with THIS device)
- Cohort-level issues (known bug affecting ALL devices of this type)
- Environmental factors (location-based signal issues)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# COHORT ISSUE TRACKER
# =============================================================================


@dataclass
class CohortIssue:
    """Represents a known issue affecting a device cohort."""

    cohort_id: str
    issue_type: str  # "firmware_wifi", "battery_defect", "os_crash", "location_signal"
    severity: float  # 0.0 to 1.0
    affected_metric: str
    description: str
    first_detected: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    device_count: int = 0
    confidence: float = 0.0  # Statistical confidence in the issue


class CohortIssueTracker:
    """
    Tracks known issues at the cohort level.

    When many devices of the same type (Model + OS + Firmware) show
    similar anomalous behavior, it's likely a cohort-level issue,
    not individual device problems.

    Usage:
        tracker = CohortIssueTracker()

        # Register observed issues
        tracker.register_observation(
            cohort_id="Samsung_SM-G991B_13_G991BXXU5DVJB",
            metric="WifiDropCount",
            value=45.0,  # High disconnect count
        )

        # Check if a device's behavior is explained by known issues
        context = tracker.get_issue_context(cohort_id, features)
    """

    def __init__(
        self,
        min_devices_for_issue: int = 5,
        z_score_threshold: float = 2.0,
        lookback_days: int = 30,
    ):
        self.min_devices_for_issue = min_devices_for_issue
        self.z_score_threshold = z_score_threshold
        self.lookback_days = lookback_days

        # Track per-cohort statistics
        self._cohort_stats: dict[str, dict[str, _RunningStats]] = {}

        # Detected issues
        self._issues: dict[str, list[CohortIssue]] = {}

        # Device observations per cohort (for counting unique devices)
        self._cohort_devices: dict[str, set[int]] = {}

    def register_observation(
        self,
        cohort_id: str,
        device_id: int,
        metric: str,
        value: float,
    ) -> CohortIssue | None:
        """
        Register an observation and detect potential cohort-level issues.

        Returns a CohortIssue if a new issue was detected.
        """
        if cohort_id not in self._cohort_stats:
            self._cohort_stats[cohort_id] = {}
            self._cohort_devices[cohort_id] = set()

        if metric not in self._cohort_stats[cohort_id]:
            self._cohort_stats[cohort_id][metric] = _RunningStats()

        # Update stats
        stats = self._cohort_stats[cohort_id][metric]
        stats.update(value)
        self._cohort_devices[cohort_id].add(device_id)

        # Check for issue detection
        return self._detect_issue(cohort_id, metric, value, stats)

    def _detect_issue(
        self,
        cohort_id: str,
        metric: str,
        value: float,
        stats: _RunningStats,
    ) -> CohortIssue | None:
        """Detect if current observations indicate a cohort-level issue."""
        # Need minimum observations
        if stats.count < self.min_devices_for_issue:
            return None

        device_count = len(self._cohort_devices[cohort_id])
        if device_count < self.min_devices_for_issue:
            return None

        # Check if mean is significantly elevated compared to global
        # (This requires comparison with global stats - simplified here)
        mean = stats.mean
        std = stats.std

        # Heuristic: if the cohort's mean is high AND consistent (low std),
        # it suggests a cohort-level issue rather than random anomalies
        cv = std / (abs(mean) + 1e-6)  # Coefficient of variation

        # Detect issue if: high mean, consistent behavior (low CV)
        issue_type = self._classify_issue(metric)
        if issue_type and cv < 0.5:  # Consistent behavior across cohort
            issue = CohortIssue(
                cohort_id=cohort_id,
                issue_type=issue_type,
                severity=min(1.0, mean / 100),  # Normalize
                affected_metric=metric,
                description=f"Cohort shows elevated {metric} (mean={mean:.2f}, n={device_count})",
                device_count=device_count,
                confidence=1.0 - cv,
            )

            if cohort_id not in self._issues:
                self._issues[cohort_id] = []
            self._issues[cohort_id].append(issue)

            logger.info(
                "Detected cohort issue: %s - %s (confidence=%.2f)",
                cohort_id,
                issue_type,
                issue.confidence,
            )

            return issue

        return None

    def _classify_issue(self, metric: str) -> str | None:
        """Classify issue type based on affected metric."""
        metric_lower = metric.lower()

        # WiFi/Firmware issues
        if any(x in metric_lower for x in ["wifi", "drop", "disconnect", "signal"]):
            return "firmware_wifi"

        # Battery issues
        if any(x in metric_lower for x in ["battery", "drain", "charge"]):
            return "battery_defect"

        # OS/Crash issues
        if any(x in metric_lower for x in ["crash", "anr", "error", "forcestop"]):
            return "os_crash"

        return None

    def get_issue_context(
        self,
        cohort_id: str,
        features: dict[str, float],
    ) -> dict[str, Any]:
        """
        Get issue context for a device's features.

        Returns information about known cohort-level issues that might
        explain the device's behavior.
        """
        issues = self._issues.get(cohort_id, [])

        if not issues:
            return {
                "has_known_issues": False,
                "issues": [],
                "adjustment_factors": {},
            }

        adjustment_factors = {}
        relevant_issues = []

        for issue in issues:
            if issue.affected_metric in features:
                relevant_issues.append(
                    {
                        "type": issue.issue_type,
                        "metric": issue.affected_metric,
                        "severity": issue.severity,
                        "confidence": issue.confidence,
                        "device_count": issue.device_count,
                        "description": issue.description,
                    }
                )

                # Suggest adjustment factor for anomaly scoring
                # If it's a known cohort issue, reduce the anomaly weight
                adjustment_factors[issue.affected_metric] = max(
                    0.3,  # Minimum weight (don't completely ignore)
                    1.0 - (issue.confidence * issue.severity),
                )

        return {
            "has_known_issues": len(relevant_issues) > 0,
            "issues": relevant_issues,
            "adjustment_factors": adjustment_factors,
        }

    def get_all_issues(self) -> list[CohortIssue]:
        """Get all detected cohort issues."""
        all_issues = []
        for issues in self._issues.values():
            all_issues.extend(issues)
        return all_issues


class _RunningStats:
    """Welford's online algorithm for running mean/variance."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / self.count if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        return np.sqrt(self.variance)


# =============================================================================
# CROSS-CORRELATION FEATURE BUILDER
# =============================================================================


class CrossCorrelationFeatureBuilder:
    """
    Builds advanced cross-correlation features that capture
    relationships between device attributes and behavior patterns.

    Features:
    1. Firmware-Signal Interaction: OEM version × WiFi behavior
    2. Model-Battery Interaction: Device model × battery patterns
    3. OS-Stability Interaction: OS version × crash patterns
    4. Location-Signal Interaction: Geographic × signal quality

    Usage:
        builder = CrossCorrelationFeatureBuilder()
        df_enhanced = builder.transform(df)
    """

    def __init__(
        self,
        issue_tracker: CohortIssueTracker | None = None,
        enable_known_issue_features: bool = True,
        enable_interaction_features: bool = True,
        enable_relative_features: bool = True,
    ):
        self.issue_tracker = issue_tracker or CohortIssueTracker()
        self.enable_known_issue_features = enable_known_issue_features
        self.enable_interaction_features = enable_interaction_features
        self.enable_relative_features = enable_relative_features

        # Known problematic firmware versions (would be loaded from database in production)
        self._known_firmware_issues: dict[str, dict[str, float]] = {}

        # Known model-specific battery issues
        self._known_battery_models: dict[str, float] = {}

        # Known OS stability issues
        self._known_os_issues: dict[str, float] = {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all cross-correlation transformations."""
        df = df.copy()

        logger.info("Building cross-correlation features for %d rows...", len(df))

        # 1. Firmware + Signal correlation
        df = self._add_firmware_signal_features(df)

        # 2. Model + Battery correlation
        df = self._add_model_battery_features(df)

        # 3. OS + Crash correlation
        df = self._add_os_crash_features(df)

        # 4. Location + Signal correlation (if location data available)
        df = self._add_location_signal_features(df)

        # 5. Known issue adjustment features
        if self.enable_known_issue_features:
            df = self._add_known_issue_features(df)

        # 6. Cross-domain interaction features
        if self.enable_interaction_features:
            df = self._add_interaction_features(df)

        logger.info("Cross-correlation features complete: %d columns", len(df.columns))

        return df

    # =========================================================================
    # 1. FIRMWARE + SIGNAL CORRELATION
    # =========================================================================

    def _add_firmware_signal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features correlating firmware version with WiFi/signal behavior.

        Captures: "This firmware version has known WiFi issues"
        """
        # Check for required columns
        firmware_cols = ["FirmwareVersion", "OEMVersion", "OsVersionId"]
        signal_cols = ["WifiDropCount", "WifiDisconnectCount", "TotalDropCnt", "AvgSignalStrength"]

        has_firmware = any(c in df.columns for c in firmware_cols)
        has_signal = any(c in df.columns for c in signal_cols)

        if not has_firmware or not has_signal:
            return df

        # Create firmware identifier
        if "FirmwareVersion" in df.columns:
            firmware_id = df["FirmwareVersion"].fillna("unknown").astype(str)
        elif "OEMVersion" in df.columns:
            firmware_id = df["OEMVersion"].fillna("unknown").astype(str)
        else:
            firmware_id = df["OsVersionId"].fillna(0).astype(str)

        # Compute per-firmware WiFi statistics
        if "WifiDropCount" in df.columns:
            # Firmware-specific drop rate
            firmware_drop_mean = df.groupby(firmware_id)["WifiDropCount"].transform("mean")
            firmware_drop_std = df.groupby(firmware_id)["WifiDropCount"].transform("std").fillna(0)

            # Device's deviation from firmware norm
            df["WifiDrop_vs_Firmware"] = (df["WifiDropCount"] - firmware_drop_mean) / (
                firmware_drop_std + 1e-6
            )

            # Is this firmware problematic overall?
            global_drop_mean = df["WifiDropCount"].mean()
            df["Firmware_WifiIssue_Flag"] = (firmware_drop_mean > global_drop_mean * 1.5).astype(
                int
            )

        if "TotalDropCnt" in df.columns:
            firmware_total_drop_mean = df.groupby(firmware_id)["TotalDropCnt"].transform("mean")
            global_total_drop_mean = df["TotalDropCnt"].mean()

            # Firmware connectivity quality score (0 = problematic, 1 = good)
            df["Firmware_ConnectivityScore"] = 1.0 - np.clip(
                firmware_total_drop_mean / (global_total_drop_mean * 2 + 1e-6), 0, 1
            )

        if "AvgSignalStrength" in df.columns:
            # Signal-to-drop ratio by firmware
            if "TotalDropCnt" in df.columns:
                signal_quality = df["AvgSignalStrength"] + 100  # Convert dBm to positive
                df["Firmware_SignalDropRatio"] = df["TotalDropCnt"] / (signal_quality + 1)

                # Compare to firmware average
                fw_avg_ratio = df.groupby(firmware_id)["Firmware_SignalDropRatio"].transform("mean")
                df["SignalDropRatio_vs_Firmware"] = df["Firmware_SignalDropRatio"] - fw_avg_ratio

        return df

    # =========================================================================
    # 2. MODEL + BATTERY CORRELATION
    # =========================================================================

    def _add_model_battery_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features correlating device model with battery behavior.

        Captures: "This device model has known battery issues"
        """
        model_cols = ["ModelId", "Model", "ModelName"]
        battery_cols = ["TotalBatteryLevelDrop", "BatteryDrainPerHour", "CalculatedBatteryCapacity"]

        has_model = any(c in df.columns for c in model_cols)
        has_battery = any(c in df.columns for c in battery_cols)

        if not has_model or not has_battery:
            return df

        # Create model identifier
        for col in model_cols:
            if col in df.columns:
                model_id = df[col].fillna("unknown").astype(str)
                break

        if "TotalBatteryLevelDrop" in df.columns:
            # Model-specific drain rate
            model_drain_mean = df.groupby(model_id)["TotalBatteryLevelDrop"].transform("mean")
            model_drain_std = (
                df.groupby(model_id)["TotalBatteryLevelDrop"].transform("std").fillna(0)
            )

            # Device's deviation from model norm
            df["BatteryDrain_vs_Model"] = (df["TotalBatteryLevelDrop"] - model_drain_mean) / (
                model_drain_std + 1e-6
            )

            # Is this model a battery hog?
            global_drain_mean = df["TotalBatteryLevelDrop"].mean()
            df["Model_BatteryIssue_Flag"] = (model_drain_mean > global_drain_mean * 1.5).astype(int)

            # Model battery efficiency score
            df["Model_BatteryEfficiency"] = 1.0 - np.clip(
                model_drain_mean / (global_drain_mean * 2 + 1e-6), 0, 1
            )

        if "CalculatedBatteryCapacity" in df.columns:
            # Model-specific capacity degradation
            model_capacity_mean = df.groupby(model_id)["CalculatedBatteryCapacity"].transform(
                "mean"
            )

            # Is this device's battery worse than other same-model devices?
            df["Capacity_vs_Model"] = (df["CalculatedBatteryCapacity"] - model_capacity_mean) / (
                model_capacity_mean + 1e-6
            )

        return df

    # =========================================================================
    # 3. OS VERSION + CRASH CORRELATION
    # =========================================================================

    def _add_os_crash_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features correlating OS version with app stability.

        Captures: "This OS version causes apps to crash"
        """
        os_cols = ["OsVersionId", "OSVersion", "AndroidApiLevel"]
        crash_cols = ["CrashCount", "ANRCount", "ForceStopCount"]

        has_os = any(c in df.columns for c in os_cols)
        has_crash = any(c in df.columns for c in crash_cols)

        if not has_os or not has_crash:
            return df

        # Create OS identifier
        for col in os_cols:
            if col in df.columns:
                os_id = df[col].fillna("unknown").astype(str)
                break

        # Combine crash metrics
        crash_total = pd.Series(0.0, index=df.index)
        for col in crash_cols:
            if col in df.columns:
                crash_total += df[col].fillna(0)

        if crash_total.sum() > 0:
            df["TotalCrashEvents"] = crash_total

            # OS-specific crash rate
            os_crash_mean = df.groupby(os_id)["TotalCrashEvents"].transform("mean")
            os_crash_std = df.groupby(os_id)["TotalCrashEvents"].transform("std").fillna(0)

            # Device's deviation from OS norm
            df["Crashes_vs_OS"] = (df["TotalCrashEvents"] - os_crash_mean) / (os_crash_std + 1e-6)

            # Is this OS version unstable?
            global_crash_mean = df["TotalCrashEvents"].mean()
            df["OS_StabilityIssue_Flag"] = (os_crash_mean > global_crash_mean * 1.5).astype(int)

            # OS stability score
            df["OS_StabilityScore"] = 1.0 - np.clip(
                os_crash_mean / (global_crash_mean * 2 + 1e-6), 0, 1
            )

        return df

    # =========================================================================
    # 4. LOCATION + SIGNAL CORRELATION
    # =========================================================================

    def _add_location_signal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features correlating location with signal quality.

        Captures: "This location has poor signal coverage"
        """
        # Location columns (if available)
        location_cols = ["Latitude", "Longitude", "LocationId", "SiteId", "BuildingId"]
        signal_cols = ["AvgSignalStrength", "CellSignalStrength", "WifiSignalStrength"]

        has_location = any(c in df.columns for c in location_cols)
        has_signal = any(c in df.columns for c in signal_cols)

        if not has_location or not has_signal:
            # No location data - add placeholder features
            df["Location_SignalQuality"] = 0.5  # Neutral
            df["Signal_vs_Location"] = 0.0  # No deviation
            return df

        # Create location identifier
        for col in location_cols:
            if col in df.columns:
                if col in ["Latitude", "Longitude"]:
                    # Grid-based location (0.01 degree = ~1km)
                    if "Latitude" in df.columns and "Longitude" in df.columns:
                        lat_grid = (df["Latitude"].fillna(0) * 100).astype(int)
                        lon_grid = (df["Longitude"].fillna(0) * 100).astype(int)
                        location_id = lat_grid.astype(str) + "_" + lon_grid.astype(str)
                        break
                else:
                    location_id = df[col].fillna("unknown").astype(str)
                    break

        # Signal quality by location
        for signal_col in signal_cols:
            if signal_col not in df.columns:
                continue

            # Location-specific signal stats
            loc_signal_mean = df.groupby(location_id)[signal_col].transform("mean")
            loc_signal_std = df.groupby(location_id)[signal_col].transform("std").fillna(0)

            # Device's signal vs location norm
            df[f"{signal_col}_vs_Location"] = (df[signal_col] - loc_signal_mean) / (
                loc_signal_std + 1e-6
            )

            # Is this location a dead zone?
            global_signal_mean = df[signal_col].mean()

            # Convert dBm to quality score (higher = better)
            loc_quality = (loc_signal_mean + 100) / 100  # -100 dBm = 0, 0 dBm = 1
            df["Location_SignalQuality"] = np.clip(loc_quality, 0, 1)

            # Location environment flag
            df["Location_PoorSignal_Flag"] = (loc_signal_mean < global_signal_mean - 10).astype(int)

            break  # Use first available signal column

        return df

    # =========================================================================
    # 5. KNOWN ISSUE ADJUSTMENT FEATURES
    # =========================================================================

    def _add_known_issue_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features based on known cohort-level issues.

        These features help the model understand when anomalous behavior
        is expected due to known issues.
        """
        # Build cohort identifier
        cohort_parts = []
        for col in ["ManufacturerId", "ModelId", "OsVersionId", "FirmwareVersion"]:
            if col in df.columns:
                cohort_parts.append(df[col].fillna("na").astype(str))

        if not cohort_parts:
            return df

        cohort_id = cohort_parts[0]
        for part in cohort_parts[1:]:
            cohort_id = cohort_id + "_" + part

        df["cohort_id"] = cohort_id

        # Count known issues per cohort
        df["Cohort_KnownIssueCount"] = 0
        df["Cohort_IssueAdjustment"] = 1.0  # Default: no adjustment

        # Check for firmware issues
        if "Firmware_WifiIssue_Flag" in df.columns:
            df["Cohort_KnownIssueCount"] += df["Firmware_WifiIssue_Flag"]

        if "Model_BatteryIssue_Flag" in df.columns:
            df["Cohort_KnownIssueCount"] += df["Model_BatteryIssue_Flag"]

        if "OS_StabilityIssue_Flag" in df.columns:
            df["Cohort_KnownIssueCount"] += df["OS_StabilityIssue_Flag"]

        if "Location_PoorSignal_Flag" in df.columns:
            df["Cohort_KnownIssueCount"] += df["Location_PoorSignal_Flag"]

        # Adjustment factor: reduce anomaly weight for known issues
        # More issues = lower weight (behavior is "expected")
        df["Cohort_IssueAdjustment"] = 1.0 / (1.0 + df["Cohort_KnownIssueCount"] * 0.3)

        return df

    # =========================================================================
    # 6. CROSS-DOMAIN INTERACTION FEATURES
    # =========================================================================

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features capturing interactions between domains.

        These capture complex patterns like:
        - "High drain + poor signal + old firmware = likely firmware bug"
        - "High crashes + new OS + specific model = OS compatibility issue"
        """
        # Firmware × Signal × Battery interaction
        if all(c in df.columns for c in ["Firmware_ConnectivityScore", "Model_BatteryEfficiency"]):
            df["Firmware_Battery_Interaction"] = (
                df["Firmware_ConnectivityScore"] * df["Model_BatteryEfficiency"]
            )

        # OS × Model × Stability interaction
        if all(c in df.columns for c in ["OS_StabilityScore", "Model_BatteryEfficiency"]):
            df["OS_Model_Stability"] = df["OS_StabilityScore"] * df["Model_BatteryEfficiency"]

        # Location × Firmware × Signal interaction
        if all(c in df.columns for c in ["Location_SignalQuality", "Firmware_ConnectivityScore"]):
            # If both location and firmware have issues, it's compounded
            df["Location_Firmware_Signal"] = (
                df["Location_SignalQuality"] * df["Firmware_ConnectivityScore"]
            )

        # Composite "Expected Behavior" score
        # High score = behavior is likely due to known issues, not device problem
        expected_scores = []

        if "Firmware_ConnectivityScore" in df.columns:
            # Invert: low connectivity score = more expected issues
            expected_scores.append(1.0 - df["Firmware_ConnectivityScore"])

        if "Model_BatteryEfficiency" in df.columns:
            expected_scores.append(1.0 - df["Model_BatteryEfficiency"])

        if "OS_StabilityScore" in df.columns:
            expected_scores.append(1.0 - df["OS_StabilityScore"])

        if "Location_SignalQuality" in df.columns:
            expected_scores.append(1.0 - df["Location_SignalQuality"])

        if expected_scores:
            df["KnownIssue_ExpectedScore"] = sum(expected_scores) / len(expected_scores)

            # Anomaly should be weighted by how "unexpected" it is
            df["Anomaly_Unexpectedness"] = 1.0 - df["KnownIssue_ExpectedScore"]

        return df


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def build_cross_correlation_features(
    df: pd.DataFrame,
    issue_tracker: CohortIssueTracker | None = None,
) -> pd.DataFrame:
    """
    Convenience function to add all cross-correlation features.

    Args:
        df: DataFrame with device telemetry and cohort columns
        issue_tracker: Optional tracker for known issues

    Returns:
        DataFrame with cross-correlation features added
    """
    builder = CrossCorrelationFeatureBuilder(issue_tracker=issue_tracker)
    return builder.transform(df)


def get_cohort_context(
    device_id: int,
    cohort_id: str,
    features: dict[str, float],
    issue_tracker: CohortIssueTracker,
) -> dict[str, Any]:
    """
    Get context about known cohort issues for a device.

    Args:
        device_id: Device identifier
        cohort_id: Cohort identifier (model+os+firmware)
        features: Current device features
        issue_tracker: The issue tracker

    Returns:
        Dictionary with issue context and adjustment factors
    """
    return issue_tracker.get_issue_context(cohort_id, features)
