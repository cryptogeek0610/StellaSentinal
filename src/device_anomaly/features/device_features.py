"""
Comprehensive Device Feature Engineering for ML Anomaly Detection.

This module transforms raw telemetry data into rich ML features including:
- Rolling window statistics (mean, std, min, max, median)
- Day-over-day deltas and trends
- Temporal context (hour, day, weekend, business hours)
- Cohort-aware z-scores (normalized by device type)
- Derived efficiency ratios (battery/MB, drops/hour, etc.)
- Cross-domain interaction features
- Volatility and stability metrics
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from device_anomaly.config.feature_config import FeatureConfig

logger = logging.getLogger(__name__)

FEATURE_SPEC_VERSION = "device_features_v1"

_DEFAULT_FEATURE_SPEC = {
    "version": FEATURE_SPEC_VERSION,
    "window": 14,
    "rolling_windows": [7, 14, 30],
    "min_periods": 3,
    "compute_derived": True,
    "compute_volatility": True,
    "compute_cohort": True,
}


class DeviceFeatureBuilder:
    """
    Comprehensive feature builder for device anomaly detection.

    Transforms raw telemetry into ML-ready features with:
    - Per-device rolling statistics (7, 14, 30 day windows)
    - Temporal context features
    - Cohort-normalized z-scores
    - Derived efficiency metrics
    - Cross-domain interaction features
    """

    def __init__(
        self,
        window: int = 14,
        rolling_windows: list[int] | None = None,
        hourly_windows: list[int] | None = None,
        compute_derived: bool = True,
        compute_volatility: bool = True,
        compute_cohort: bool = True,
        min_periods: int = 3,
        feature_norms: dict[str, float] | None = None,
    ):
        """
        Initialize the feature builder.

        Args:
            window: Default rolling window length in days
            rolling_windows: List of window sizes for rolling stats (default: [7, 14, 30])
            hourly_windows: List of hour-based window sizes for hourly data (e.g., [6, 12, 24, 48, 168])
                           Only used when data has 'Hour' column indicating hourly granularity
            compute_derived: Whether to compute derived efficiency features
            compute_volatility: Whether to compute volatility (CV) features
            compute_cohort: Whether to compute cohort z-scores in this pass
            min_periods: Minimum periods for rolling calculations
            feature_norms: Pre-computed feature normalization values
        """
        self.window = window
        self.rolling_windows = rolling_windows or [7, 14, 30]
        self.hourly_windows = hourly_windows or []  # e.g., [6, 12, 24, 48, 168] hours
        self.compute_derived = compute_derived
        self.compute_volatility = compute_volatility
        self.compute_cohort = compute_cohort
        self.min_periods = min_periods
        self.feature_norms = feature_norms or {}

    def get_feature_spec(self) -> dict[str, Any]:
        return {
            "version": FEATURE_SPEC_VERSION,
            "window": self.window,
            "rolling_windows": list(self.rolling_windows),
            "hourly_windows": list(self.hourly_windows),
            "min_periods": self.min_periods,
            "compute_derived": self.compute_derived,
            "compute_volatility": self.compute_volatility,
            "compute_cohort": self.compute_cohort,
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build comprehensive time-series features per device.

        Expects at minimum:
          - DeviceId
          - Timestamp
          - Numeric telemetry columns from DW/MC loaders

        Returns DataFrame with all engineered features.
        """
        logger.info(f"Starting feature engineering on {len(df):,} rows...")

        # Ensure we have required columns
        if "DeviceId" not in df.columns or "Timestamp" not in df.columns:
            logger.warning("Missing DeviceId or Timestamp column")
            return df

        # Step 1: Per-device rolling features
        logger.info("Computing per-device rolling features...")
        df_feat = df.groupby("DeviceId", group_keys=False).apply(
            self._per_device_features
        )
        df_feat = df_feat.reset_index(drop=True)

        # Step 2: Derived efficiency features
        if self.compute_derived:
            logger.info("Computing derived efficiency features...")
            df_feat = self._add_derived_features(df_feat)

        # Step 3: Cohort-aware z-scores
        if self.compute_cohort:
            logger.info("Computing cohort-aware z-scores...")
            df_feat = self._add_cohort_features(df_feat)

        # Step 4: Volatility features
        if self.compute_volatility:
            logger.info("Computing volatility features...")
            df_feat = self._add_volatility_features(df_feat)

        # Step 5: Cross-domain features
        logger.info("Computing cross-domain interaction features...")
        df_feat = self._add_cross_domain_features(df_feat)

        logger.info(f"Feature engineering complete: {len(df_feat.columns)} columns")
        return df_feat

    # Alias for backward compatibility
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alias for transform() for backward compatibility."""
        return self.transform(df)

    # =========================================================================
    # PER-DEVICE ROLLING FEATURES
    # =========================================================================

    def _per_device_features(self, df_dev: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-device time-series features.

        Applied to each device's data separately via groupby.
        """
        df_dev = df_dev.sort_values("Timestamp").copy()

        # --- Normalize column names ---
        df_dev = self._normalize_columns(df_dev)

        # --- Rolling features for all numeric columns in config ---
        df_dev = self._add_rolling_stats(df_dev)

        # --- Temporal context features ---
        df_dev = self._add_temporal_features(df_dev)

        # --- Delta (day-over-day change) features ---
        df_dev = self._add_delta_features(df_dev)

        return df_dev

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names for consistency."""
        # Download/Upload aliases
        if "Download" not in df.columns and "TotalDownload" in df.columns:
            df["Download"] = df["TotalDownload"]
        if "Upload" not in df.columns and "TotalUpload" in df.columns:
            df["Upload"] = df["TotalUpload"]

        return df

    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics for key numeric columns."""
        # Get numeric columns that should have rolling stats
        numeric_cols = self._get_numeric_feature_cols(df)

        for col in numeric_cols:
            if col not in df.columns:
                continue

            # Primary window (default 14 days)
            df[f"{col}_roll_mean"] = (
                df[col].rolling(window=self.window, min_periods=self.min_periods).mean()
            )
            df[f"{col}_roll_std"] = (
                df[col].rolling(window=self.window, min_periods=self.min_periods).std()
            )

            # Multi-window statistics for key features
            if col in FeatureConfig.rolling_feature_candidates:
                for win in self.rolling_windows:
                    if win != self.window:  # Skip if same as primary
                        df[f"{col}_roll_{win}d_mean"] = (
                            df[col].rolling(window=win, min_periods=self.min_periods).mean()
                        )
                        df[f"{col}_roll_{win}d_std"] = (
                            df[col].rolling(window=win, min_periods=self.min_periods).std()
                        )

                # Rolling min/max for key features
                df[f"{col}_roll_min"] = (
                    df[col].rolling(window=self.window, min_periods=self.min_periods).min()
                )
                df[f"{col}_roll_max"] = (
                    df[col].rolling(window=self.window, min_periods=self.min_periods).max()
                )
                df[f"{col}_roll_median"] = (
                    df[col].rolling(window=self.window, min_periods=self.min_periods).median()
                )

        # Hourly rolling windows (for hourly granularity data)
        if self.hourly_windows and "Hour" in df.columns:
            self._add_hourly_rolling_stats(df, numeric_cols)

        return df

    def _add_hourly_rolling_stats(self, df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
        """
        Add hourly rolling statistics for fine-grained anomaly detection.

        Only applied when data has hourly granularity (indicated by 'Hour' column).
        Uses smaller windows measured in hours instead of days.
        """
        logger.info(f"Computing hourly rolling features with windows: {self.hourly_windows}")

        # Limit to subset of columns to avoid explosion
        hourly_cols = [c for c in numeric_cols if c in FeatureConfig.rolling_feature_candidates][:15]

        for col in hourly_cols:
            if col not in df.columns:
                continue

            for hours in self.hourly_windows:
                # Mean and std for each hourly window
                df[f"{col}_roll_{hours}h_mean"] = (
                    df[col].rolling(window=hours, min_periods=max(2, hours // 4)).mean()
                )
                df[f"{col}_roll_{hours}h_std"] = (
                    df[col].rolling(window=hours, min_periods=max(2, hours // 4)).std()
                )

                # Only add min/max for smaller windows (6h, 12h) to limit feature count
                if hours <= 24:
                    df[f"{col}_roll_{hours}h_min"] = (
                        df[col].rolling(window=hours, min_periods=max(2, hours // 4)).min()
                    )
                    df[f"{col}_roll_{hours}h_max"] = (
                        df[col].rolling(window=hours, min_periods=max(2, hours // 4)).max()
                    )

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal context features."""
        ts = df["Timestamp"]

        # Hour of day
        df["hour_of_day"] = ts.dt.hour
        df["hour_of_day_norm"] = df["hour_of_day"] / 23.0

        # Day of week (0 = Monday, 6 = Sunday)
        df["day_of_week"] = ts.dt.dayofweek
        df["day_of_week_norm"] = df["day_of_week"] / 6.0

        # Weekend flag
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Business hours flag (9 AM - 5 PM on weekdays)
        df["is_business_hours"] = (
            (df["hour_of_day"] >= 9) &
            (df["hour_of_day"] < 17) &
            (df["day_of_week"] < 5)
        ).astype(int)

        # Additional temporal
        df["day_of_month"] = ts.dt.day
        df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
        df["month"] = ts.dt.month
        df["quarter"] = ts.dt.quarter

        # Cyclical encoding for better ML representation
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        return df

    def _add_delta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add day-over-day change (delta) features."""
        delta_cols = FeatureConfig.rolling_feature_candidates

        for col in delta_cols:
            if col in df.columns and np.issubdtype(df[col].dtype, np.number):
                # Simple delta
                df[f"{col}_delta"] = df[col].diff()

                # Percent change (with protection against division by zero)
                prev_val = df[col].shift(1)
                df[f"{col}_pct_change"] = df[col].sub(prev_val).div(prev_val.abs() + 1e-6)

                # 7-day trend (slope proxy)
                df[f"{col}_trend_7d"] = (
                    df[col] - df[col].rolling(7, min_periods=3).mean()
                ) / (df[col].rolling(7, min_periods=3).std() + 1e-6)

        return df

    def _get_numeric_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Get list of numeric columns that should have rolling stats."""
        # Start with config features
        candidates = set(FeatureConfig.genericFeatures)
        candidates.update(FeatureConfig.battery_features)
        candidates.update(FeatureConfig.app_usage_features)
        candidates.update(FeatureConfig.data_usage_features)
        candidates.update(FeatureConfig.rf_signal_features)
        candidates.update(FeatureConfig.connectivity_features)

        # Filter to columns that exist and are numeric
        numeric_cols = []
        for col in candidates:
            if col in df.columns and np.issubdtype(df[col].dtype, np.number):
                numeric_cols.append(col)

        return numeric_cols

    # =========================================================================
    # DERIVED EFFICIENCY FEATURES
    # =========================================================================

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived efficiency features from raw metrics.

        These capture cross-domain relationships like:
        - Battery drain per hour of usage
        - Data throughput per signal quality
        - Crash rate per app visit
        """
        df = df.copy()

        # ----- Battery Efficiency -----
        if all(c in df.columns for c in ["TotalBatteryLevelDrop", "TotalDischargeTime_Sec"]):
            discharge_hours = df["TotalDischargeTime_Sec"] / 3600 + 1
            df["BatteryDrainPerHour"] = df["TotalBatteryLevelDrop"] / discharge_hours

        if all(c in df.columns for c in ["TotalBatteryAppDrain", "AppForegroundTime"]):
            app_hours = df["AppForegroundTime"] / 3600 + 1
            df["BatteryDrainPerAppHour"] = df["TotalBatteryAppDrain"] / app_hours

        if all(c in df.columns for c in ["TotalBatteryLevelDrop", "Download", "Upload"]):
            data_mb = (df["Download"] + df["Upload"]) / 1e6 + 1
            df["BatteryDrainPerMB"] = df["TotalBatteryLevelDrop"] / data_mb

        if all(c in df.columns for c in ["ChargePatternGoodCount", "ChargePatternBadCount"]):
            total_charges = df["ChargePatternGoodCount"] + df["ChargePatternBadCount"] + 1
            df["ChargeQualityScore"] = df["ChargePatternGoodCount"] / total_charges

        if all(c in df.columns for c in ["WirelessChargeCount", "AcChargeCount", "UsbChargeCount"]):
            total = df["WirelessChargeCount"] + df["AcChargeCount"] + df["UsbChargeCount"] + 1
            df["WirelessChargePreference"] = df["WirelessChargeCount"] / total

        if all(c in df.columns for c in ["CalculatedBatteryCapacity", "DesignCapacity"]):
            design = df["DesignCapacity"].fillna(0) + 1
            df["BatteryHealthRatio"] = df["CalculatedBatteryCapacity"] / design

        if all(c in df.columns for c in ["ScreenOnTime_Sec", "ScreenOffTime_Sec"]):
            total = df["ScreenOnTime_Sec"] + df["ScreenOffTime_Sec"] + 1
            df["ScreenOnRatio"] = df["ScreenOnTime_Sec"] / total

        # ----- Network Efficiency -----
        if all(c in df.columns for c in ["Download", "Upload", "AvgSignalStrength"]):
            # Signal is in dBm (negative), add 100 to make positive
            signal_quality = df["AvgSignalStrength"] + 100
            df["DataPerSignalQuality"] = (df["Download"] + df["Upload"]) / (signal_quality + 1)

        if all(c in df.columns for c in ["TotalDropCnt", "AppForegroundTime"]):
            active_hours = df["AppForegroundTime"] / 3600 + 1
            df["DropsPerActiveHour"] = df["TotalDropCnt"] / active_hours

        if all(c in df.columns for c in ["TotalDropCnt", "TotalSignalReadings"]):
            df["DropRate"] = df["TotalDropCnt"] / (df["TotalSignalReadings"] + 1)
            df["ConnectionStabilityScore"] = 1 - df["DropRate"].clip(0, 1)

        if all(c in df.columns for c in ["DisconnectCount", "OfflineMinutes"]):
            df["DisconnectSeverity"] = df["DisconnectCount"] * df["OfflineMinutes"]

        if all(c in df.columns for c in ["MinSignalStrength", "MaxSignalStrength", "AvgSignalStrength"]):
            signal_range = df["MaxSignalStrength"] - df["MinSignalStrength"]
            df["SignalVariability"] = signal_range / (df["AvgSignalStrength"].abs() + 1)

        if all(c in df.columns for c in ["TimeOnWifi", "TimeOn4G", "TimeOn5G"]):
            total_time = df["TimeOn4G"] + df["TimeOn5G"] + df["TimeOnWifi"] + 1
            df["WifiVsCellRatio"] = df["TimeOnWifi"] / total_time

        # ----- Storage Utilization -----
        if all(c in df.columns for c in ["AvailableStorage", "TotalStorage"]):
            total = df["TotalStorage"].fillna(0) + 1
            df["StorageUtilization"] = 1 - (df["AvailableStorage"] / total)

        if all(c in df.columns for c in ["AvailableRAM", "TotalRAM"]):
            total = df["TotalRAM"].fillna(0) + 1
            df["RAMPressure"] = 1 - (df["AvailableRAM"] / total)

        if all(c in df.columns for c in ["AvailableInternalStorage", "TotalInternalStorage"]):
            total = df["TotalInternalStorage"].fillna(0) + 1
            df["InternalStorageUtilization"] = 1 - (df["AvailableInternalStorage"] / total)

        # ----- Usage Intensity -----
        if "AppForegroundTime" in df.columns:
            df["UsageIntensity"] = df["AppForegroundTime"] / (24 * 3600)  # Fraction of day

        if all(c in df.columns for c in ["UniqueAppsUsed", "AppVisitCount"]):
            df["AppDiversity"] = df["UniqueAppsUsed"] / (df["AppVisitCount"] + 1)

        if all(c in df.columns for c in ["CrashCount", "AppVisitCount"]):
            df["CrashRate"] = df["CrashCount"] / (df["AppVisitCount"] + 1)

        if all(c in df.columns for c in ["ANRCount", "AppVisitCount"]):
            df["ANRRate"] = df["ANRCount"] / (df["AppVisitCount"] + 1)

        if all(c in df.columns for c in ["WebErrorCount", "WebVisitCount"]):
            df["WebErrorsPerVisit"] = df["WebErrorCount"] / (df["WebVisitCount"] + 1)

        if all(c in df.columns for c in ["NotificationClickCount", "NotificationCount"]):
            df["NotificationEngagement"] = df["NotificationClickCount"] / (df["NotificationCount"] + 1)

        if all(c in df.columns for c in ["BackgroundTime", "AppForegroundTime"]):
            df["BackgroundVsForeground"] = df["BackgroundTime"] / (df["AppForegroundTime"] + 1)

        # ----- Data Usage Patterns -----
        if all(c in df.columns for c in ["Upload", "Download"]):
            df["UploadToDownloadRatio"] = df["Upload"] / (df["Download"] + 1)

        if all(c in df.columns for c in ["MobileDownload", "MobileUpload", "WifiDownload", "WifiUpload"]):
            mobile = df["MobileDownload"] + df["MobileUpload"]
            wifi = df["WifiDownload"] + df["WifiUpload"] + 1
            df["MobileToWifiDataRatio"] = mobile / wifi

        if all(c in df.columns for c in ["BackgroundDownload", "BackgroundUpload", "Download", "Upload"]):
            bg = df["BackgroundDownload"] + df["BackgroundUpload"]
            total = df["Download"] + df["Upload"] + 1
            df["BackgroundDataRatio"] = bg / total

        if all(c in df.columns for c in ["Download", "Upload", "UniqueAppsUsed"]):
            df["DataPerApp"] = (df["Download"] + df["Upload"]) / (df["UniqueAppsUsed"] + 1)

        return df

    # =========================================================================
    # COHORT-AWARE Z-SCORES
    # =========================================================================

    def _add_cohort_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add robust z-scores per device cohort (ManufacturerId, ModelId, OsVersionId).

        This enables detection of "weird for this device type" anomalies,
        not just global outliers.
        """
        from device_anomaly.features.cohort_stats import build_cohort_id, select_cohort_feature_cols

        cohort_id = build_cohort_id(df)
        if cohort_id is None:
            logger.warning("Missing cohort columns, skipping cohort z-scores")
            return df

        df = df.copy()
        df["cohort_id"] = cohort_id.astype(str)
        numeric_cols = select_cohort_feature_cols(df)

        # Compute cohort z-scores using robust statistics (median, MAD)
        for col in numeric_cols:
            grp = df.groupby("cohort_id")[col]

            median = grp.transform("median")
            mad = grp.transform(lambda x: (x - x.median()).abs().median())
            mad = mad.replace(0, np.nan)

            # Robust z-score: (value - median) / MAD
            df[f"{col}_cohort_z"] = (df[col] - median) / (mad + 1e-6)

        return df

    # =========================================================================
    # VOLATILITY FEATURES
    # =========================================================================

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility (coefficient of variation) features.

        CV = std / mean captures normalized volatility - useful for detecting
        erratic behavior patterns.
        """
        df = df.copy()

        # Compute CV for key features that have rolling stats
        key_features = FeatureConfig.rolling_feature_candidates

        for col in key_features:
            mean_col = f"{col}_roll_mean"
            std_col = f"{col}_roll_std"

            if mean_col in df.columns and std_col in df.columns:
                # Coefficient of variation
                df[f"{col}_cv"] = df[std_col] / (df[mean_col].abs() + 1e-6)

                # Clip extreme values
                df[f"{col}_cv"] = df[f"{col}_cv"].clip(-10, 10)

        return df

    # =========================================================================
    # CROSS-DOMAIN INTERACTION FEATURES
    # =========================================================================

    def _add_cross_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features that capture interactions between domains.

        These composite features often have high signal for anomaly detection.
        """
        df = df.copy()

        # Device Health Score (composite)
        health_components = []
        if "BatteryHealthRatio" in df.columns:
            health_components.append(df["BatteryHealthRatio"].fillna(0.5))
        if "StorageUtilization" in df.columns:
            # Invert so higher = healthier
            health_components.append(1 - df["StorageUtilization"].fillna(0.5))
        if "ConnectionStabilityScore" in df.columns:
            health_components.append(df["ConnectionStabilityScore"].fillna(0.5))
        if "CrashRate" in df.columns:
            # Invert so lower crash rate = healthier
            health_components.append(1 - df["CrashRate"].clip(0, 1).fillna(0))

        if health_components:
            df["DeviceHealthScore"] = sum(health_components) / len(health_components)

        # Anomaly Risk Score (composite of bad indicators)
        risk_components = []
        if "CrashRate" in df.columns:
            risk_components.append(df["CrashRate"].clip(0, 1).fillna(0))
        if "DisconnectSeverity" in df.columns:
            max_sev = self.feature_norms.get("disconnect_severity_p99")
            if max_sev is None:
                max_sev = df["DisconnectSeverity"].quantile(0.99)
            max_sev = float(max_sev) + 1
            risk_components.append((df["DisconnectSeverity"] / max_sev).clip(0, 1))
        if "BatteryDrainPerHour" in df.columns:
            # High drain = risk
            max_drain = self.feature_norms.get("battery_drain_per_hour_p99")
            if max_drain is None:
                max_drain = df["BatteryDrainPerHour"].quantile(0.99)
            max_drain = float(max_drain) + 1
            risk_components.append((df["BatteryDrainPerHour"] / max_drain).clip(0, 1))
        if "DropRate" in df.columns:
            risk_components.append(df["DropRate"].clip(0, 1).fillna(0))

        if risk_components:
            df["AnomalyRiskScore"] = sum(risk_components) / len(risk_components)

        # Battery-Network interaction
        if all(c in df.columns for c in ["BatteryDrainPerHour", "DropRate"]):
            df["BatteryNetworkStress"] = df["BatteryDrainPerHour"] * (1 + df["DropRate"])

        # Usage-Storage interaction
        if all(c in df.columns for c in ["UsageIntensity", "StorageUtilization"]):
            df["UsageStoragePressure"] = df["UsageIntensity"] * df["StorageUtilization"]

        return df


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_features(
    df: pd.DataFrame,
    window: int = 14,
    compute_all: bool = True,
    include_cross_correlation: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to build all features.

    Args:
        df: Raw telemetry DataFrame
        window: Rolling window size in days
        compute_all: Whether to compute all feature types
        include_cross_correlation: Whether to include cross-correlation features

    Returns:
        DataFrame with engineered features
    """
    builder = DeviceFeatureBuilder(
        window=window,
        compute_derived=compute_all,
        compute_volatility=compute_all,
    )
    df_features = builder.transform(df)

    # Add cross-correlation features
    if include_cross_correlation:
        try:
            from device_anomaly.features.cross_correlation import build_cross_correlation_features
            df_features = build_cross_correlation_features(df_features)
        except Exception as e:
            logger.warning("Cross-correlation features failed: %s", e)

    return df_features


def get_feature_summary(df: pd.DataFrame) -> dict:
    """
    Get a summary of available features in a DataFrame.

    Returns counts by feature type (raw, rolling, derived, cohort, etc.)
    """
    cols = df.columns.tolist()

    summary = {
        "total_columns": len(cols),
        "raw_features": len([c for c in cols if c in FeatureConfig.get_all_raw_features()]),
        "rolling_features": len([c for c in cols if "_roll_" in c]),
        "delta_features": len([c for c in cols if "_delta" in c or "_pct_change" in c]),
        "cohort_features": len([c for c in cols if "_cohort_z" in c]),
        "volatility_features": len([c for c in cols if "_cv" in c]),
        "temporal_features": len([c for c in cols if c in FeatureConfig.temporal_features]),
        "derived_features": len([c for c in cols if c in FeatureConfig.get_derived_feature_names()]),
    }

    # Count by domain
    domain_counts = {}
    for col in cols:
        domain = FeatureConfig.get_domain_for_feature(col)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    summary["by_domain"] = domain_counts

    return summary


def resolve_feature_spec(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = dict(_DEFAULT_FEATURE_SPEC)
    payload = (metadata or {}).get("feature_spec") if metadata else None
    if isinstance(payload, dict):
        spec.update({k: v for k, v in payload.items() if v is not None})

    spec["window"] = int(spec.get("window", _DEFAULT_FEATURE_SPEC["window"]))
    spec["min_periods"] = int(spec.get("min_periods", _DEFAULT_FEATURE_SPEC["min_periods"]))
    rolling = spec.get("rolling_windows", _DEFAULT_FEATURE_SPEC["rolling_windows"])
    spec["rolling_windows"] = [int(v) for v in list(rolling)]
    spec["compute_derived"] = bool(spec.get("compute_derived", True))
    spec["compute_volatility"] = bool(spec.get("compute_volatility", True))
    spec["compute_cohort"] = bool(spec.get("compute_cohort", True))
    spec["version"] = spec.get("version") or FEATURE_SPEC_VERSION
    return spec


def resolve_feature_norms(metadata: dict[str, Any] | None = None) -> dict[str, float]:
    norms = {}
    payload = (metadata or {}).get("feature_norms") if metadata else None
    if isinstance(payload, dict):
        for key, value in payload.items():
            if value is None:
                continue
            try:
                norms[key] = float(value)
            except (TypeError, ValueError):
                continue
    return norms


def build_feature_builder(
    feature_spec: dict[str, Any] | None = None,
    feature_norms: dict[str, float] | None = None,
    compute_cohort: bool | None = None,
) -> DeviceFeatureBuilder:
    spec = resolve_feature_spec({"feature_spec": feature_spec} if feature_spec else None)
    return DeviceFeatureBuilder(
        window=spec["window"],
        rolling_windows=spec["rolling_windows"],
        compute_derived=spec["compute_derived"],
        compute_volatility=spec["compute_volatility"],
        compute_cohort=spec["compute_cohort"] if compute_cohort is None else compute_cohort,
        min_periods=spec["min_periods"],
        feature_norms=feature_norms,
    )


def compute_feature_norms(df: pd.DataFrame) -> dict[str, float]:
    norms: dict[str, float] = {}
    if df.empty:
        return norms

    if "DisconnectSeverity" in df.columns:
        series = pd.to_numeric(df["DisconnectSeverity"], errors="coerce").dropna()
        if not series.empty:
            norms["disconnect_severity_p99"] = float(series.quantile(0.99))

    if "BatteryDrainPerHour" in df.columns:
        series = pd.to_numeric(df["BatteryDrainPerHour"], errors="coerce").dropna()
        if not series.empty:
            norms["battery_drain_per_hour_p99"] = float(series.quantile(0.99))

    return norms


def load_feature_metadata(models_dir: Path | None = None) -> dict[str, Any]:
    from device_anomaly.models.model_registry import load_latest_training_metadata

    metadata = load_latest_training_metadata(models_dir)
    return metadata or {}


def build_feature_builder_from_metadata(
    metadata: dict[str, Any] | None = None,
    compute_cohort: bool | None = None,
) -> DeviceFeatureBuilder:
    if not metadata:
        logger.warning("Feature metadata missing; falling back to default feature spec.")
    spec = resolve_feature_spec(metadata)
    norms = resolve_feature_norms(metadata)
    if not norms:
        logger.warning("Feature norms missing; cross-domain normalization will use dataset quantiles.")
    return build_feature_builder(feature_spec=spec, feature_norms=norms, compute_cohort=compute_cohort)


# =============================================================================
# REBOOT FEATURE INTEGRATION
# =============================================================================


def add_reboot_features(
    df: pd.DataFrame,
    start_dt: str,
    end_dt: str,
    device_id_column: str = "DeviceId",
) -> pd.DataFrame:
    """
    Add reboot-related features to a device DataFrame.

    Carl's requirement: "Devices with excessive reboots"

    This function:
    1. Loads reboot events from MobiControl (MainLog + LastBootTime changes)
    2. Aggregates per device: count, frequency, consecutive patterns
    3. Merges with the main feature DataFrame

    Args:
        df: Device DataFrame with DeviceId column
        start_dt: Start datetime for reboot search (ISO format)
        end_dt: End datetime for reboot search (ISO format)
        device_id_column: Name of the device ID column

    Returns:
        DataFrame with added reboot features:
        - reboot_count: Total reboots in period
        - reboot_rate_per_day: Reboots per day
        - avg_uptime_hours: Average time between reboots
        - consecutive_reboot_count: Reboots within 1 hour of each other (boot loops)
        - has_excessive_reboots: Flag for >3 reboots/week
        - has_boot_loop_pattern: Flag for consecutive reboots
    """
    if df.empty or device_id_column not in df.columns:
        return df

    try:
        from device_anomaly.data_access.mc_loader import aggregate_reboot_counts

        # Get device IDs to query
        device_ids = df[device_id_column].unique().tolist()

        # Load aggregated reboot counts
        reboot_df = aggregate_reboot_counts(start_dt, end_dt, device_ids)

        if reboot_df.empty:
            # No reboot data - add columns with defaults
            df = df.copy()
            df["reboot_count"] = 0
            df["reboot_rate_per_day"] = 0.0
            df["avg_uptime_hours"] = 0.0
            df["consecutive_reboot_count"] = 0
            df["has_excessive_reboots"] = False
            df["has_boot_loop_pattern"] = False
            logger.info("No reboot data found - added default reboot features")
            return df

        # Calculate period length for rate calculation
        from datetime import datetime
        start = datetime.fromisoformat(start_dt.replace("Z", "+00:00") if "Z" in start_dt else start_dt)
        end = datetime.fromisoformat(end_dt.replace("Z", "+00:00") if "Z" in end_dt else end_dt)
        period_days = max(1, (end - start).days)

        # Add derived features to reboot_df
        reboot_df["reboot_rate_per_day"] = reboot_df["reboot_count"] / period_days
        reboot_df["has_excessive_reboots"] = reboot_df["reboot_count"] > (period_days / 7 * 3)  # >3/week
        reboot_df["has_boot_loop_pattern"] = reboot_df["consecutive_reboot_count"] > 1

        # Select columns to merge
        merge_cols = [
            "DeviceId",
            "reboot_count",
            "reboot_rate_per_day",
            "avg_hours_between_reboots",
            "consecutive_reboot_count",
            "has_excessive_reboots",
            "has_boot_loop_pattern",
        ]
        reboot_df = reboot_df[[c for c in merge_cols if c in reboot_df.columns]]

        # Rename for consistency
        reboot_df = reboot_df.rename(columns={"avg_hours_between_reboots": "avg_uptime_hours"})

        # Merge with main DataFrame
        df = df.merge(reboot_df, left_on=device_id_column, right_on="DeviceId", how="left")

        # Fill NaN for devices without reboots
        fill_defaults = {
            "reboot_count": 0,
            "reboot_rate_per_day": 0.0,
            "avg_uptime_hours": 0.0,
            "consecutive_reboot_count": 0,
            "has_excessive_reboots": False,
            "has_boot_loop_pattern": False,
        }
        for col, default in fill_defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)

        # Clean up duplicate DeviceId column if created
        if "DeviceId_y" in df.columns:
            df = df.drop(columns=["DeviceId_y"])
        if "DeviceId_x" in df.columns:
            df = df.rename(columns={"DeviceId_x": device_id_column})

        logger.info(f"Added reboot features for {len(reboot_df):,} devices")
        return df

    except ImportError as e:
        logger.warning(f"Could not import reboot loader: {e}")
        return df
    except Exception as e:
        logger.warning(f"Failed to add reboot features: {e}")
        return df


# =============================================================================
# EXTENDED FEATURE INTEGRATION
# =============================================================================

def build_extended_features(
    df: pd.DataFrame,
    include_location: bool = True,
    include_events: bool = True,
    include_system_health: bool = True,
    include_temporal: bool = True,
    include_network_traffic: bool = True,
    include_security: bool = True,
    include_hourly: bool = False,
    window: int = 14,
) -> pd.DataFrame:
    """
    Build comprehensive features including all extended feature modules.

    This is the main entry point for full feature engineering including:
    - Core device features (rolling, derived, cohort)
    - Location features (mobility, dead zones, WiFi patterns)
    - Event features (crash patterns, alert severity)
    - System health features (CPU, RAM, storage, thermal)
    - Temporal features (STL decomposition, change points)
    - Network traffic features (per-app usage, exfiltration detection)
    - Security posture features (encryption, root detection, patch age)
    - Hourly granularity features (peak hours, temporal entropy)

    Args:
        df: Raw telemetry DataFrame
        include_location: Include location-based features
        include_events: Include event/log-based features
        include_system_health: Include system health features
        include_temporal: Include temporal decomposition features
        include_network_traffic: Include network traffic analysis features
        include_security: Include security posture features
        include_hourly: Include hourly granularity features
        window: Rolling window size in days

    Returns:
        DataFrame with all engineered features
    """
    logger.info(f"Building extended features on {len(df):,} rows...")

    # Step 1: Core device features
    builder = DeviceFeatureBuilder(
        window=window,
        compute_derived=True,
        compute_volatility=True,
        compute_cohort=True,
    )
    df_features = builder.transform(df)

    # Step 2: Location features (if data available)
    if include_location:
        try:
            from device_anomaly.features.location_features import LocationFeatureBuilder
            location_builder = LocationFeatureBuilder()
            df_features = location_builder.transform(df_features)
            logger.info("Added location features")
        except ImportError:
            logger.debug("Location features module not available")
        except Exception as e:
            logger.warning(f"Location features failed: {e}")

    # Step 3: Event features (if data available)
    if include_events:
        try:
            from device_anomaly.features.event_features import EventFeatureBuilder
            event_builder = EventFeatureBuilder(window_days=window)
            df_features = event_builder.transform(df_features)
            logger.info("Added event features")
        except ImportError:
            logger.debug("Event features module not available")
        except Exception as e:
            logger.warning(f"Event features failed: {e}")

    # Step 4: System health features (if data available)
    if include_system_health:
        try:
            from device_anomaly.features.system_health_features import SystemHealthFeatureBuilder
            health_builder = SystemHealthFeatureBuilder(window_days=window)
            df_features = health_builder.transform(df_features)
            logger.info("Added system health features")
        except ImportError:
            logger.debug("System health features module not available")
        except Exception as e:
            logger.warning(f"System health features failed: {e}")

    # Step 5: Temporal features (if sufficient time-series data)
    if include_temporal:
        try:
            from device_anomaly.features.temporal_features import TemporalFeatureBuilder
            temporal_builder = TemporalFeatureBuilder()
            df_features = temporal_builder.transform(df_features)
            logger.info("Added temporal features")
        except ImportError:
            logger.debug("Temporal features module not available")
        except Exception as e:
            logger.warning(f"Temporal features failed: {e}")

    # Step 6: Network traffic features (per-app usage, exfiltration detection)
    if include_network_traffic:
        try:
            from device_anomaly.features.network_traffic_features import (
                NetworkTrafficFeatureBuilder,
            )
            network_builder = NetworkTrafficFeatureBuilder()
            df_features = network_builder.transform(df_features)
            logger.info("Added network traffic features")
        except ImportError:
            logger.debug("Network traffic features module not available")
        except Exception as e:
            logger.warning(f"Network traffic features failed: {e}")

    # Step 7: Security posture features (encryption, root detection, patch age)
    if include_security:
        try:
            from device_anomaly.features.security_features import SecurityFeatureBuilder
            security_builder = SecurityFeatureBuilder()
            df_features = security_builder.transform(df_features)
            logger.info("Added security posture features")
        except ImportError:
            logger.debug("Security features module not available")
        except Exception as e:
            logger.warning(f"Security posture features failed: {e}")

    # Step 8: Hourly granularity features (peak hours, temporal entropy)
    if include_hourly:
        try:
            from device_anomaly.features.hourly_features import HourlyFeatureBuilder
            hourly_builder = HourlyFeatureBuilder()
            df_features = hourly_builder.transform(df_features)
            logger.info("Added hourly granularity features")
        except ImportError:
            logger.debug("Hourly features module not available")
        except Exception as e:
            logger.warning(f"Hourly features failed: {e}")

    # Step 9: Cross-correlation features
    try:
        from device_anomaly.features.cross_correlation import build_cross_correlation_features
        df_features = build_cross_correlation_features(df_features)
        logger.info("Added cross-correlation features")
    except ImportError:
        logger.debug("Cross-correlation module not available")
    except Exception as e:
        logger.warning(f"Cross-correlation features failed: {e}")

    logger.info(f"Extended feature engineering complete: {len(df_features.columns)} columns")
    return df_features


def get_extended_feature_summary(df: pd.DataFrame) -> dict:
    """
    Get detailed summary of available features including extended features.

    Returns counts by feature type and module.
    """
    cols = df.columns.tolist()

    summary = get_feature_summary(df)

    # Add extended feature counts
    summary["location_features"] = len([
        c for c in cols if any(x in c.lower() for x in [
            "location", "distance", "cluster", "dead_zone", "ap_", "wifi",
            "mobility", "entropy"
        ])
    ])

    summary["event_features"] = len([
        c for c in cols if any(x in c.lower() for x in [
            "event", "crash", "alert", "error_", "warning_", "log"
        ])
    ])

    summary["system_health_features"] = len([
        c for c in cols if any(x in c.lower() for x in [
            "cpu", "ram", "temp", "thermal", "health_score", "pressure"
        ])
    ])

    summary["temporal_decomposition_features"] = len([
        c for c in cols if any(x in c for x in [
            "_trend", "_seasonal", "_residual", "_change_point"
        ])
    ])

    # Network traffic features
    summary["network_traffic_features"] = len([
        c for c in cols if any(x in c.lower() for x in [
            "exfiltration", "upload_ratio", "download_ratio", "traffic_concentration",
            "interface_diversity", "interface_switching", "wifi_data_ratio",
            "cellular_data", "night_data_pct", "business_hour_data", "hourly_entropy"
        ])
    ])

    # Security posture features
    summary["security_features"] = len([
        c for c in cols if any(x in c.lower() for x in [
            "security_score", "is_rooted", "is_encrypted", "has_passcode",
            "risk_score", "risk_level", "patch_age", "attestation",
            "developer_risk", "is_compliant", "is_managed"
        ])
    ])

    # Hourly granularity features
    summary["hourly_features"] = len([
        c for c in cols if any(x in c for x in [
            "_peak_hour", "_business_ratio", "_night_ratio", "_weekend_ratio",
            "_consistency", "_roll_6h", "_roll_12h", "_roll_24h", "_roll_48h"
        ])
    ])

    return summary


# =============================================================================
# FEATURE SELECTION FOR HIGH DIMENSIONALITY
# =============================================================================



def select_features_for_training(
    df: pd.DataFrame,
    max_features: int = 200,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    exclude_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Select features for ML training to handle high dimensionality.

    When using extended features, location features, hourly data, and auto-discovery,
    the feature count can explode to 500+. This function reduces dimensionality by:

    1. Removing near-zero variance features (uninformative)
    2. Removing highly correlated feature pairs (redundant)
    3. Optionally limiting to top N features by variance

    Args:
        df: DataFrame with engineered features
        max_features: Maximum number of features to retain
        variance_threshold: Remove features with variance below this threshold (relative to max)
        correlation_threshold: Remove one of each feature pair with correlation above this
        exclude_columns: Columns to always exclude from selection (e.g., DeviceId, Timestamp)

    Returns:
        Tuple of (filtered DataFrame, list of selected feature names)

    Example:
        df_features = build_extended_features(df)
        df_selected, selected_cols = select_features_for_training(df_features, max_features=200)
        print(f"Reduced from {len(df_features.columns)} to {len(selected_cols)} features")
    """
    if exclude_columns is None:
        exclude_columns = {"DeviceId", "Timestamp", "CollectedDate", "CollectedTime"}

    logger.info(f"Feature selection: {len(df.columns)} columns, max_features={max_features}")

    # Identify numeric feature columns
    feature_cols = [
        c for c in df.columns
        if c not in exclude_columns
        and np.issubdtype(df[c].dtype, np.number)
    ]

    if len(feature_cols) == 0:
        logger.warning("No numeric feature columns found")
        return df, []

    df_features = df[feature_cols].copy()

    # Step 1: Remove near-zero variance features
    variances = df_features.var()
    max_variance = variances.max()
    if max_variance > 0:
        variance_cutoff = variance_threshold * max_variance
        low_var_cols = variances[variances < variance_cutoff].index.tolist()
        if low_var_cols:
            logger.info(f"Removing {len(low_var_cols)} low-variance features")
            df_features = df_features.drop(columns=low_var_cols)

    # Step 2: Remove highly correlated features
    if len(df_features.columns) > 1:
        corr_matrix = df_features.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        )
        to_drop = [
            col for col in upper_triangle.columns
            if any(upper_triangle[col] > correlation_threshold)
        ]
        if to_drop:
            logger.info(f"Removing {len(to_drop)} highly-correlated features")
            df_features = df_features.drop(columns=to_drop)

    # Step 3: If still too many, keep top N by variance
    if len(df_features.columns) > max_features:
        variances = df_features.var().sort_values(ascending=False)
        top_features = variances.head(max_features).index.tolist()
        logger.info(f"Limiting to top {max_features} features by variance")
        df_features = df_features[top_features]

    selected_features = df_features.columns.tolist()

    # Reconstruct DataFrame with non-feature columns preserved
    non_feature_cols = [c for c in df.columns if c in exclude_columns]
    result_df = pd.concat([df[non_feature_cols], df_features], axis=1)

    logger.info(f"Feature selection complete: {len(selected_features)} features retained")
    return result_df, selected_features


def get_feature_importance_estimate(
    df: pd.DataFrame,
    target_col: str | None = None,
    method: str = "variance",
) -> pd.Series:
    """
    Estimate feature importance without model training.

    Provides quick feature ranking for selection/debugging.

    Args:
        df: Feature DataFrame
        target_col: Optional target column for supervised ranking
        method: Ranking method - "variance", "correlation", or "both"

    Returns:
        Series with feature names as index and importance scores as values
    """
    exclude_cols = {"DeviceId", "Timestamp", "CollectedDate"}
    numeric_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and np.issubdtype(df[c].dtype, np.number)
    ]

    if method == "variance":
        # Rank by variance (higher = potentially more informative)
        importance = df[numeric_cols].var()
        importance = importance / importance.max()  # Normalize to 0-1

    elif method == "correlation" and target_col and target_col in df.columns:
        # Rank by correlation with target
        importance = df[numeric_cols].corrwith(df[target_col]).abs()

    else:
        # Combined: variance * (1 - self-correlation clustering)
        importance = df[numeric_cols].var()
        importance = importance / importance.max()

    return importance.sort_values(ascending=False)
