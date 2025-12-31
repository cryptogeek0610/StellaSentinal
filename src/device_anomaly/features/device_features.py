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
from typing import List, Optional, Set

import numpy as np
import pandas as pd

from device_anomaly.config.feature_config import FeatureConfig

logger = logging.getLogger(__name__)


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
        rolling_windows: Optional[List[int]] = None,
        compute_derived: bool = True,
        compute_volatility: bool = True,
        min_periods: int = 3,
    ):
        """
        Initialize the feature builder.

        Args:
            window: Default rolling window length in days
            rolling_windows: List of window sizes for rolling stats (default: [7, 14, 30])
            compute_derived: Whether to compute derived efficiency features
            compute_volatility: Whether to compute volatility (CV) features
            min_periods: Minimum periods for rolling calculations
        """
        self.window = window
        self.rolling_windows = rolling_windows or [7, 14, 30]
        self.compute_derived = compute_derived
        self.compute_volatility = compute_volatility
        self.min_periods = min_periods

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
                if len(df) >= 7:
                    df[f"{col}_trend_7d"] = (
                        df[col] - df[col].rolling(7, min_periods=3).mean()
                    ) / (df[col].rolling(7, min_periods=3).std() + 1e-6)

        return df

    def _get_numeric_feature_cols(self, df: pd.DataFrame) -> List[str]:
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
        # Required cohort columns
        required = {"ManufacturerId", "ModelId", "OsVersionId"}
        if not required.issubset(df.columns):
            logger.warning("Missing cohort columns, skipping cohort z-scores")
            return df

        df = df.copy()

        # Build cohort identifier
        cohort_parts = [
            df["ManufacturerId"].astype(str),
            df["ModelId"].astype(str),
            df["OsVersionId"].astype(str),
        ]
        if "FirmwareVersion" in df.columns:
            cohort_parts.append(df["FirmwareVersion"].fillna("na").astype(str))

        cohort_id = cohort_parts[0]
        for part in cohort_parts[1:]:
            cohort_id = cohort_id + "_" + part
        df["cohort_id"] = cohort_id.astype(str)

        # Get numeric columns to normalize
        exclude_cols = FeatureConfig.excluded_columns | {"cohort_id"}
        numeric_cols = [
            c for c in df.columns
            if c not in exclude_cols
            and np.issubdtype(df[c].dtype, np.number)
            and not c.endswith("_cohort_z")  # Don't double-normalize
        ]

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
            # Normalize to 0-1 scale
            max_sev = df["DisconnectSeverity"].quantile(0.99) + 1
            risk_components.append((df["DisconnectSeverity"] / max_sev).clip(0, 1))
        if "BatteryDrainPerHour" in df.columns:
            # High drain = risk
            max_drain = df["BatteryDrainPerHour"].quantile(0.99) + 1
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
