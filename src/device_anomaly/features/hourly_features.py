"""
Hourly Granularity Feature Engineering for Anomaly Detection.

This module transforms hourly telemetry data into ML features including:
- Peak usage hour detection
- Hourly usage entropy (distribution patterns)
- Night vs business hours ratios
- Weekend vs weekday patterns
- Hourly rolling statistics

Data Sources:
- XSight: cs_DataUsageByHour (104M rows), cs_BatteryLevelDrop (14.8M rows), cs_WifiHour (755K rows)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HourlyFeatureBuilder:
    """
    Hourly granularity feature engineering for device anomaly detection.

    Transforms hourly telemetry data into ML-ready features:
    - Temporal distribution features (peak hours, entropy)
    - Business/night/weekend patterns
    - Hourly aggregations and rolling statistics
    - Consistency metrics (CV of hourly values)
    """

    def __init__(
        self,
        business_hour_start: int = 9,
        business_hour_end: int = 17,
        night_hour_start: int = 22,
        night_hour_end: int = 6,
        rolling_windows: list[int] | None = None,
    ):
        """
        Initialize the hourly feature builder.

        Args:
            business_hour_start: Start of business hours (9 AM)
            business_hour_end: End of business hours (5 PM)
            night_hour_start: Start of night hours (10 PM)
            night_hour_end: End of night hours (6 AM)
            rolling_windows: Rolling window sizes in hours (default: [6, 12, 24, 48])
        """
        self.business_hour_start = business_hour_start
        self.business_hour_end = business_hour_end
        self.night_hour_start = night_hour_start
        self.night_hour_end = night_hour_end
        self.rolling_windows = rolling_windows or [6, 12, 24, 48]

    def transform(
        self,
        df: pd.DataFrame,
        metric_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Build hourly features from time-series data.

        Expects columns like:
          - DeviceId
          - Hour (0-23) or Timestamp
          - CollectedDate (optional, for daily aggregation)
          - Metric columns (e.g., BatteryDrop, Download, Upload, ConnectionTime)

        Args:
            df: DataFrame with hourly telemetry data
            metric_columns: Columns to compute hourly features for

        Returns DataFrame with hourly features added.
        """
        if df.empty:
            return df

        df = df.copy()

        # Normalize hour column
        df = self._ensure_hour_column(df)

        # Get default metric columns if not specified
        if metric_columns is None:
            metric_columns = self._get_default_metric_columns(df)

        if not metric_columns:
            logger.warning("No metric columns found for hourly feature extraction")
            return df

        # Add temporal classification features
        logger.info("Computing temporal classification features...")
        df = self._add_temporal_classification(df)

        # Add peak hour features per device
        logger.info("Computing peak hour features...")
        df = self._add_peak_hour_features(df, metric_columns)

        # Add temporal distribution features
        logger.info("Computing temporal distribution features...")
        df = self._add_temporal_distribution_features(df, metric_columns)

        # Add rolling statistics
        logger.info("Computing rolling statistics...")
        df = self._add_rolling_statistics(df, metric_columns)

        # Add consistency metrics
        logger.info("Computing consistency metrics...")
        df = self._add_consistency_metrics(df, metric_columns)

        return df

    def _ensure_hour_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure Hour column exists."""
        if "Hour" in df.columns:
            return df

        # Try to extract from timestamp columns
        for col in ["Timestamp", "CollectedTime", "DateTime"]:
            if col in df.columns:
                df["Hour"] = pd.to_datetime(df[col], errors="coerce").dt.hour
                return df

        logger.warning("No Hour or timestamp column found")
        return df

    def _get_default_metric_columns(self, df: pd.DataFrame) -> list[str]:
        """Get default metric columns for hourly analysis."""
        candidates = [
            "BatteryDrop",
            "BatteryLevelDrop",
            "TotalBatteryLevelDrop",
            "Download",
            "Upload",
            "TotalDownload",
            "TotalUpload",
            "ConnectionTime",
            "DisconnectCount",
            "TotalDropCnt",
            "AppForegroundTime",
            "ScreenOnTime",
            "WifiSignalStrength",
            "AvgSignalStrength",
        ]
        return [c for c in candidates if c in df.columns]

    def _add_temporal_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal classification columns."""
        if "Hour" not in df.columns:
            return df

        hour = df["Hour"]

        # Business hours flag
        df["is_business_hour"] = (
            (hour >= self.business_hour_start) & (hour <= self.business_hour_end)
        ).astype(int)

        # Night hours flag (handles wrap-around)
        df["is_night_hour"] = (
            (hour >= self.night_hour_start) | (hour <= self.night_hour_end)
        ).astype(int)

        # Weekend flag (if we have date information)
        if "CollectedDate" in df.columns:
            df["_date"] = pd.to_datetime(df["CollectedDate"], errors="coerce")
            df["is_weekend"] = df["_date"].dt.dayofweek.isin([5, 6]).astype(int)
            df = df.drop(columns=["_date"], errors="ignore")

        return df

    def _add_peak_hour_features(
        self,
        df: pd.DataFrame,
        metric_columns: list[str],
    ) -> pd.DataFrame:
        """Add peak hour detection features per device."""
        if "DeviceId" not in df.columns or "Hour" not in df.columns:
            return df

        # For each metric, find peak usage hour per device
        for col in metric_columns[:5]:  # Limit to top 5 metrics
            if col not in df.columns:
                continue

            peak_hours = df.groupby(["DeviceId", "Hour"])[col].sum().reset_index()
            peak_hour_per_device = peak_hours.loc[peak_hours.groupby("DeviceId")[col].idxmax()][
                ["DeviceId", "Hour"]
            ]
            peak_hour_per_device.columns = ["DeviceId", f"{col}_peak_hour"]

            df = df.merge(peak_hour_per_device, on="DeviceId", how="left")

        return df

    def _add_temporal_distribution_features(
        self,
        df: pd.DataFrame,
        metric_columns: list[str],
    ) -> pd.DataFrame:
        """Add temporal distribution features per device."""
        if "DeviceId" not in df.columns:
            return df

        def compute_distribution(grp: pd.DataFrame) -> pd.Series:
            result = {}

            for col in metric_columns[:5]:
                if col not in grp.columns:
                    continue

                total = grp[col].sum()
                if total == 0:
                    result[f"{col}_business_ratio"] = 0.5
                    result[f"{col}_night_ratio"] = 0.0
                    result[f"{col}_entropy"] = 0.0
                    continue

                # Business hours ratio
                if "is_business_hour" in grp.columns:
                    business_total = grp[grp["is_business_hour"] == 1][col].sum()
                    result[f"{col}_business_ratio"] = business_total / total

                # Night ratio
                if "is_night_hour" in grp.columns:
                    night_total = grp[grp["is_night_hour"] == 1][col].sum()
                    result[f"{col}_night_ratio"] = night_total / total

                # Weekend ratio
                if "is_weekend" in grp.columns:
                    weekend_total = grp[grp["is_weekend"] == 1][col].sum()
                    result[f"{col}_weekend_ratio"] = weekend_total / total

                # Hourly entropy (distribution across hours)
                if "Hour" in grp.columns:
                    hourly = grp.groupby("Hour")[col].sum()
                    hourly_pct = hourly / (total + 1e-6)
                    entropy = -np.sum(hourly_pct * np.log2(hourly_pct + 1e-10))
                    # Normalize to 0-1 (max entropy for 24 hours is log2(24))
                    result[f"{col}_entropy"] = entropy / np.log2(24)

            return pd.Series(result)

        distribution_features = df.groupby("DeviceId").apply(compute_distribution).reset_index()
        df = df.merge(distribution_features, on="DeviceId", how="left")

        return df

    def _add_rolling_statistics(
        self,
        df: pd.DataFrame,
        metric_columns: list[str],
    ) -> pd.DataFrame:
        """Add rolling window statistics."""
        if "DeviceId" not in df.columns:
            return df

        # Sort by device and time
        sort_cols = ["DeviceId"]
        if "Timestamp" in df.columns:
            sort_cols.append("Timestamp")
        elif "CollectedDate" in df.columns and "Hour" in df.columns:
            sort_cols.extend(["CollectedDate", "Hour"])

        df = df.sort_values(sort_cols).reset_index(drop=True)

        # Add rolling statistics for each window size
        for window in self.rolling_windows:
            for col in metric_columns[:3]:  # Top 3 metrics to limit columns
                if col not in df.columns:
                    continue

                # Rolling mean within device
                df[f"{col}_roll_{window}h_mean"] = df.groupby("DeviceId")[col].transform(
                    lambda x, w=window: x.rolling(window=w, min_periods=1).mean()
                )

                # Rolling std
                df[f"{col}_roll_{window}h_std"] = df.groupby("DeviceId")[col].transform(
                    lambda x, w=window: x.rolling(window=w, min_periods=2).std()
                )

        return df

    def _add_consistency_metrics(
        self,
        df: pd.DataFrame,
        metric_columns: list[str],
    ) -> pd.DataFrame:
        """Add consistency metrics (coefficient of variation) per device."""
        if "DeviceId" not in df.columns:
            return df

        def compute_consistency(grp: pd.DataFrame) -> pd.Series:
            result = {}

            for col in metric_columns[:5]:
                if col not in grp.columns:
                    continue

                values = grp[col].dropna()
                if len(values) < 3:
                    result[f"{col}_cv"] = np.nan
                    result[f"{col}_consistency"] = np.nan
                    continue

                mean_val = values.mean()
                std_val = values.std()

                # Coefficient of variation
                cv = std_val / (mean_val + 1e-6) if mean_val > 0 else 0
                result[f"{col}_cv"] = cv

                # Consistency score (inverse of CV, normalized)
                result[f"{col}_consistency"] = 1 / (1 + cv)

            return pd.Series(result)

        consistency_features = df.groupby("DeviceId").apply(compute_consistency).reset_index()
        df = df.merge(consistency_features, on="DeviceId", how="left")

        return df


def build_hourly_features(
    df: pd.DataFrame,
    metric_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convenience function to build hourly features.

    Args:
        df: DataFrame with hourly telemetry data
        metric_columns: Columns to compute features for

    Returns:
        DataFrame with hourly features added
    """
    builder = HourlyFeatureBuilder()
    return builder.transform(df, metric_columns)


def aggregate_hourly_to_daily(
    df: pd.DataFrame,
    metric_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate hourly data to daily level with rich statistics.

    Useful for merging with daily training data.

    Args:
        df: DataFrame with hourly data
        metric_columns: Columns to aggregate

    Returns:
        DataFrame aggregated to device-day level
    """
    if df.empty or "DeviceId" not in df.columns:
        return df

    # Ensure we have a date column
    if "CollectedDate" not in df.columns:
        if "Timestamp" in df.columns:
            df["CollectedDate"] = pd.to_datetime(df["Timestamp"]).dt.date
        else:
            logger.warning("No date column available for daily aggregation")
            return df

    # Get metric columns
    if metric_columns is None:
        metric_columns = [
            c
            for c in df.columns
            if df[c].dtype in ["float64", "int64", "float32", "int32"]
            and c not in ["DeviceId", "Hour"]
        ]

    # Build aggregation dictionary
    agg_dict = {}
    for col in metric_columns[:10]:  # Limit columns
        if col not in df.columns:
            continue
        agg_dict[col] = ["sum", "mean", "max", "min", "std"]

    if not agg_dict:
        return df

    # Aggregate
    daily = df.groupby(["DeviceId", "CollectedDate"]).agg(agg_dict)
    daily.columns = [f"{col}_{stat}_hourly" for col, stat in daily.columns]
    daily = daily.reset_index()

    return daily


def get_hourly_feature_names() -> list[str]:
    """Get list of hourly feature names that this module generates."""
    base_features = [
        "is_business_hour",
        "is_night_hour",
        "is_weekend",
    ]

    # Template features (applied per metric)

    # Rolling features (per metric, per window)

    return base_features
