"""
Temporal Feature Engineering for Anomaly Detection.

This module provides advanced time-series features including:
- Seasonal-Trend decomposition using LOESS (STL)
- Trend slope and change point detection
- Seasonality pattern features
- Residual-based anomaly signals
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TemporalFeatureBuilder:
    """
    Temporal feature engineering for device anomaly detection.

    Uses statsmodels STL decomposition to extract:
    - Trend components and slopes
    - Seasonal patterns
    - Residuals (potential anomaly signals)
    """

    def __init__(
        self,
        seasonal_period: int = 24,  # Hourly data with daily seasonality
        trend_window: int = 7,
        min_observations: int = 48,  # At least 2 days of data
    ):
        """
        Initialize the temporal feature builder.

        Args:
            seasonal_period: Number of observations per seasonal cycle
            trend_window: Window for trend calculations in days
            min_observations: Minimum observations needed for decomposition
        """
        self.seasonal_period = seasonal_period
        self.trend_window = trend_window
        self.min_observations = min_observations

    def transform(
        self,
        df: pd.DataFrame,
        target_columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Build temporal features for specified columns.

        Args:
            df: DataFrame with time-series data
            target_columns: Columns to decompose (defaults to key metrics)

        Returns DataFrame with temporal features added.
        """
        if df.empty or "DeviceId" not in df.columns or "Timestamp" not in df.columns:
            return df

        df = df.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        # Default target columns
        if target_columns is None:
            target_columns = self._get_default_target_columns(df)

        # Add STL decomposition features for each target
        for col in target_columns:
            if col in df.columns:
                logger.info(f"Computing STL decomposition for {col}...")
                df = self._add_stl_features(df, col)

        # Add general temporal pattern features
        logger.info("Computing temporal pattern features...")
        df = self._add_temporal_pattern_features(df)

        # Add change point detection
        logger.info("Computing change point features...")
        df = self._add_change_point_features(df, target_columns)

        return df

    def _get_default_target_columns(self, df: pd.DataFrame) -> list[str]:
        """Get default columns for temporal decomposition."""
        candidates = [
            "TotalBatteryLevelDrop",
            "AppForegroundTime",
            "Download",
            "Upload",
            "TotalDropCnt",
            "CrashCount",
            "cpu_usage",
            "ram_utilization",
        ]
        return [c for c in candidates if c in df.columns]

    def _add_stl_features(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Add STL decomposition features for a single column."""
        try:
            from statsmodels.tsa.seasonal import STL
        except ImportError:
            logger.warning("statsmodels not installed, skipping STL decomposition")
            return df

        def decompose_device(grp: pd.DataFrame) -> pd.DataFrame:
            grp = grp.sort_values("Timestamp").copy()

            if len(grp) < self.min_observations:
                # Not enough data for decomposition
                grp[f"{column}_trend"] = np.nan
                grp[f"{column}_seasonal"] = np.nan
                grp[f"{column}_residual"] = np.nan
                grp[f"{column}_residual_zscore"] = np.nan
                return grp

            # Prepare time series
            series = pd.to_numeric(grp[column], errors="coerce")

            # Fill NaN with interpolation for STL
            if series.isna().sum() > len(series) * 0.5:
                # Too many missing values
                grp[f"{column}_trend"] = np.nan
                grp[f"{column}_seasonal"] = np.nan
                grp[f"{column}_residual"] = np.nan
                grp[f"{column}_residual_zscore"] = np.nan
                return grp

            series = series.interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")

            try:
                # Perform STL decomposition
                stl = STL(
                    series,
                    period=self.seasonal_period,
                    robust=True,
                )
                result = stl.fit()

                grp[f"{column}_trend"] = result.trend.values
                grp[f"{column}_seasonal"] = result.seasonal.values
                grp[f"{column}_residual"] = result.resid.values

                # Residual z-score (normalized residual)
                resid_std = result.resid.std()
                if resid_std > 1e-6:
                    grp[f"{column}_residual_zscore"] = result.resid / resid_std
                else:
                    grp[f"{column}_residual_zscore"] = 0.0

            except Exception as e:
                logger.warning(f"STL decomposition failed for {column}: {e}")
                grp[f"{column}_trend"] = np.nan
                grp[f"{column}_seasonal"] = np.nan
                grp[f"{column}_residual"] = np.nan
                grp[f"{column}_residual_zscore"] = np.nan

            return grp

        df = df.groupby("DeviceId", group_keys=False).apply(decompose_device)
        return df.reset_index(drop=True)

    def _add_temporal_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on temporal patterns."""
        if "Timestamp" not in df.columns:
            return df

        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        # Hour of day patterns
        df["hour"] = df["Timestamp"].dt.hour

        # Expected behavior based on hour (business hours vs off-hours)
        df["is_business_hour"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype(int)
        df["is_night_hour"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)

        # Day patterns
        df["day_of_week"] = df["Timestamp"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Hour-of-day cyclical encoding for ML
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Day-of-week cyclical encoding
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Time since last observation per device
        if "DeviceId" in df.columns:
            df = df.sort_values(["DeviceId", "Timestamp"])
            df["time_since_last"] = df.groupby("DeviceId")["Timestamp"].diff().dt.total_seconds()
            df["time_since_last_hours"] = df["time_since_last"] / 3600

            # Long gaps might indicate device issues
            df["is_long_gap"] = (df["time_since_last_hours"] > 24).astype(int)

        return df

    def _add_change_point_features(
        self,
        df: pd.DataFrame,
        target_columns: list[str],
    ) -> pd.DataFrame:
        """Add change point detection features."""
        if "DeviceId" not in df.columns or "Timestamp" not in df.columns:
            return df

        def detect_change_points(grp: pd.DataFrame, column: str) -> pd.DataFrame:
            grp = grp.sort_values("Timestamp").copy()

            if len(grp) < 10 or column not in grp.columns:
                grp[f"{column}_change_point_score"] = np.nan
                return grp

            series = pd.to_numeric(grp[column], errors="coerce")

            # Simple change point detection using CUSUM-like approach
            # Compare rolling mean before/after each point
            window = min(self.trend_window * 24, len(grp) // 4)
            if window < 3:
                grp[f"{column}_change_point_score"] = np.nan
                return grp

            before_mean = series.rolling(window=window, min_periods=3).mean().shift(1)
            after_mean = series.iloc[::-1].rolling(window=window, min_periods=3).mean().iloc[::-1]

            # Change magnitude
            change = (after_mean - before_mean).abs()
            series_std = series.std()

            if series_std > 1e-6:
                grp[f"{column}_change_point_score"] = change / series_std
            else:
                grp[f"{column}_change_point_score"] = 0.0

            return grp

        for col in target_columns:
            if col in df.columns:
                df = df.groupby("DeviceId", group_keys=False).apply(
                    lambda grp: detect_change_points(grp, col)
                )

        # Aggregate change point scores
        change_cols = [c for c in df.columns if c.endswith("_change_point_score")]
        if change_cols:
            # Max change point score across all metrics
            df["max_change_point_score"] = df[change_cols].max(axis=1)

        return df.reset_index(drop=True)


def build_temporal_features(
    df: pd.DataFrame,
    target_columns: Optional[list[str]] = None,
    seasonal_period: int = 24,
) -> pd.DataFrame:
    """
    Convenience function to build temporal features.

    Args:
        df: DataFrame with time-series data
        target_columns: Columns for STL decomposition
        seasonal_period: Observations per seasonal cycle

    Returns:
        DataFrame with temporal features added
    """
    builder = TemporalFeatureBuilder(seasonal_period=seasonal_period)
    return builder.transform(df, target_columns)


def get_temporal_feature_names(column: str) -> list[str]:
    """Get list of temporal feature names for a given column."""
    return [
        f"{column}_trend",
        f"{column}_seasonal",
        f"{column}_residual",
        f"{column}_residual_zscore",
        f"{column}_change_point_score",
    ]


def get_general_temporal_feature_names() -> list[str]:
    """Get list of general temporal feature names."""
    return [
        "hour",
        "is_business_hour",
        "is_night_hour",
        "day_of_week",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "time_since_last",
        "time_since_last_hours",
        "is_long_gap",
        "max_change_point_score",
    ]
