"""
System Health Feature Engineering for Anomaly Detection.

This module transforms CPU, RAM, storage, and temperature data into ML features:
- CPU usage patterns and spikes
- RAM pressure and memory events
- Storage utilization and exhaustion forecasting
- Thermal events and throttling risk
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SystemHealthFeatureBuilder:
    """
    System health feature engineering for device anomaly detection.

    Transforms DeviceStatInt and system metrics into ML-ready features:
    - CPU spike detection and patterns
    - RAM pressure and low memory events
    - Storage exhaustion forecasting
    - Thermal throttling risk assessment
    """

    def __init__(
        self,
        cpu_spike_threshold: float = 90.0,
        ram_pressure_threshold: float = 0.85,
        temp_warning_threshold: float = 45.0,
        storage_critical_threshold: float = 0.90,
        window_days: int = 7,
    ):
        """
        Initialize the system health feature builder.

        Args:
            cpu_spike_threshold: CPU % above which is considered a spike
            ram_pressure_threshold: RAM utilization above which is pressure
            temp_warning_threshold: Temperature (C) above which is warning
            storage_critical_threshold: Storage utilization above which is critical
            window_days: Rolling window for health statistics
        """
        self.cpu_spike_threshold = cpu_spike_threshold
        self.ram_pressure_threshold = ram_pressure_threshold
        self.temp_warning_threshold = temp_warning_threshold
        self.storage_critical_threshold = storage_critical_threshold
        self.window_days = window_days

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build system health features per device.

        Expects columns like:
          - DeviceId
          - Timestamp
          - CPU/CpuUsage (optional)
          - RAM/AvailableRAM/TotalRAM (optional)
          - Storage/AvailableStorage/TotalStorage (optional)
          - Temperature/DeviceTemp (optional)

        Returns DataFrame with system health features added.
        """
        if df.empty or "DeviceId" not in df.columns:
            return df

        df = df.copy()

        # Add CPU features if available
        if self._has_cpu_data(df):
            logger.info("Computing CPU health features...")
            df = self._add_cpu_features(df)

        # Add RAM features if available
        if self._has_ram_data(df):
            logger.info("Computing RAM pressure features...")
            df = self._add_ram_features(df)

        # Add storage features if available
        if self._has_storage_data(df):
            logger.info("Computing storage health features...")
            df = self._add_storage_features(df)

        # Add temperature features if available
        if self._has_temp_data(df):
            logger.info("Computing thermal features...")
            df = self._add_thermal_features(df)

        # Add composite health score
        logger.info("Computing composite system health score...")
        df = self._add_composite_health_score(df)

        return df

    def _has_cpu_data(self, df: pd.DataFrame) -> bool:
        """Check if CPU data is available."""
        return any(col in df.columns for col in ["CpuUsage", "CPU", "cpu_usage", "CPUPercentage"])

    def _has_ram_data(self, df: pd.DataFrame) -> bool:
        """Check if RAM data is available."""
        return any(col in df.columns for col in [
            "AvailableRAM", "TotalRAM", "RAM", "MemoryUsage", "available_ram", "FreeMemory"
        ])

    def _has_storage_data(self, df: pd.DataFrame) -> bool:
        """Check if storage data is available."""
        return any(col in df.columns for col in [
            "AvailableStorage", "TotalStorage", "StorageUsage", "FreeStorage", "available_storage"
        ])

    def _has_temp_data(self, df: pd.DataFrame) -> bool:
        """Check if temperature data is available."""
        return any(col in df.columns for col in [
            "Temperature", "DeviceTemp", "BatteryTemp", "CPUTemp", "temperature"
        ])

    def _find_column(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        """Find first matching column from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _add_cpu_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add CPU usage features."""
        cpu_col = self._find_column(df, ["CpuUsage", "CPU", "cpu_usage", "CPUPercentage"])
        if cpu_col is None:
            return df

        # Ensure numeric
        df["cpu_value"] = pd.to_numeric(df[cpu_col], errors="coerce")

        # CPU spike detection
        df["is_cpu_spike"] = (df["cpu_value"] >= self.cpu_spike_threshold).astype(int)

        # Per-device CPU statistics
        cpu_stats = df.groupby("DeviceId").agg(
            cpu_usage_avg=("cpu_value", "mean"),
            cpu_usage_max=("cpu_value", "max"),
            cpu_usage_p50=("cpu_value", "median"),
            cpu_usage_p95=("cpu_value", lambda x: x.quantile(0.95) if len(x) > 0 else np.nan),
            cpu_spike_count=("is_cpu_spike", "sum"),
            cpu_spike_rate=("is_cpu_spike", "mean"),
        ).reset_index()

        df = df.merge(cpu_stats, on="DeviceId", how="left")

        # Rolling CPU statistics
        if "Timestamp" in df.columns:
            df = df.sort_values(["DeviceId", "Timestamp"])
            df["cpu_roll_mean"] = df.groupby("DeviceId")["cpu_value"].transform(
                lambda x: x.rolling(window=self.window_days * 24, min_periods=3).mean()  # Assuming hourly data
            )
            df["cpu_roll_std"] = df.groupby("DeviceId")["cpu_value"].transform(
                lambda x: x.rolling(window=self.window_days * 24, min_periods=3).std()
            )

            # CPU volatility (coefficient of variation)
            df["cpu_volatility"] = df["cpu_roll_std"] / (df["cpu_roll_mean"] + 1e-6)

        # Cleanup
        df = df.drop(columns=["cpu_value"], errors="ignore")

        return df

    def _add_ram_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RAM pressure features."""
        avail_col = self._find_column(df, ["AvailableRAM", "available_ram", "FreeMemory"])
        total_col = self._find_column(df, ["TotalRAM", "total_ram", "TotalMemory"])

        if avail_col is None:
            return df

        df["ram_available"] = pd.to_numeric(df[avail_col], errors="coerce")

        # Calculate utilization if total available
        if total_col is not None:
            df["ram_total"] = pd.to_numeric(df[total_col], errors="coerce")
            df["ram_utilization"] = 1 - (df["ram_available"] / (df["ram_total"] + 1e-6))
        else:
            # Use available as proxy (lower = more pressure)
            df["ram_utilization"] = np.nan

        # RAM pressure detection
        if "ram_utilization" in df.columns:
            df["is_ram_pressure"] = (df["ram_utilization"] >= self.ram_pressure_threshold).astype(int)
        else:
            # Low available RAM threshold (e.g., < 500MB)
            df["is_ram_pressure"] = (df["ram_available"] < 500e6).astype(int)

        # Per-device RAM statistics
        agg_cols = {
            "ram_pressure_rate": ("is_ram_pressure", "mean"),
            "ram_pressure_count": ("is_ram_pressure", "sum"),
        }
        if "ram_utilization" in df.columns and not df["ram_utilization"].isna().all():
            agg_cols["ram_utilization_avg"] = ("ram_utilization", "mean")
            agg_cols["ram_utilization_max"] = ("ram_utilization", "max")

        ram_stats = df.groupby("DeviceId").agg(**agg_cols).reset_index()
        df = df.merge(ram_stats, on="DeviceId", how="left")

        # Cleanup
        df = df.drop(columns=["ram_available", "ram_total"], errors="ignore")

        return df

    def _add_storage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add storage utilization and forecasting features."""
        avail_col = self._find_column(df, ["AvailableStorage", "available_storage", "FreeStorage"])
        total_col = self._find_column(df, ["TotalStorage", "total_storage", "TotalDiskSpace"])

        if avail_col is None:
            return df

        df["storage_available"] = pd.to_numeric(df[avail_col], errors="coerce")

        if total_col is not None:
            df["storage_total"] = pd.to_numeric(df[total_col], errors="coerce")
            df["storage_utilization"] = 1 - (df["storage_available"] / (df["storage_total"] + 1e-6))
        else:
            df["storage_utilization"] = np.nan

        # Critical storage detection
        if "storage_utilization" in df.columns:
            df["is_storage_critical"] = (df["storage_utilization"] >= self.storage_critical_threshold).astype(int)
        else:
            # Less than 1GB free
            df["is_storage_critical"] = (df["storage_available"] < 1e9).astype(int)

        # Per-device storage statistics
        agg_cols = {
            "storage_critical_rate": ("is_storage_critical", "mean"),
            "storage_critical_count": ("is_storage_critical", "sum"),
        }
        if "storage_utilization" in df.columns and not df["storage_utilization"].isna().all():
            agg_cols["storage_utilization_avg"] = ("storage_utilization", "mean")
            agg_cols["storage_utilization_max"] = ("storage_utilization", "max")

        storage_stats = df.groupby("DeviceId").agg(**agg_cols).reset_index()
        df = df.merge(storage_stats, on="DeviceId", how="left")

        # Storage exhaustion forecasting
        if "Timestamp" in df.columns and not df["storage_available"].isna().all():
            df = self._add_storage_forecast(df)

        # Cleanup
        df = df.drop(columns=["storage_available", "storage_total"], errors="ignore")

        return df

    def _add_storage_forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forecast days until storage exhaustion."""
        def forecast_days(grp: pd.DataFrame) -> float:
            if len(grp) < 5:
                return np.nan

            # Sort by time
            sorted_grp = grp.sort_values("Timestamp")
            storage = sorted_grp["storage_available"].values
            times = pd.to_datetime(sorted_grp["Timestamp"])

            # Skip if constant
            if storage.std() < 1e-6:
                return np.nan

            # Days since start
            days = (times - times.min()).dt.total_seconds() / 86400
            days = days.values

            try:
                # Linear regression
                slope, intercept = np.polyfit(days, storage, 1)

                if slope >= 0:
                    # Storage increasing or stable
                    return np.nan

                # Days until zero
                current = storage[-1]
                days_until_zero = -current / slope
                return max(0, days_until_zero)

            except Exception:
                return np.nan

        forecast = df.groupby("DeviceId").apply(forecast_days).reset_index()
        forecast.columns = ["DeviceId", "storage_days_until_full"]
        df = df.merge(forecast, on="DeviceId", how="left")

        return df

    def _add_thermal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add thermal/temperature features."""
        temp_col = self._find_column(df, ["Temperature", "DeviceTemp", "BatteryTemp", "CPUTemp", "temperature"])
        if temp_col is None:
            return df

        df["temp_value"] = pd.to_numeric(df[temp_col], errors="coerce")

        # Thermal warning detection
        df["is_thermal_warning"] = (df["temp_value"] >= self.temp_warning_threshold).astype(int)

        # Per-device thermal statistics
        thermal_stats = df.groupby("DeviceId").agg(
            temp_avg=("temp_value", "mean"),
            temp_max=("temp_value", "max"),
            temp_p95=("temp_value", lambda x: x.quantile(0.95) if len(x) > 0 else np.nan),
            thermal_warning_count=("is_thermal_warning", "sum"),
            thermal_warning_rate=("is_thermal_warning", "mean"),
        ).reset_index()

        df = df.merge(thermal_stats, on="DeviceId", how="left")

        # Thermal throttle risk (high temp + high CPU)
        if "is_cpu_spike" in df.columns:
            df["thermal_throttle_risk"] = (
                df["is_thermal_warning"].astype(float) * 0.6 +
                df["is_cpu_spike"].astype(float) * 0.4
            )
        else:
            df["thermal_throttle_risk"] = df["is_thermal_warning"].astype(float)

        # Cleanup
        df = df.drop(columns=["temp_value"], errors="ignore")

        return df

    def _add_composite_health_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add composite system health score combining all metrics."""
        health_components = []
        weights = []

        # CPU health (lower usage = better)
        if "cpu_usage_avg" in df.columns:
            cpu_health = 1 - (df["cpu_usage_avg"].fillna(50) / 100).clip(0, 1)
            health_components.append(cpu_health)
            weights.append(0.25)

        # RAM health (lower pressure = better)
        if "ram_pressure_rate" in df.columns:
            ram_health = 1 - df["ram_pressure_rate"].fillna(0).clip(0, 1)
            health_components.append(ram_health)
            weights.append(0.25)

        # Storage health (lower utilization = better)
        if "storage_utilization_avg" in df.columns:
            storage_health = 1 - df["storage_utilization_avg"].fillna(0.5).clip(0, 1)
            health_components.append(storage_health)
            weights.append(0.25)

        # Thermal health (lower temp = better)
        if "thermal_warning_rate" in df.columns:
            thermal_health = 1 - df["thermal_warning_rate"].fillna(0).clip(0, 1)
            health_components.append(thermal_health)
            weights.append(0.25)

        if health_components:
            # Weighted average
            total_weight = sum(weights)
            df["system_health_score"] = sum(
                c * w for c, w in zip(health_components, weights, strict=False)
            ) / total_weight

            # Health risk score (inverse of health)
            df["system_health_risk"] = 1 - df["system_health_score"]

        return df


def build_system_health_features(
    df: pd.DataFrame,
    cpu_spike_threshold: float = 90.0,
    ram_pressure_threshold: float = 0.85,
    temp_warning_threshold: float = 45.0,
) -> pd.DataFrame:
    """
    Convenience function to build system health features.

    Args:
        df: DataFrame with system metrics
        cpu_spike_threshold: CPU % threshold for spike detection
        ram_pressure_threshold: RAM utilization threshold for pressure
        temp_warning_threshold: Temperature threshold for warnings

    Returns:
        DataFrame with system health features added
    """
    builder = SystemHealthFeatureBuilder(
        cpu_spike_threshold=cpu_spike_threshold,
        ram_pressure_threshold=ram_pressure_threshold,
        temp_warning_threshold=temp_warning_threshold,
    )
    return builder.transform(df)


def get_system_health_feature_names() -> list[str]:
    """Get list of system health feature names that this module generates."""
    return [
        "cpu_usage_avg",
        "cpu_usage_max",
        "cpu_usage_p50",
        "cpu_usage_p95",
        "cpu_spike_count",
        "cpu_spike_rate",
        "cpu_roll_mean",
        "cpu_roll_std",
        "cpu_volatility",
        "ram_utilization",
        "ram_pressure_rate",
        "ram_pressure_count",
        "ram_utilization_avg",
        "ram_utilization_max",
        "storage_utilization",
        "storage_critical_rate",
        "storage_critical_count",
        "storage_utilization_avg",
        "storage_utilization_max",
        "storage_days_until_full",
        "temp_avg",
        "temp_max",
        "temp_p95",
        "thermal_warning_count",
        "thermal_warning_rate",
        "thermal_throttle_risk",
        "system_health_score",
        "system_health_risk",
    ]
