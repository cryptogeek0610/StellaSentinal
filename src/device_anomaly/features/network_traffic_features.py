"""
Network Traffic Feature Engineering for Anomaly Detection.

This module transforms network usage data into ML features including:
- Per-app traffic analysis (unusual app data usage)
- Upload/download ratio analysis (data exfiltration detection)
- Interface diversity (WiFi vs cellular patterns)
- Network switching behavior
- Background vs foreground data patterns

Data Sources:
- XSight: cs_DataUsageByHour (104M rows), cs_DataUsage
- MobiControl: DeviceStatNetTraffic (244K rows)
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NetworkTrafficFeatureBuilder:
    """
    Network traffic feature engineering for device anomaly detection.

    Transforms raw network traffic data into ML-ready features:
    - Data usage patterns (download/upload ratios)
    - Per-app anomaly indicators
    - Interface switching behavior
    - Temporal network patterns (peak hours, night usage)
    """

    def __init__(
        self,
        upload_ratio_threshold: float = 2.0,  # Upload > 2x download is suspicious
        background_data_threshold_mb: float = 100.0,  # >100MB background is unusual
        peak_hour_start: int = 9,
        peak_hour_end: int = 17,
    ):
        """
        Initialize the network traffic feature builder.

        Args:
            upload_ratio_threshold: Upload/download ratio threshold for exfiltration risk
            background_data_threshold_mb: Threshold for excessive background data
            peak_hour_start: Business hours start (for pattern analysis)
            peak_hour_end: Business hours end (for pattern analysis)
        """
        self.upload_ratio_threshold = upload_ratio_threshold
        self.background_data_threshold_mb = background_data_threshold_mb
        self.peak_hour_start = peak_hour_start
        self.peak_hour_end = peak_hour_end

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build network traffic features.

        Expects columns like:
          - DeviceId
          - Download, Upload (or TotalDownload, TotalUpload)
          - AppId (optional, for per-app analysis)
          - ConnectionTypeId or InterfaceType (optional)
          - Hour (optional, for temporal patterns)

        Returns DataFrame with network traffic features added.
        """
        if df.empty:
            return df

        df = df.copy()

        # Normalize column names
        df = self._normalize_columns(df)

        # Add basic traffic features
        if self._has_traffic_data(df):
            logger.info("Computing basic network traffic features...")
            df = self._add_basic_traffic_features(df)
            df = self._add_traffic_ratio_features(df)

        # Add per-app traffic features if app data available
        if self._has_app_data(df):
            logger.info("Computing per-app traffic features...")
            df = self._add_app_traffic_features(df)

        # Add interface diversity features if available
        if self._has_interface_data(df):
            logger.info("Computing interface diversity features...")
            df = self._add_interface_features(df)

        # Add temporal traffic patterns if hour data available
        if self._has_temporal_data(df):
            logger.info("Computing temporal traffic patterns...")
            df = self._add_temporal_traffic_features(df)

        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names for consistency."""
        rename_map = {
            "TotalDownload": "Download",
            "TotalUpload": "Upload",
            "download": "Download",
            "upload": "Upload",
            "InterfaceType": "ConnectionType",
            "Deviceid": "DeviceId",
        }
        for old_name, new_name in rename_map.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})
        return df

    def _has_traffic_data(self, df: pd.DataFrame) -> bool:
        """Check if download/upload data is available."""
        return "Download" in df.columns or "Upload" in df.columns

    def _has_app_data(self, df: pd.DataFrame) -> bool:
        """Check if per-app data is available."""
        return "AppId" in df.columns

    def _has_interface_data(self, df: pd.DataFrame) -> bool:
        """Check if interface/connection type data is available."""
        return "ConnectionType" in df.columns or "ConnectionTypeId" in df.columns

    def _has_temporal_data(self, df: pd.DataFrame) -> bool:
        """Check if hour data is available."""
        return "Hour" in df.columns or "Timestamp" in df.columns

    def _add_basic_traffic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic traffic volume features."""
        # Total data volume
        download = df["Download"].fillna(0) if "Download" in df.columns else 0
        upload = df["Upload"].fillna(0) if "Upload" in df.columns else 0

        df["total_data"] = download + upload
        df["total_data_mb"] = df["total_data"] / (1024 * 1024)

        # Log-transformed (reduces outlier impact)
        df["total_data_log"] = np.log1p(df["total_data"])
        df["download_log"] = np.log1p(download) if isinstance(download, pd.Series) else 0
        df["upload_log"] = np.log1p(upload) if isinstance(upload, pd.Series) else 0

        return df

    def _add_traffic_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add traffic ratio features for anomaly detection."""
        download = df.get("Download", pd.Series(0, index=df.index)).fillna(0)
        upload = df.get("Upload", pd.Series(0, index=df.index)).fillna(0)

        # Upload/download ratio (potential exfiltration indicator)
        total = download + upload
        df["upload_ratio"] = np.where(total > 0, upload / (total + 1), 0)
        df["download_ratio"] = np.where(total > 0, download / (total + 1), 0)

        # Exfiltration risk score (high upload, low download)
        # Normal devices download more than upload (browsing, updates, etc.)
        upload_download_ratio = np.where(download > 0, upload / (download + 1), upload)
        df["exfiltration_risk"] = (upload_download_ratio > self.upload_ratio_threshold).astype(int)

        # High upload anomaly score
        df["upload_anomaly_score"] = np.clip(upload_download_ratio / self.upload_ratio_threshold, 0, 10)

        return df

    def _add_app_traffic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add per-app traffic analysis features."""
        if "DeviceId" not in df.columns:
            return df

        # Count unique apps per device
        app_counts = df.groupby("DeviceId")["AppId"].nunique().reset_index()
        app_counts.columns = ["DeviceId", "unique_data_apps"]
        df = df.merge(app_counts, on="DeviceId", how="left")

        # Traffic concentration (is data dominated by few apps?)
        def compute_traffic_concentration(grp: pd.DataFrame) -> pd.Series:
            if grp["total_data"].sum() == 0:
                return pd.Series({"traffic_concentration": 0, "top_app_data_pct": 0})

            # Traffic per app
            app_traffic = grp.groupby("AppId")["total_data"].sum()
            total_traffic = app_traffic.sum()

            if total_traffic == 0 or len(app_traffic) == 0:
                return pd.Series({"traffic_concentration": 0, "top_app_data_pct": 0})

            # Herfindahl-Hirschman Index (concentration measure)
            shares = app_traffic / total_traffic
            hhi = (shares ** 2).sum()

            # Top app percentage
            top_app_pct = app_traffic.max() / total_traffic if total_traffic > 0 else 0

            return pd.Series({
                "traffic_concentration": hhi,
                "top_app_data_pct": top_app_pct,
            })

        if "total_data" in df.columns:
            concentration = df.groupby("DeviceId").apply(compute_traffic_concentration).reset_index()
            df = df.merge(concentration, on="DeviceId", how="left")

        return df

    def _add_interface_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interface diversity features."""
        if "DeviceId" not in df.columns:
            return df

        interface_col = "ConnectionTypeId" if "ConnectionTypeId" in df.columns else "ConnectionType"

        if interface_col not in df.columns:
            return df

        # Interface diversity (number of different connection types used)
        interface_counts = df.groupby("DeviceId")[interface_col].nunique().reset_index()
        interface_counts.columns = ["DeviceId", "interface_diversity"]
        df = df.merge(interface_counts, on="DeviceId", how="left")

        # Interface switching rate
        df["interface_changed"] = (
            df.groupby("DeviceId")[interface_col].shift(1) != df[interface_col]
        ).astype(int)

        switching_rate = df.groupby("DeviceId")["interface_changed"].mean().reset_index()
        switching_rate.columns = ["DeviceId", "interface_switching_rate"]
        df = df.merge(switching_rate, on="DeviceId", how="left")

        # WiFi vs Cellular ratio (if we can identify them)
        # Common convention: ConnectionTypeId 1 = WiFi, 2 = Cellular
        if df[interface_col].dtype in [np.int64, np.float64, "int64", "float64"]:
            wifi_data = df[df[interface_col] == 1].groupby("DeviceId")["total_data"].sum().reset_index()
            wifi_data.columns = ["DeviceId", "wifi_data_total"]

            cellular_data = df[df[interface_col] == 2].groupby("DeviceId")["total_data"].sum().reset_index()
            cellular_data.columns = ["DeviceId", "cellular_data_total"]

            df = df.merge(wifi_data, on="DeviceId", how="left")
            df = df.merge(cellular_data, on="DeviceId", how="left")

            df["wifi_data_total"] = df["wifi_data_total"].fillna(0)
            df["cellular_data_total"] = df["cellular_data_total"].fillna(0)

            total = df["wifi_data_total"] + df["cellular_data_total"]
            df["wifi_data_ratio"] = np.where(total > 0, df["wifi_data_total"] / total, 0.5)

        return df

    def _add_temporal_traffic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal traffic pattern features."""
        # Get hour column
        if "Hour" in df.columns:
            df["_hour"] = df["Hour"]
        elif "Timestamp" in df.columns:
            df["_hour"] = pd.to_datetime(df["Timestamp"]).dt.hour
        else:
            return df

        if "DeviceId" not in df.columns or "total_data" not in df.columns:
            return df

        # Business hours vs off-hours traffic
        df["is_business_hour"] = (
            (df["_hour"] >= self.peak_hour_start) & (df["_hour"] <= self.peak_hour_end)
        ).astype(int)

        df["is_night_hour"] = ((df["_hour"] >= 22) | (df["_hour"] <= 6)).astype(int)

        # Traffic during different periods
        def compute_temporal_pattern(grp: pd.DataFrame) -> pd.Series:
            total = grp["total_data"].sum()
            if total == 0:
                return pd.Series({
                    "business_hour_data_pct": 0.5,
                    "night_data_pct": 0.0,
                    "peak_hour": 12,
                    "hourly_entropy": 0,
                })

            business_data = grp[grp["is_business_hour"] == 1]["total_data"].sum()
            night_data = grp[grp["is_night_hour"] == 1]["total_data"].sum()

            # Peak usage hour
            hourly_data = grp.groupby("_hour")["total_data"].sum()
            peak_hour = hourly_data.idxmax() if len(hourly_data) > 0 else 12

            # Hourly entropy (distribution of usage across hours)
            hourly_pct = hourly_data / (total + 1e-6)
            entropy = -np.sum(hourly_pct * np.log2(hourly_pct + 1e-10)) / np.log2(24)  # Normalized

            return pd.Series({
                "business_hour_data_pct": business_data / total,
                "night_data_pct": night_data / total,
                "peak_hour": peak_hour,
                "hourly_entropy": entropy,
            })

        temporal_patterns = df.groupby("DeviceId").apply(compute_temporal_pattern).reset_index()
        df = df.merge(temporal_patterns, on="DeviceId", how="left")

        # Unusual night activity flag
        df["unusual_night_activity"] = (df["night_data_pct"] > 0.3).astype(int)

        # Clean up
        df = df.drop(columns=["_hour", "is_business_hour", "is_night_hour"], errors="ignore")

        return df


def build_network_traffic_features(
    df: pd.DataFrame,
    upload_ratio_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Convenience function to build network traffic features.

    Args:
        df: DataFrame with network traffic data
        upload_ratio_threshold: Threshold for exfiltration risk detection

    Returns:
        DataFrame with network traffic features added
    """
    builder = NetworkTrafficFeatureBuilder(upload_ratio_threshold=upload_ratio_threshold)
    return builder.transform(df)


def get_network_traffic_feature_names() -> List[str]:
    """Get list of network traffic feature names that this module generates."""
    return [
        # Basic traffic
        "total_data",
        "total_data_mb",
        "total_data_log",
        "download_log",
        "upload_log",
        # Ratios
        "upload_ratio",
        "download_ratio",
        "exfiltration_risk",
        "upload_anomaly_score",
        # Per-app
        "unique_data_apps",
        "traffic_concentration",
        "top_app_data_pct",
        # Interface
        "interface_diversity",
        "interface_switching_rate",
        "wifi_data_total",
        "cellular_data_total",
        "wifi_data_ratio",
        # Temporal
        "business_hour_data_pct",
        "night_data_pct",
        "peak_hour",
        "hourly_entropy",
        "unusual_night_activity",
    ]
