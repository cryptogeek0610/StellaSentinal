"""
Location-Based Feature Engineering for Anomaly Detection.

This module transforms GPS and WiFi location data into ML features including:
- Mobility patterns (distance traveled, location clusters)
- Dead zone detection (time in low-signal areas)
- WiFi coverage analysis (AP hopping, signal gaps)
- Geographic entropy (randomness of movement patterns)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Haversine formula constants
EARTH_RADIUS_KM = 6371.0


def haversine_distance(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Calculate haversine distance between coordinate pairs in kilometers."""
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return EARTH_RADIUS_KM * c


class LocationFeatureBuilder:
    """
    Location-based feature engineering for device anomaly detection.

    Transforms GPS/WiFi location data into ML-ready features:
    - Mobility features (daily distance, cluster count)
    - Dead zone features (time in low-signal areas)
    - WiFi features (AP diversity, signal quality)
    """

    def __init__(
        self,
        signal_threshold_dbm: float = -80.0,
        cluster_eps_km: float = 0.5,
        min_cluster_samples: int = 3,
    ):
        """
        Initialize the location feature builder.

        Args:
            signal_threshold_dbm: Signal strength below which is considered a dead zone
            cluster_eps_km: DBSCAN epsilon in kilometers for location clustering
            min_cluster_samples: Minimum samples for a location cluster
        """
        self.signal_threshold_dbm = signal_threshold_dbm
        self.cluster_eps_km = cluster_eps_km
        self.min_cluster_samples = min_cluster_samples

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build location-based features per device.

        Expects columns like:
          - DeviceId
          - Timestamp
          - Latitude, Longitude (optional)
          - SignalStrength or AvgSignalStrength (optional)
          - SSID or AccessPoint (optional)

        Returns DataFrame with location features added.
        """
        if df.empty:
            return df

        df = df.copy()

        # Add mobility features if GPS data available
        if self._has_gps_data(df):
            logger.info("Computing mobility features from GPS data...")
            df = self._add_mobility_features(df)
            df = self._add_location_cluster_features(df)
            df = self._add_location_entropy_features(df)

        # Add dead zone features if signal data available
        if self._has_signal_data(df):
            logger.info("Computing dead zone features...")
            df = self._add_dead_zone_features(df)

        # Add WiFi features if SSID data available
        if self._has_wifi_data(df):
            logger.info("Computing WiFi pattern features...")
            df = self._add_wifi_features(df)

        return df

    def _has_gps_data(self, df: pd.DataFrame) -> bool:
        """Check if GPS coordinates are available."""
        return "Latitude" in df.columns and "Longitude" in df.columns

    def _has_signal_data(self, df: pd.DataFrame) -> bool:
        """Check if signal strength data is available."""
        return any(col in df.columns for col in ["SignalStrength", "AvgSignalStrength", "RssiSignal"])

    def _has_wifi_data(self, df: pd.DataFrame) -> bool:
        """Check if WiFi SSID/AP data is available."""
        return any(col in df.columns for col in ["SSID", "AccessPoint", "WifiSSID", "BSSID"])

    def _add_mobility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add mobility-related features from GPS coordinates."""
        if "DeviceId" not in df.columns:
            return df

        # Sort by device and time
        df = df.sort_values(["DeviceId", "Timestamp"]).reset_index(drop=True)

        # Calculate distance between consecutive points per device
        df["prev_lat"] = df.groupby("DeviceId")["Latitude"].shift(1)
        df["prev_lon"] = df.groupby("DeviceId")["Longitude"].shift(1)

        # Filter valid coordinate pairs
        valid_mask = (
            df["Latitude"].notna() &
            df["Longitude"].notna() &
            df["prev_lat"].notna() &
            df["prev_lon"].notna()
        )

        df["point_distance_km"] = np.nan
        if valid_mask.any():
            df.loc[valid_mask, "point_distance_km"] = haversine_distance(
                df.loc[valid_mask, "prev_lat"].values,
                df.loc[valid_mask, "prev_lon"].values,
                df.loc[valid_mask, "Latitude"].values,
                df.loc[valid_mask, "Longitude"].values,
            )

        # Daily aggregated mobility features per device
        if "Timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["Timestamp"]).dt.date

            daily_distance = df.groupby(["DeviceId", "date"])["point_distance_km"].sum().reset_index()
            daily_distance.columns = ["DeviceId", "date", "daily_distance_km"]

            df = df.merge(daily_distance, on=["DeviceId", "date"], how="left")

            # Rolling average distance
            df["distance_roll_mean"] = df.groupby("DeviceId")["point_distance_km"].transform(
                lambda x: x.rolling(window=14, min_periods=3).mean()
            )
            df["distance_roll_std"] = df.groupby("DeviceId")["point_distance_km"].transform(
                lambda x: x.rolling(window=14, min_periods=3).std()
            )

            df = df.drop(columns=["date"], errors="ignore")

        # Clean up temporary columns
        df = df.drop(columns=["prev_lat", "prev_lon"], errors="ignore")

        return df

    def _add_location_cluster_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add location cluster features using HDBSCAN clustering."""
        if not self._has_gps_data(df):
            return df

        try:
            import hdbscan
        except ImportError:
            logger.warning("hdbscan not installed, skipping location clustering")
            df["location_cluster_count"] = np.nan
            return df

        def count_clusters(grp: pd.DataFrame) -> pd.Series:
            coords = grp[["Latitude", "Longitude"]].dropna()
            if len(coords) < self.min_cluster_samples:
                return pd.Series({"location_cluster_count": np.nan})

            # Convert to radians for haversine distance
            coords_rad = np.radians(coords.values)

            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_cluster_samples,
                    metric="haversine",
                    cluster_selection_epsilon=self.cluster_eps_km / EARTH_RADIUS_KM,
                )
                labels = clusterer.fit_predict(coords_rad)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            except Exception as e:
                logger.warning(f"Clustering failed: {e}")
                n_clusters = np.nan

            return pd.Series({"location_cluster_count": n_clusters})

        # Compute per device
        cluster_counts = df.groupby("DeviceId").apply(count_clusters).reset_index()
        df = df.merge(cluster_counts, on="DeviceId", how="left")

        return df

    def _add_location_entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add location entropy features measuring movement randomness."""
        if not self._has_gps_data(df):
            return df

        def compute_entropy(grp: pd.DataFrame) -> pd.Series:
            coords = grp[["Latitude", "Longitude"]].dropna()
            if len(coords) < 5:
                return pd.Series({"location_entropy": np.nan})

            # Grid-based entropy (discretize coordinates)
            lat_bins = pd.cut(coords["Latitude"], bins=10, labels=False)
            lon_bins = pd.cut(coords["Longitude"], bins=10, labels=False)

            # Combined cell index
            cells = lat_bins.astype(str) + "_" + lon_bins.astype(str)
            cell_counts = cells.value_counts(normalize=True)

            # Shannon entropy
            entropy = -np.sum(cell_counts * np.log2(cell_counts + 1e-10))

            return pd.Series({"location_entropy": entropy})

        entropy_df = df.groupby("DeviceId").apply(compute_entropy).reset_index()
        df = df.merge(entropy_df, on="DeviceId", how="left")

        return df

    def _add_dead_zone_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add dead zone features based on signal strength."""
        # Find signal column
        signal_col = None
        for col in ["SignalStrength", "AvgSignalStrength", "RssiSignal"]:
            if col in df.columns:
                signal_col = col
                break

        if signal_col is None:
            return df

        # Mark dead zone records
        df["is_dead_zone"] = (df[signal_col] < self.signal_threshold_dbm).astype(int)

        # Calculate dead zone percentage per device
        if "DeviceId" in df.columns:
            dead_zone_pct = df.groupby("DeviceId")["is_dead_zone"].mean().reset_index()
            dead_zone_pct.columns = ["DeviceId", "dead_zone_time_pct"]
            df = df.merge(dead_zone_pct, on="DeviceId", how="left")

            # Count dead zone transitions (entering/exiting dead zones)
            df["dead_zone_transition"] = df.groupby("DeviceId")["is_dead_zone"].diff().abs()
            transition_count = df.groupby("DeviceId")["dead_zone_transition"].sum().reset_index()
            transition_count.columns = ["DeviceId", "dead_zone_transitions"]
            df = df.merge(transition_count, on="DeviceId", how="left")

        return df

    def _add_wifi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add WiFi-related features from SSID/AP data."""
        # Find SSID column
        ssid_col = None
        for col in ["SSID", "AccessPoint", "WifiSSID", "BSSID"]:
            if col in df.columns:
                ssid_col = col
                break

        if ssid_col is None or "DeviceId" not in df.columns:
            return df

        # Count unique APs per device
        ap_counts = df.groupby("DeviceId")[ssid_col].nunique().reset_index()
        ap_counts.columns = ["DeviceId", "unique_ap_count"]
        df = df.merge(ap_counts, on="DeviceId", how="left")

        # AP hopping rate (AP changes per record)
        df["ap_changed"] = (df.groupby("DeviceId")[ssid_col].shift(1) != df[ssid_col]).astype(int)
        ap_change_rate = df.groupby("DeviceId")["ap_changed"].mean().reset_index()
        ap_change_rate.columns = ["DeviceId", "ap_hopping_rate"]
        df = df.merge(ap_change_rate, on="DeviceId", how="left")

        # AP stickiness (time on primary AP)
        def compute_stickiness(grp: pd.DataFrame) -> float:
            if grp[ssid_col].isna().all():
                return np.nan
            primary_ap = grp[ssid_col].mode()
            if len(primary_ap) == 0:
                return np.nan
            return (grp[ssid_col] == primary_ap.iloc[0]).mean()

        stickiness = df.groupby("DeviceId").apply(compute_stickiness).reset_index()
        stickiness.columns = ["DeviceId", "primary_ap_stickiness"]
        df = df.merge(stickiness, on="DeviceId", how="left")

        return df


def build_location_features(
    df: pd.DataFrame,
    signal_threshold_dbm: float = -80.0,
    cluster_eps_km: float = 0.5,
) -> pd.DataFrame:
    """
    Convenience function to build location features.

    Args:
        df: DataFrame with location/signal data
        signal_threshold_dbm: Signal threshold for dead zone detection
        cluster_eps_km: Clustering epsilon in kilometers

    Returns:
        DataFrame with location features added
    """
    builder = LocationFeatureBuilder(
        signal_threshold_dbm=signal_threshold_dbm,
        cluster_eps_km=cluster_eps_km,
    )
    return builder.transform(df)


def get_location_feature_names() -> list[str]:
    """Get list of location feature names that this module generates."""
    return [
        "point_distance_km",
        "daily_distance_km",
        "distance_roll_mean",
        "distance_roll_std",
        "location_cluster_count",
        "location_entropy",
        "is_dead_zone",
        "dead_zone_time_pct",
        "dead_zone_transitions",
        "unique_ap_count",
        "ap_hopping_rate",
        "primary_ap_stickiness",
    ]
