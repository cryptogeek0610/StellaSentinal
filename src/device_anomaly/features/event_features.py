"""
Event-Based Feature Engineering for Anomaly Detection.

This module transforms event log and alert data into ML features including:
- Crash patterns (frequency, trend, concentration)
- Error event ratios and log anomalies
- Alert patterns (severity, escalation, resolution)
- Event type diversity and entropy
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EventFeatureBuilder:
    """
    Event-based feature engineering for device anomaly detection.

    Transforms MainLog and Alert data into ML-ready features:
    - Crash and error patterns
    - Alert severity and escalation metrics
    - Event frequency and diversity
    """

    def __init__(
        self,
        window_days: int = 7,
        min_events: int = 5,
    ):
        """
        Initialize the event feature builder.

        Args:
            window_days: Rolling window for event statistics
            min_events: Minimum events required for pattern analysis
        """
        self.window_days = window_days
        self.min_events = min_events

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build event-based features per device.

        Expects columns like:
          - DeviceId
          - Timestamp or EventTime
          - EventType or LogLevel (optional)
          - Severity (optional)
          - Message or Description (optional)

        Returns DataFrame with event features added.
        """
        if df.empty or "DeviceId" not in df.columns:
            return df

        df = df.copy()

        # Ensure timestamp column
        if "Timestamp" not in df.columns and "EventTime" in df.columns:
            df["Timestamp"] = df["EventTime"]

        # Add event frequency features
        logger.info("Computing event frequency features...")
        df = self._add_event_frequency_features(df)

        # Add crash pattern features if crash data available
        if self._has_crash_data(df):
            logger.info("Computing crash pattern features...")
            df = self._add_crash_features(df)

        # Add error event features if log level available
        if self._has_log_level_data(df):
            logger.info("Computing error event features...")
            df = self._add_error_features(df)

        # Add alert features if severity data available
        if self._has_alert_data(df):
            logger.info("Computing alert pattern features...")
            df = self._add_alert_features(df)

        # Add event type diversity features
        if self._has_event_type_data(df):
            logger.info("Computing event diversity features...")
            df = self._add_event_diversity_features(df)

        return df

    def _has_crash_data(self, df: pd.DataFrame) -> bool:
        """Check if crash-related data is available."""
        return any(col in df.columns for col in ["CrashCount", "is_crash", "EventType"]) or (
            "Message" in df.columns
            and df["Message"]
            .astype(str)
            .str.contains("crash|exception|error", case=False, na=False)
            .any()
        )

    def _has_log_level_data(self, df: pd.DataFrame) -> bool:
        """Check if log level/event type data is available."""
        return any(col in df.columns for col in ["LogLevel", "EventType", "Level"])

    def _has_alert_data(self, df: pd.DataFrame) -> bool:
        """Check if alert severity data is available."""
        return any(col in df.columns for col in ["Severity", "AlertSeverity", "Priority"])

    def _has_event_type_data(self, df: pd.DataFrame) -> bool:
        """Check if event type data is available."""
        return any(col in df.columns for col in ["EventType", "LogLevel", "Category", "ActionType"])

    def _add_event_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add event frequency-based features."""
        if "Timestamp" not in df.columns:
            return df

        # Ensure datetime
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        # Count events per device
        event_counts = df.groupby("DeviceId").size().reset_index(name="total_event_count")
        df = df.merge(event_counts, on="DeviceId", how="left")

        # Events per day
        df["date"] = df["Timestamp"].dt.date
        daily_counts = df.groupby(["DeviceId", "date"]).size().reset_index(name="daily_event_count")

        # Rolling event frequency
        daily_counts["date"] = pd.to_datetime(daily_counts["date"])
        daily_counts = daily_counts.sort_values(["DeviceId", "date"])

        daily_counts["event_freq_roll_mean"] = daily_counts.groupby("DeviceId")[
            "daily_event_count"
        ].transform(lambda x: x.rolling(window=self.window_days, min_periods=2).mean())
        daily_counts["event_freq_roll_std"] = daily_counts.groupby("DeviceId")[
            "daily_event_count"
        ].transform(lambda x: x.rolling(window=self.window_days, min_periods=2).std())

        # Event frequency trend (slope)
        def compute_trend(grp: pd.DataFrame) -> float:
            if len(grp) < 3:
                return 0.0
            x = np.arange(len(grp))
            y = grp["daily_event_count"].values
            try:
                slope, _ = np.polyfit(x, y, 1)
                return slope
            except Exception:
                return 0.0

        trends = daily_counts.groupby("DeviceId").apply(compute_trend).reset_index()
        trends.columns = ["DeviceId", "event_frequency_trend"]

        df = df.merge(
            daily_counts[
                ["DeviceId", "date", "event_freq_roll_mean", "event_freq_roll_std"]
            ].drop_duplicates(),
            on=["DeviceId", "date"],
            how="left",
        )
        df = df.merge(trends, on="DeviceId", how="left")

        df = df.drop(columns=["date"], errors="ignore")
        return df

    def _add_crash_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add crash pattern features."""
        # Identify crash events
        if "is_crash" not in df.columns:
            if "EventType" in df.columns:
                df["is_crash"] = (
                    df["EventType"]
                    .astype(str)
                    .str.contains("crash|exception|fatal", case=False, na=False)
                    .astype(int)
                )
            elif "Message" in df.columns:
                df["is_crash"] = (
                    df["Message"]
                    .astype(str)
                    .str.contains("crash|exception|fatal|error", case=False, na=False)
                    .astype(int)
                )
            else:
                df["is_crash"] = 0

        # Crash rate per device
        crash_stats = (
            df.groupby("DeviceId")
            .agg(crash_count=("is_crash", "sum"), total_events=("is_crash", "count"))
            .reset_index()
        )
        crash_stats["crash_rate"] = crash_stats["crash_count"] / (crash_stats["total_events"] + 1)

        df = df.merge(
            crash_stats[["DeviceId", "crash_count", "crash_rate"]], on="DeviceId", how="left"
        )

        # Crash trend (increasing/decreasing over time)
        if "Timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["Timestamp"]).dt.date
            daily_crashes = df.groupby(["DeviceId", "date"])["is_crash"].sum().reset_index()
            daily_crashes.columns = ["DeviceId", "date", "daily_crashes"]

            def crash_trend(grp: pd.DataFrame) -> float:
                if len(grp) < 3:
                    return 0.0
                x = np.arange(len(grp))
                y = grp["daily_crashes"].values
                try:
                    slope, _ = np.polyfit(x, y, 1)
                    return slope
                except Exception:
                    return 0.0

            trends = daily_crashes.groupby("DeviceId").apply(crash_trend).reset_index()
            trends.columns = ["DeviceId", "crash_frequency_trend"]
            df = df.merge(trends, on="DeviceId", how="left")

            df = df.drop(columns=["date"], errors="ignore")

        # Crash concentration (are crashes from one app?)
        if "AppName" in df.columns or "ApplicationId" in df.columns:
            app_col = "AppName" if "AppName" in df.columns else "ApplicationId"
            crash_df = df[df["is_crash"] == 1]
            if not crash_df.empty:
                crash_by_app = (
                    crash_df.groupby(["DeviceId", app_col]).size().reset_index(name="app_crashes")
                )

                def concentration(grp: pd.DataFrame) -> float:
                    if grp["app_crashes"].sum() == 0:
                        return 0.0
                    max_app = grp["app_crashes"].max()
                    total = grp["app_crashes"].sum()
                    return max_app / total

                conc = crash_by_app.groupby("DeviceId").apply(concentration).reset_index()
                conc.columns = ["DeviceId", "crash_concentration"]
                df = df.merge(conc, on="DeviceId", how="left")

        return df

    def _add_error_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add error event features from log levels."""
        # Find log level column
        level_col = None
        for col in ["LogLevel", "Level", "EventType"]:
            if col in df.columns:
                level_col = col
                break

        if level_col is None:
            return df

        # Categorize events by severity
        level_lower = df[level_col].astype(str).str.lower()

        df["is_error"] = level_lower.str.contains("error|critical|fatal", na=False).astype(int)
        df["is_warning"] = level_lower.str.contains("warning|warn", na=False).astype(int)
        df["is_info"] = level_lower.str.contains("info|debug", na=False).astype(int)

        # Error ratio per device
        error_stats = (
            df.groupby("DeviceId")
            .agg(
                error_count=("is_error", "sum"),
                warning_count=("is_warning", "sum"),
                total_events=("is_error", "count"),
            )
            .reset_index()
        )

        error_stats["error_event_ratio"] = error_stats["error_count"] / (
            error_stats["total_events"] + 1
        )
        error_stats["warning_event_ratio"] = error_stats["warning_count"] / (
            error_stats["total_events"] + 1
        )
        error_stats["warning_to_error_ratio"] = error_stats["warning_count"] / (
            error_stats["error_count"] + 1
        )

        df = df.merge(
            error_stats[
                [
                    "DeviceId",
                    "error_count",
                    "error_event_ratio",
                    "warning_event_ratio",
                    "warning_to_error_ratio",
                ]
            ],
            on="DeviceId",
            how="left",
        )

        return df

    def _add_alert_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add alert pattern features from severity data."""
        # Find severity column
        severity_col = None
        for col in ["Severity", "AlertSeverity", "Priority"]:
            if col in df.columns:
                severity_col = col
                break

        if severity_col is None:
            return df

        # Map severity to numeric if string
        severity_values = df[severity_col]
        if severity_values.dtype == object:
            severity_map = {
                "low": 1,
                "minor": 1,
                "info": 1,
                "medium": 2,
                "warning": 2,
                "moderate": 2,
                "high": 3,
                "major": 3,
                "error": 3,
                "critical": 4,
                "severe": 4,
                "fatal": 4,
            }
            df["severity_numeric"] = (
                severity_values.astype(str).str.lower().map(severity_map).fillna(2)
            )
        else:
            df["severity_numeric"] = pd.to_numeric(severity_values, errors="coerce").fillna(2)

        # Alert severity stats per device
        severity_stats = (
            df.groupby("DeviceId")
            .agg(
                alert_severity_avg=("severity_numeric", "mean"),
                alert_severity_max=("severity_numeric", "max"),
                critical_alert_count=("severity_numeric", lambda x: (x >= 4).sum()),
                high_alert_count=("severity_numeric", lambda x: (x >= 3).sum()),
            )
            .reset_index()
        )

        df = df.merge(severity_stats, on="DeviceId", how="left")

        # Alert escalation (increasing severity over time)
        if "Timestamp" in df.columns:
            df = df.sort_values(["DeviceId", "Timestamp"])
            df["severity_change"] = df.groupby("DeviceId")["severity_numeric"].diff()

            escalation = (
                df.groupby("DeviceId")["severity_change"]
                .agg(lambda x: (x > 0).sum() / (len(x) + 1))
                .reset_index()
            )
            escalation.columns = ["DeviceId", "alert_escalation_rate"]
            df = df.merge(escalation, on="DeviceId", how="left")

        return df

    def _add_event_diversity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add event type diversity and entropy features."""
        # Find event type column
        type_col = None
        for col in ["EventType", "LogLevel", "Category", "ActionType"]:
            if col in df.columns:
                type_col = col
                break

        if type_col is None:
            return df

        # Unique event types per device
        diversity = df.groupby("DeviceId")[type_col].nunique().reset_index()
        diversity.columns = ["DeviceId", "event_type_diversity"]
        df = df.merge(diversity, on="DeviceId", how="left")

        # Event type entropy (Shannon entropy)
        def compute_entropy(grp: pd.DataFrame) -> float:
            if grp[type_col].isna().all():
                return 0.0
            counts = grp[type_col].value_counts(normalize=True)
            entropy = -np.sum(counts * np.log2(counts + 1e-10))
            return entropy

        entropy_df = df.groupby("DeviceId").apply(compute_entropy).reset_index()
        entropy_df.columns = ["DeviceId", "event_type_entropy"]
        df = df.merge(entropy_df, on="DeviceId", how="left")

        return df


def build_event_features(
    df: pd.DataFrame,
    window_days: int = 7,
    min_events: int = 5,
) -> pd.DataFrame:
    """
    Convenience function to build event features.

    Args:
        df: DataFrame with event/log data
        window_days: Rolling window for event statistics
        min_events: Minimum events required for pattern analysis

    Returns:
        DataFrame with event features added
    """
    builder = EventFeatureBuilder(
        window_days=window_days,
        min_events=min_events,
    )
    return builder.transform(df)


def get_event_feature_names() -> list[str]:
    """Get list of event feature names that this module generates."""
    return [
        "total_event_count",
        "daily_event_count",
        "event_freq_roll_mean",
        "event_freq_roll_std",
        "event_frequency_trend",
        "crash_count",
        "crash_rate",
        "crash_frequency_trend",
        "crash_concentration",
        "is_error",
        "is_warning",
        "error_count",
        "error_event_ratio",
        "warning_event_ratio",
        "warning_to_error_ratio",
        "alert_severity_avg",
        "alert_severity_max",
        "critical_alert_count",
        "high_alert_count",
        "alert_escalation_rate",
        "event_type_diversity",
        "event_type_entropy",
    ]
