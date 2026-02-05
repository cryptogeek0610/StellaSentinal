"""
Weak Labeling for Semi-Supervised Anomaly Detection.

This module generates pseudo-labels when ground truth is unavailable:
- Heuristic rules from domain knowledge
- High-confidence ensemble predictions
- Temporal consistency (repeated anomalies)
- Cross-device correlation patterns
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WeakLabelConfig:
    """Configuration for weak label generation."""

    # High-confidence thresholds
    anomaly_confidence_threshold: float = 0.95
    normal_confidence_threshold: float = 0.90

    # Temporal consistency
    min_consecutive_days: int = 3
    temporal_weight: float = 0.3

    # Cross-device correlation
    min_device_correlation: float = 0.5
    cross_device_weight: float = 0.2

    # Heuristic rule weight
    heuristic_weight: float = 0.5

    # Label values
    anomaly_label: int = -1
    normal_label: int = 1
    unknown_label: int = 0


@dataclass
class WeakLabel:
    """A weak label with confidence and source."""

    device_id: Any
    timestamp: Any
    label: int  # -1 = anomaly, 1 = normal, 0 = unknown
    confidence: float
    sources: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class WeakLabelGenerator:
    """
    Generate pseudo-labels for semi-supervised learning.

    Strategies:
    1. Heuristic rules from domain knowledge
    2. High-confidence model predictions
    3. Temporal consistency (repeated anomalies)
    4. Cross-device correlation patterns
    """

    def __init__(self, config: WeakLabelConfig | None = None):
        self.config = config or WeakLabelConfig()
        self._heuristic_rules: list[dict] = []
        self._load_default_rules()

    def _load_default_rules(self) -> None:
        """Load default heuristic rules for anomaly detection."""
        self._heuristic_rules = [
            # Battery anomalies
            {
                "name": "rapid_battery_drain",
                "column": "BatteryDrainPerHour",
                "op": ">=",
                "threshold": 20.0,
                "label": -1,
                "confidence": 0.8,
            },
            {
                "name": "battery_health_critical",
                "column": "BatteryHealthRatio",
                "op": "<",
                "threshold": 0.5,
                "label": -1,
                "confidence": 0.7,
            },
            # Crash anomalies
            {
                "name": "high_crash_rate",
                "column": "CrashRate",
                "op": ">=",
                "threshold": 0.1,
                "label": -1,
                "confidence": 0.85,
            },
            {
                "name": "excessive_anr",
                "column": "ANRRate",
                "op": ">=",
                "threshold": 0.05,
                "label": -1,
                "confidence": 0.75,
            },
            # Network anomalies
            {
                "name": "high_drop_rate",
                "column": "DropRate",
                "op": ">=",
                "threshold": 0.3,
                "label": -1,
                "confidence": 0.8,
            },
            {
                "name": "poor_signal",
                "column": "AvgSignalStrength",
                "op": "<",
                "threshold": -95.0,
                "label": -1,
                "confidence": 0.6,
            },
            # Storage anomalies
            {
                "name": "storage_critical",
                "column": "StorageUtilization",
                "op": ">=",
                "threshold": 0.95,
                "label": -1,
                "confidence": 0.9,
            },
            # Cohort z-score anomalies
            {
                "name": "extreme_battery_zscore",
                "column": "TotalBatteryLevelDrop_cohort_z",
                "op": "abs>=",
                "threshold": 3.0,
                "label": -1,
                "confidence": 0.7,
            },
            {
                "name": "extreme_crash_zscore",
                "column": "CrashCount_cohort_z",
                "op": ">=",
                "threshold": 3.0,
                "label": -1,
                "confidence": 0.75,
            },
        ]

    def add_rule(
        self,
        name: str,
        column: str,
        op: str,
        threshold: float,
        label: int = -1,
        confidence: float = 0.7,
    ) -> None:
        """
        Add a custom heuristic rule.

        Args:
            name: Rule name for tracking
            column: Column to apply rule to
            op: Operator ('>=', '<=', '<', '>', '==', 'abs>=')
            threshold: Threshold value
            label: Label to assign (-1=anomaly, 1=normal)
            confidence: Confidence level for this rule
        """
        self._heuristic_rules.append(
            {
                "name": name,
                "column": column,
                "op": op,
                "threshold": threshold,
                "label": label,
                "confidence": confidence,
            }
        )

    def generate_heuristic_labels(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply rule-based labeling from heuristic rules.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with heuristic_label and heuristic_confidence columns
        """
        df = df.copy()
        df["heuristic_label"] = self.config.unknown_label
        df["heuristic_confidence"] = 0.0
        df["heuristic_rules_matched"] = ""

        for rule in self._heuristic_rules:
            col = rule["column"]
            if col not in df.columns:
                continue

            op = rule["op"]
            threshold = rule["threshold"]
            label = rule["label"]
            confidence = rule["confidence"]

            # Apply operator
            if op == ">=":
                mask = df[col] >= threshold
            elif op == "<=":
                mask = df[col] <= threshold
            elif op == ">":
                mask = df[col] > threshold
            elif op == "<":
                mask = df[col] < threshold
            elif op == "==":
                mask = df[col] == threshold
            elif op == "abs>=":
                mask = df[col].abs() >= threshold
            else:
                continue

            # Update labels where rule matches and confidence is higher
            update_mask = mask & (confidence > df["heuristic_confidence"])
            df.loc[update_mask, "heuristic_label"] = label
            df.loc[update_mask, "heuristic_confidence"] = confidence

            # Track matched rules
            matched = df.loc[mask, "heuristic_rules_matched"]
            df.loc[mask, "heuristic_rules_matched"] = matched.apply(
                lambda x: f"{x},{rule['name']}" if x else rule["name"]
            )

        return df

    def generate_self_training_labels(
        self,
        df: pd.DataFrame,
        scores: np.ndarray,
        score_col: str = "anomaly_score",
    ) -> pd.DataFrame:
        """
        Self-training: use high-confidence predictions as labels.

        Args:
            df: DataFrame with features
            scores: Anomaly scores (lower = more anomalous)
            score_col: Name for score column

        Returns:
            DataFrame with self_training_label column
        """
        df = df.copy()

        if len(scores) != len(df):
            raise ValueError("Scores length must match dataframe length")

        # Percentile-based thresholds
        anomaly_threshold = np.percentile(
            scores, 100 * (1 - self.config.anomaly_confidence_threshold)
        )
        normal_threshold = np.percentile(scores, 100 * self.config.normal_confidence_threshold)

        df[score_col] = scores
        df["self_training_label"] = self.config.unknown_label
        df["self_training_confidence"] = 0.0

        # High-confidence anomalies (very low scores)
        anomaly_mask = scores <= anomaly_threshold
        df.loc[anomaly_mask, "self_training_label"] = self.config.anomaly_label
        df.loc[anomaly_mask, "self_training_confidence"] = self.config.anomaly_confidence_threshold

        # High-confidence normals (very high scores)
        normal_mask = scores >= normal_threshold
        df.loc[normal_mask, "self_training_label"] = self.config.normal_label
        df.loc[normal_mask, "self_training_confidence"] = self.config.normal_confidence_threshold

        logger.info(
            f"Self-training labels: {anomaly_mask.sum()} anomalies, "
            f"{normal_mask.sum()} normals, {(~anomaly_mask & ~normal_mask).sum()} unknown"
        )

        return df

    def generate_temporal_labels(
        self,
        df: pd.DataFrame,
        label_col: str = "anomaly_label",
    ) -> pd.DataFrame:
        """
        Generate labels based on temporal consistency.

        Devices flagged as anomalies on multiple consecutive days
        are more confidently anomalous.

        Args:
            df: DataFrame with DeviceId, Timestamp, and existing labels
            label_col: Column with existing labels

        Returns:
            DataFrame with temporal_label column
        """
        if "DeviceId" not in df.columns or "Timestamp" not in df.columns:
            logger.warning("Missing DeviceId or Timestamp, skipping temporal labels")
            df["temporal_label"] = self.config.unknown_label
            df["temporal_confidence"] = 0.0
            return df

        df = df.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["date"] = df["Timestamp"].dt.date

        df["temporal_label"] = self.config.unknown_label
        df["temporal_confidence"] = 0.0

        # Group by device and check consecutive anomaly days
        for device_id, grp in df.groupby("DeviceId"):
            grp = grp.sort_values("date")

            if label_col not in grp.columns:
                continue

            # Count consecutive anomaly days
            is_anomaly = (grp[label_col] == self.config.anomaly_label).astype(int)

            # Rolling sum of anomalies
            consecutive_count = is_anomaly.rolling(
                window=self.config.min_consecutive_days, min_periods=1
            ).sum()

            # Mark as confident anomaly if consistently flagged
            consistent_mask = consecutive_count >= self.config.min_consecutive_days

            device_mask = df["DeviceId"] == device_id
            df.loc[device_mask & consistent_mask.values, "temporal_label"] = (
                self.config.anomaly_label
            )
            df.loc[device_mask & consistent_mask.values, "temporal_confidence"] = (
                self.config.temporal_weight
            )

        df = df.drop(columns=["date"], errors="ignore")
        return df

    def generate_cross_device_labels(
        self,
        df: pd.DataFrame,
        cohort_col: str = "cohort_id",
        label_col: str = "anomaly_label",
    ) -> pd.DataFrame:
        """
        Generate labels based on cross-device correlation.

        If many devices in a cohort show the same anomaly pattern,
        it may indicate a fleet-wide issue.

        Args:
            df: DataFrame with cohort and label columns
            cohort_col: Column identifying device cohorts
            label_col: Column with existing labels

        Returns:
            DataFrame with cross_device_label column
        """
        if cohort_col not in df.columns or label_col not in df.columns:
            logger.warning("Missing cohort or label column, skipping cross-device labels")
            df["cross_device_label"] = self.config.unknown_label
            df["cross_device_confidence"] = 0.0
            return df

        df = df.copy()
        df["cross_device_label"] = self.config.unknown_label
        df["cross_device_confidence"] = 0.0

        # Calculate anomaly rate per cohort
        cohort_rates = df.groupby(cohort_col)[label_col].apply(
            lambda x: (x == self.config.anomaly_label).mean()
        )

        # High cohort anomaly rate suggests fleet-wide issue
        high_rate_cohorts = cohort_rates[cohort_rates >= self.config.min_device_correlation].index

        for cohort in high_rate_cohorts:
            cohort_mask = df[cohort_col] == cohort
            anomaly_mask = df[label_col] == self.config.anomaly_label

            # Devices in high-rate cohorts that are anomalous get higher confidence
            df.loc[cohort_mask & anomaly_mask, "cross_device_label"] = self.config.anomaly_label
            df.loc[cohort_mask & anomaly_mask, "cross_device_confidence"] = cohort_rates[cohort]

        return df

    def combine_labels(
        self,
        df: pd.DataFrame,
        label_cols: list[str] | None = None,
        confidence_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Combine multiple weak label sources into final pseudo-labels.

        Args:
            df: DataFrame with various label columns
            label_cols: Label columns to combine
            confidence_cols: Corresponding confidence columns

        Returns:
            DataFrame with combined pseudo_label and pseudo_confidence
        """
        if label_cols is None:
            label_cols = [
                "heuristic_label",
                "self_training_label",
                "temporal_label",
                "cross_device_label",
            ]

        if confidence_cols is None:
            confidence_cols = [
                "heuristic_confidence",
                "self_training_confidence",
                "temporal_confidence",
                "cross_device_confidence",
            ]

        df = df.copy()

        # Filter to existing columns
        existing_labels = [c for c in label_cols if c in df.columns]
        existing_confs = [c for c in confidence_cols if c in df.columns]

        if not existing_labels:
            df["pseudo_label"] = self.config.unknown_label
            df["pseudo_confidence"] = 0.0
            return df

        # Weighted voting
        label_matrix = df[existing_labels].values
        conf_matrix = df[existing_confs].values if existing_confs else np.ones_like(label_matrix)

        # Calculate weighted vote for anomaly
        anomaly_votes = np.sum((label_matrix == self.config.anomaly_label) * conf_matrix, axis=1)
        normal_votes = np.sum((label_matrix == self.config.normal_label) * conf_matrix, axis=1)
        total_confidence = np.sum(conf_matrix, axis=1) + 1e-6

        # Assign final label based on weighted majority
        df["pseudo_label"] = self.config.unknown_label
        df.loc[anomaly_votes > normal_votes, "pseudo_label"] = self.config.anomaly_label
        df.loc[normal_votes > anomaly_votes, "pseudo_label"] = self.config.normal_label

        # Confidence is the margin
        df["pseudo_confidence"] = np.abs(anomaly_votes - normal_votes) / total_confidence

        return df

    def generate_all_labels(
        self,
        df: pd.DataFrame,
        scores: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Generate weak labels using all available strategies.

        Args:
            df: DataFrame with features
            scores: Optional anomaly scores for self-training

        Returns:
            DataFrame with all weak label columns and combined pseudo_label
        """
        # Heuristic labels
        df = self.generate_heuristic_labels(df)

        # Self-training labels (if scores provided)
        if scores is not None:
            df = self.generate_self_training_labels(df, scores)

        # Temporal labels (if we have existing labels)
        if "anomaly_label" in df.columns or "heuristic_label" in df.columns:
            label_col = "anomaly_label" if "anomaly_label" in df.columns else "heuristic_label"
            df = self.generate_temporal_labels(df, label_col)

        # Cross-device labels
        if "cohort_id" in df.columns:
            label_col = "anomaly_label" if "anomaly_label" in df.columns else "heuristic_label"
            df = self.generate_cross_device_labels(df, label_col=label_col)

        # Combine all labels
        df = self.combine_labels(df)

        return df


def create_weak_label_generator(
    anomaly_confidence: float = 0.95,
    normal_confidence: float = 0.90,
) -> WeakLabelGenerator:
    """
    Create a weak label generator.

    Args:
        anomaly_confidence: Confidence threshold for anomaly labels
        normal_confidence: Confidence threshold for normal labels

    Returns:
        Configured WeakLabelGenerator
    """
    config = WeakLabelConfig(
        anomaly_confidence_threshold=anomaly_confidence,
        normal_confidence_threshold=normal_confidence,
    )
    return WeakLabelGenerator(config=config)
