"""Base anomaly detector interface.

This module defines the abstract base class for all anomaly detectors.
It ensures a consistent API across different detection methods
(Isolation Forest, Z-score, seasonal decomposition, ensemble, etc.).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DetectorConfig:
    """Base configuration for anomaly detectors.

    Attributes:
        name: Unique identifier for this detector instance
        enabled: Whether this detector is active
        contamination: Expected proportion of anomalies (0.0-1.0)
        severity_thresholds: Score thresholds for severity levels
        metadata: Additional detector-specific configuration
    """

    name: str
    enabled: bool = True
    contamination: float = 0.03
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "critical": -0.5,  # Most extreme anomalies
        "high": -0.3,
        "medium": -0.1,
        "low": 0.0,  # Threshold for any anomaly
    })
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a single data point.

    Attributes:
        index: Index in the original DataFrame
        device_id: Device identifier
        timestamp: Time of the observation
        score: Anomaly score (lower = more anomalous for most detectors)
        is_anomaly: Whether this point is classified as an anomaly
        severity: Severity level (low, medium, high, critical)
        contributing_features: Features that contributed most to the anomaly
        explanation: Human-readable explanation
        detector_name: Name of the detector that flagged this
    """

    index: int
    device_id: str
    timestamp: datetime
    score: float
    is_anomaly: bool
    severity: str
    contributing_features: Dict[str, float] = field(default_factory=dict)
    explanation: Optional[str] = None
    detector_name: str = ""


class BaseAnomalyDetector(ABC):
    """Abstract base class for anomaly detectors.

    All anomaly detectors must implement these core methods to ensure
    a consistent interface for the detection pipeline.

    Example:
        class ZScoreDetector(BaseAnomalyDetector):
            def fit(self, df: pd.DataFrame) -> None:
                self.mean_ = df[self.feature_cols].mean()
                self.std_ = df[self.feature_cols].std()
                self._is_fitted = True

            def score(self, df: pd.DataFrame) -> np.ndarray:
                z_scores = (df[self.feature_cols] - self.mean_) / self.std_
                return -z_scores.abs().max(axis=1).values
    """

    def __init__(self, config: DetectorConfig):
        """Initialize the detector with configuration.

        Args:
            config: DetectorConfig with detector settings
        """
        self.config = config
        self._is_fitted = False
        self._feature_cols: List[str] = []
        self._fit_timestamp: Optional[datetime] = None

    @property
    def is_fitted(self) -> bool:
        """Check if the detector has been trained."""
        return self._is_fitted

    @property
    def feature_cols(self) -> List[str]:
        """Get the feature columns used for detection."""
        return self._feature_cols

    @property
    def name(self) -> str:
        """Get the detector name."""
        return self.config.name

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Train the detector on normal data.

        Args:
            df: DataFrame with training data (assumed to be mostly normal)

        Raises:
            ValueError: If data is invalid or insufficient
        """
        pass

    @abstractmethod
    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Compute anomaly scores for data points.

        Lower scores indicate more anomalous behavior for most detectors.

        Args:
            df: DataFrame with data to score

        Returns:
            Array of anomaly scores (one per row)

        Raises:
            RuntimeError: If detector has not been fitted
        """
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict anomaly labels.

        Args:
            df: DataFrame with data to classify

        Returns:
            Array of labels: 1 for normal, -1 for anomaly

        Raises:
            RuntimeError: If detector has not been fitted
        """
        pass

    def detect(self, df: pd.DataFrame) -> List[AnomalyResult]:
        """Detect anomalies and return detailed results.

        This is the main entry point for anomaly detection that returns
        structured results with explanations.

        Args:
            df: DataFrame with device telemetry data
                Must contain 'device_id' and 'timestamp' columns

        Returns:
            List of AnomalyResult for detected anomalies
        """
        if not self._is_fitted:
            raise RuntimeError(f"Detector '{self.name}' has not been fitted")

        scores = self.score(df)
        predictions = self.predict(df)
        contributing = self.explain_contributions(df)

        results = []
        for i, (score, pred) in enumerate(zip(scores, predictions)):
            if pred == -1:  # Is anomaly
                severity = self._score_to_severity(score)
                contributions = contributing.get(i, {})

                result = AnomalyResult(
                    index=i,
                    device_id=str(df.iloc[i].get("device_id", "unknown")),
                    timestamp=df.iloc[i].get("timestamp", datetime.now(timezone.utc)),
                    score=float(score),
                    is_anomaly=True,
                    severity=severity,
                    contributing_features=contributions,
                    detector_name=self.name,
                )
                results.append(result)

        return results

    def explain_contributions(
        self,
        df: pd.DataFrame,
        top_k: int = 5,
    ) -> Dict[int, Dict[str, float]]:
        """Explain which features contributed most to anomaly scores.

        This is a default implementation that can be overridden by
        specific detectors for more accurate explanations.

        Args:
            df: DataFrame with data
            top_k: Number of top contributing features to return

        Returns:
            Dict mapping row index to feature contributions
        """
        if not self._feature_cols:
            return {}

        contributions = {}
        scores = self.score(df)

        # Simple approach: features with highest deviation from training mean
        for i in range(len(df)):
            if not hasattr(self, '_training_mean'):
                break

            row_contributions = {}
            for col in self._feature_cols[:top_k]:
                if col in df.columns:
                    val = df.iloc[i][col]
                    mean = getattr(self, '_training_mean', {}).get(col, 0)
                    row_contributions[col] = abs(val - mean)

            contributions[i] = dict(
                sorted(row_contributions.items(), key=lambda x: -x[1])[:top_k]
            )

        return contributions

    def _score_to_severity(self, score: float) -> str:
        """Convert anomaly score to severity level.

        Args:
            score: Anomaly score (lower = more anomalous)

        Returns:
            Severity level: 'critical', 'high', 'medium', or 'low'
        """
        thresholds = self.config.severity_thresholds
        if score < thresholds.get("critical", -0.5):
            return "critical"
        elif score < thresholds.get("high", -0.3):
            return "high"
        elif score < thresholds.get("medium", -0.1):
            return "medium"
        else:
            return "low"

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score data and add results as columns.

        Args:
            df: DataFrame to score

        Returns:
            DataFrame with added 'anomaly_score' and 'anomaly_label' columns
        """
        df_scored = df.copy()
        df_scored["anomaly_score"] = self.score(df)
        df_scored["anomaly_label"] = self.predict(df)
        return df_scored

    def get_metadata(self) -> Dict[str, Any]:
        """Get detector metadata for observability.

        Returns:
            Dict with detector info: name, type, configuration, etc.
        """
        return {
            "name": self.config.name,
            "type": self.__class__.__name__,
            "enabled": self.config.enabled,
            "is_fitted": self._is_fitted,
            "fit_timestamp": self._fit_timestamp.isoformat() if self._fit_timestamp else None,
            "feature_count": len(self._feature_cols),
            "contamination": self.config.contamination,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.config.name}', fitted={self._is_fitted})>"


class DetectorRegistry:
    """Registry for anomaly detector types.

    Provides a factory pattern for registering and creating detector instances.
    """

    _detectors: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, detector_class: type) -> None:
        """Register a detector class.

        Args:
            name: Unique name for the detector type
            detector_class: The detector class to register
        """
        if not issubclass(detector_class, BaseAnomalyDetector):
            raise TypeError(f"{detector_class} must inherit from BaseAnomalyDetector")
        cls._detectors[name] = detector_class

    @classmethod
    def create(cls, name: str, config: DetectorConfig) -> BaseAnomalyDetector:
        """Create a detector instance.

        Args:
            name: Registered detector type name
            config: Configuration for the detector

        Returns:
            Configured detector instance

        Raises:
            KeyError: If detector type is not registered
        """
        if name not in cls._detectors:
            available = ", ".join(cls._detectors.keys()) or "none"
            raise KeyError(f"Detector '{name}' not registered. Available: {available}")
        return cls._detectors[name](config)

    @classmethod
    def list_detectors(cls) -> List[str]:
        """List all registered detector types."""
        return list(cls._detectors.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a detector type is registered."""
        return name in cls._detectors
