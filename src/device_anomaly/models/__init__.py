"""Anomaly detection models and algorithms.

This module provides the anomaly detection layer with multiple
detector implementations that share a common interface.

Available Detectors:
- IsolationForest: Tree-based unsupervised anomaly detection
- Statistical: Z-score based deviation detection
- Ensemble: Combine multiple detectors with voting

Example:
    from device_anomaly.models import DetectorRegistry, DetectorConfig

    # Create an isolation forest detector
    config = DetectorConfig(name='prod_detector', contamination=0.03)
    detector = DetectorRegistry.create('isolation_forest', config)

    # Train and detect
    detector.fit(training_df)
    anomalies = detector.detect(new_df)
"""
from device_anomaly.models.anomaly_detector import (
    AnomalyDetectorConfig,
    AnomalyDetectorIsolationForest,
)
from device_anomaly.models.base import (
    AnomalyResult,
    BaseAnomalyDetector,
    DetectorConfig,
    DetectorRegistry,
)

__all__ = [
    # Base classes
    "AnomalyResult",
    "BaseAnomalyDetector",
    "DetectorConfig",
    "DetectorRegistry",
    # Implementations
    "AnomalyDetectorConfig",
    "AnomalyDetectorIsolationForest",
]
