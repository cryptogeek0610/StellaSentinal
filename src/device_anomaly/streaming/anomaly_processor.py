"""
Anomaly Stream Processor - Real-time anomaly detection and alerting.

Scores incoming features using the trained model and publishes
anomaly alerts in real-time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from device_anomaly.models.anomaly_detector import AnomalyDetectorIsolationForest
from device_anomaly.streaming.engine import (
    MessageType,
    StreamingEngine,
    StreamMessage,
)

logger = logging.getLogger(__name__)


@dataclass
class StreamingAnomalyResult:
    """Result of real-time anomaly detection."""

    device_id: int
    timestamp: datetime
    anomaly_score: float
    is_anomaly: bool
    confidence: float
    severity: str  # "low", "medium", "high", "critical"
    contributing_features: list[tuple[str, float]]  # Top features with z-scores
    cohort_id: str | None = None
    tenant_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "timestamp": self.timestamp.isoformat(),
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "confidence": self.confidence,
            "severity": self.severity,
            "contributing_features": [
                {"feature": f, "z_score": z} for f, z in self.contributing_features
            ],
            "cohort_id": self.cohort_id,
            "tenant_id": self.tenant_id,
        }


class AnomalyStreamProcessor:
    """
    Real-time anomaly scoring processor.

    Subscribes to computed features and scores them using the
    trained IsolationForest model. Publishes anomaly alerts
    for detected anomalies.

    Usage:
        processor = AnomalyStreamProcessor(engine, model_path="models/production/isolation_forest.pkl")
        await processor.start()

        # Anomalies are detected automatically as features arrive
        # Alerts published to ANOMALY_DETECTED channel
    """

    def __init__(
        self,
        engine: StreamingEngine,
        model_path: str | None = None,
        anomaly_threshold: float | None = None,  # Override model threshold if set
        alert_cooldown_seconds: float = 300,  # 5 minutes between alerts per device
    ):
        self.engine = engine
        self.model_path = model_path
        self.anomaly_threshold = anomaly_threshold
        self.alert_cooldown_seconds = alert_cooldown_seconds

        self._model: AnomalyDetectorIsolationForest | None = None
        self._running = False
        self._last_alert_time: dict[int, datetime] = {}
        self._alert_count = 0
        self._score_count = 0

    async def start(self) -> None:
        """Start the anomaly processor."""
        # Load model
        self._load_model()

        # Subscribe to computed features
        await self.engine.subscribe(
            MessageType.FEATURES_COMPUTED,
            self._handle_features,
        )

        # Subscribe to model updates for hot reloading
        await self.engine.subscribe(
            MessageType.MODEL_UPDATED,
            self._handle_model_update,
        )

        self._running = True
        logger.info("AnomalyStreamProcessor started with model: %s", self.model_path)

    async def stop(self) -> None:
        """Stop the anomaly processor."""
        self._running = False
        await self.engine.unsubscribe(MessageType.FEATURES_COMPUTED)
        await self.engine.unsubscribe(MessageType.MODEL_UPDATED)
        logger.info("AnomalyStreamProcessor stopped")

    def _load_model(self) -> None:
        """Load the anomaly detection model."""
        if self.model_path:
            try:
                self._model = AnomalyDetectorIsolationForest.load_model(self.model_path)
                logger.info(
                    "Loaded model with %d features",
                    len(self._model.feature_cols),
                )
            except FileNotFoundError:
                logger.warning("Model not found at %s, will load on first update", self.model_path)
                self._model = None
        else:
            # Try to load latest model
            self._model = AnomalyDetectorIsolationForest.load_latest()
            if self._model:
                logger.info("Loaded latest production model")
            else:
                logger.warning("No model available, anomaly detection disabled")

    async def _handle_model_update(self, message: StreamMessage) -> None:
        """Handle model update notification."""
        new_model_path = message.payload.get("model_path")
        if new_model_path:
            logger.info("Hot-reloading model from %s", new_model_path)
            self.model_path = new_model_path
            self._load_model()

    async def _handle_features(self, message: StreamMessage) -> None:
        """Process computed features and detect anomalies."""
        if not self._model:
            return

        try:
            device_id = message.payload["device_id"]
            timestamp = datetime.fromisoformat(message.payload["timestamp"])
            features = message.payload["features"]
            cohort_id = message.payload.get("cohort_id")
            tenant_id = message.tenant_id

            # Score the features
            result = self._score_features(
                device_id=device_id,
                timestamp=timestamp,
                features=features,
                cohort_id=cohort_id,
                tenant_id=tenant_id,
            )

            self._score_count += 1

            if result.is_anomaly:
                # Check cooldown
                if self._should_alert(device_id):
                    await self._publish_alert(result)
                    self._last_alert_time[device_id] = datetime.utcnow()
                    self._alert_count += 1

                    logger.info(
                        "ANOMALY DETECTED: device=%d score=%.3f severity=%s",
                        device_id,
                        result.anomaly_score,
                        result.severity,
                    )

        except Exception as e:
            logger.error(
                "Error processing features for device %d: %s",
                message.device_id,
                e,
                exc_info=True,
            )

    def _score_features(
        self,
        device_id: int,
        timestamp: datetime,
        features: dict[str, float],
        cohort_id: str | None,
        tenant_id: str | None,
    ) -> StreamingAnomalyResult:
        """Score features and create result."""
        import pandas as pd

        if self._model.impute_values is None:
            self._model.impute_values = pd.Series(dict.fromkeys(self._model.feature_cols, 0.0))

        # Build feature vector matching model's expected features
        feature_vector = {}
        missing_features = []

        for col in self._model.feature_cols:
            if col in features:
                feature_vector[col] = features[col]
            else:
                # Use imputed value from training
                impute_val = self._model.impute_values.get(col, 0.0)
                feature_vector[col] = impute_val
                missing_features.append(col)

        if missing_features:
            logger.debug(
                "Device %d: %d/%d features missing, using imputed values",
                device_id,
                len(missing_features),
                len(self._model.feature_cols),
            )

        # Create DataFrame for scoring
        df = pd.DataFrame([feature_vector])

        # Get anomaly score + label using model logic
        df_scored = self._model.score_dataframe(df)
        anomaly_score = float(df_scored["anomaly_score"].iloc[0])
        model_label = int(df_scored["anomaly_label"].iloc[0])

        threshold = self.anomaly_threshold if self.anomaly_threshold is not None else 0.0
        if self.anomaly_threshold is None:
            is_anomaly = model_label == -1
        else:
            is_anomaly = anomaly_score < threshold

        # Calculate confidence (how far from threshold)
        confidence = self._calculate_confidence(anomaly_score, threshold)

        # Determine severity
        severity = self._determine_severity(anomaly_score, threshold)

        # Find contributing features (highest z-scores)
        contributing = self._find_contributing_features(features)

        return StreamingAnomalyResult(
            device_id=device_id,
            timestamp=timestamp,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            confidence=confidence,
            severity=severity,
            contributing_features=contributing,
            cohort_id=cohort_id,
            tenant_id=tenant_id,
        )

    def _calculate_confidence(self, score: float, threshold: float) -> float:
        """Calculate confidence based on distance from threshold."""
        # Score ranges roughly from -0.5 (anomaly) to 0.5 (normal)
        # Map to 0-1 confidence
        distance = abs(score - threshold)
        return min(1.0, distance / 0.3)  # Max confidence at 0.3 distance

    def _determine_severity(self, score: float, threshold: float) -> str:
        """Determine anomaly severity from score."""
        if score > threshold:
            return "none"
        elif score > threshold - 0.1:
            return "low"
        elif score > threshold - 0.2:
            return "medium"
        elif score > threshold - 0.3:
            return "high"
        else:
            return "critical"

    def _find_contributing_features(
        self,
        features: dict[str, float],
        top_n: int = 5,
    ) -> list[tuple[str, float]]:
        """Find features contributing most to the anomaly."""
        # Look for cohort z-scores (highest absolute values)
        z_scores = [
            (name, value)
            for name, value in features.items()
            if name.endswith("_cohort_z") and value is not None
        ]

        # Sort by absolute z-score
        z_scores.sort(key=lambda x: abs(x[1]), reverse=True)

        return z_scores[:top_n]

    def _should_alert(self, device_id: int) -> bool:
        """Check if we should send an alert (respecting cooldown)."""
        last_alert = self._last_alert_time.get(device_id)
        if last_alert is None:
            return True

        elapsed = (datetime.utcnow() - last_alert).total_seconds()
        return elapsed >= self.alert_cooldown_seconds

    async def _publish_alert(self, result: StreamingAnomalyResult) -> None:
        """Publish anomaly alert."""
        await self.engine.publish(
            StreamMessage(
                message_type=MessageType.ANOMALY_DETECTED,
                payload=result.to_dict(),
                device_id=result.device_id,
                tenant_id=result.tenant_id,
            )
        )

    def get_stats(self) -> dict[str, Any]:
        """Get processor statistics."""
        return {
            "running": self._running,
            "model_loaded": self._model is not None,
            "model_features": len(self._model.feature_cols) if self._model else 0,
            "scores_processed": self._score_count,
            "alerts_sent": self._alert_count,
            "devices_with_alerts": len(self._last_alert_time),
        }
