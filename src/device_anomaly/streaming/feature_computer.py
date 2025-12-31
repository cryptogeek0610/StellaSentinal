"""
Streaming Feature Computer - Incremental feature computation for real-time scoring.

Computes features incrementally as telemetry arrives, without needing
to reprocess the entire history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np

from device_anomaly.config.feature_config import FeatureConfig
from device_anomaly.streaming.engine import (
    StreamingEngine,
    StreamMessage,
    MessageType,
)
from device_anomaly.streaming.telemetry_stream import (
    TelemetryBuffer,
    TelemetryEvent,
    DeviceBuffer,
)

logger = logging.getLogger(__name__)


@dataclass
class IncrementalStats:
    """Welford's online algorithm for incremental mean/variance."""

    count: int = 0
    mean: float = 0.0
    M2: float = 0.0  # Sum of squared differences from mean

    def update(self, value: float) -> None:
        """Add a new value to the running statistics."""
        if value is None or np.isnan(value):
            return

        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        """Get the sample variance."""
        if self.count < 2:
            return 0.0
        return self.M2 / (self.count - 1)

    @property
    def std(self) -> float:
        """Get the sample standard deviation."""
        return np.sqrt(self.variance)

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "variance": self.variance,
        }


@dataclass
class CohortStats:
    """Running statistics for a device cohort."""

    cohort_id: str
    metrics: dict[str, IncrementalStats] = field(default_factory=dict)
    device_count: int = 0
    last_update: datetime = field(default_factory=datetime.utcnow)

    def update_metric(self, metric_name: str, value: float) -> None:
        """Update running stats for a metric."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = IncrementalStats()
        self.metrics[metric_name].update(value)
        self.last_update = datetime.utcnow()

    def get_z_score(self, metric_name: str, value: float) -> Optional[float]:
        """Compute z-score relative to cohort."""
        if metric_name not in self.metrics:
            return None

        stats = self.metrics[metric_name]
        if stats.count < 10 or stats.std < 1e-6:
            return None

        return (value - stats.mean) / stats.std


class StreamingFeatureComputer:
    """
    Computes features incrementally for real-time anomaly detection.

    Features computed:
    - Rolling statistics (mean, std, min, max)
    - Delta features (change from previous)
    - Cohort z-scores (normalized by device type)
    - Derived efficiency metrics
    - Cross-domain correlations

    Usage:
        computer = StreamingFeatureComputer(engine, buffer)
        await computer.start()

        # Features are computed automatically when telemetry arrives
        # Results published to FEATURES_COMPUTED channel
    """

    def __init__(
        self,
        engine: StreamingEngine,
        buffer: TelemetryBuffer,
    ):
        self.engine = engine
        self.buffer = buffer
        self._cohort_stats: dict[str, CohortStats] = {}
        self._running = False

    async def start(self) -> None:
        """Start the feature computer."""
        await self.engine.subscribe(
            MessageType.TELEMETRY_ENRICHED,
            self._handle_enriched_telemetry,
        )
        self._running = True
        logger.info("StreamingFeatureComputer started")

    async def stop(self) -> None:
        """Stop the feature computer."""
        self._running = False
        await self.engine.unsubscribe(MessageType.TELEMETRY_ENRICHED)
        logger.info("StreamingFeatureComputer stopped")

    async def _handle_enriched_telemetry(self, message: StreamMessage) -> None:
        """Process enriched telemetry and compute features."""
        try:
            event_data = message.payload.get("event", {})
            event = TelemetryEvent.from_dict(event_data)

            # Get device buffer
            device_buffer = self.buffer.get_buffer(event.device_id)
            if not device_buffer:
                logger.warning("No buffer for device %d", event.device_id)
                return

            # Compute features
            features = self._compute_features(event, device_buffer)

            # Publish computed features
            await self.engine.publish(StreamMessage(
                message_type=MessageType.FEATURES_COMPUTED,
                payload={
                    "device_id": event.device_id,
                    "timestamp": event.timestamp.isoformat(),
                    "cohort_id": event.cohort_id,
                    "features": features,
                    "raw_metrics": event.metrics,
                },
                device_id=event.device_id,
                tenant_id=event.tenant_id,
            ))

            logger.debug(
                "Computed %d features for device %d",
                len(features),
                event.device_id,
            )

        except Exception as e:
            logger.error(
                "Error computing features for device %d: %s",
                message.device_id,
                e,
                exc_info=True,
            )

    def _compute_features(
        self,
        event: TelemetryEvent,
        buffer: DeviceBuffer,
    ) -> dict[str, float]:
        """Compute all features for an event."""
        features: dict[str, float] = {}

        # 1. Raw metrics as features
        for metric, value in event.metrics.items():
            if value is not None and not np.isnan(value):
                features[metric] = value

        # 2. Rolling statistics
        self._add_rolling_features(event, buffer, features)

        # 3. Delta features
        self._add_delta_features(event, buffer, features)

        # 4. Cohort z-scores
        self._add_cohort_features(event, features)

        # 5. Derived efficiency features
        self._add_derived_features(event, features)

        # 6. Cross-domain correlation features
        self._add_cross_domain_features(features)

        return features

    def _add_rolling_features(
        self,
        event: TelemetryEvent,
        buffer: DeviceBuffer,
        features: dict[str, float],
    ) -> None:
        """Add rolling window statistics."""
        for metric in FeatureConfig.rolling_feature_candidates:
            if metric not in event.metrics:
                continue

            stats = buffer.get_rolling_stats(metric)

            if stats["mean"] is not None:
                features[f"{metric}_roll_mean"] = stats["mean"]
            if stats["std"] is not None:
                features[f"{metric}_roll_std"] = stats["std"]
            if stats["min"] is not None:
                features[f"{metric}_roll_min"] = stats["min"]
            if stats["max"] is not None:
                features[f"{metric}_roll_max"] = stats["max"]

    def _add_delta_features(
        self,
        event: TelemetryEvent,
        buffer: DeviceBuffer,
        features: dict[str, float],
    ) -> None:
        """Add change-from-previous features."""
        for metric in FeatureConfig.rolling_feature_candidates:
            if metric not in event.metrics:
                continue

            delta = buffer.get_delta(metric)
            if delta is not None:
                features[f"{metric}_delta"] = delta

                # Percent change
                prev_stats = buffer.get_rolling_stats(metric)
                if prev_stats["mean"] and abs(prev_stats["mean"]) > 1e-6:
                    features[f"{metric}_pct_change"] = delta / abs(prev_stats["mean"])

    def _add_cohort_features(
        self,
        event: TelemetryEvent,
        features: dict[str, float],
    ) -> None:
        """Add cohort-normalized z-score features."""
        cohort_id = event.cohort_id

        # Get or create cohort stats
        if cohort_id not in self._cohort_stats:
            self._cohort_stats[cohort_id] = CohortStats(cohort_id=cohort_id)

        cohort = self._cohort_stats[cohort_id]

        for metric, value in event.metrics.items():
            if value is None or np.isnan(value):
                continue

            # Compute z-score before updating (so we compare against history)
            z_score = cohort.get_z_score(metric, value)
            if z_score is not None:
                # Clip extreme values
                z_score = max(-10, min(10, z_score))
                features[f"{metric}_cohort_z"] = z_score

            # Update cohort stats
            cohort.update_metric(metric, value)

    def _add_derived_features(
        self,
        event: TelemetryEvent,
        features: dict[str, float],
    ) -> None:
        """Add derived efficiency features."""
        m = event.metrics

        # Battery efficiency
        if "TotalBatteryLevelDrop" in m and "TotalDischargeTime_Sec" in m:
            discharge_hours = m["TotalDischargeTime_Sec"] / 3600 + 1
            features["BatteryDrainPerHour"] = m["TotalBatteryLevelDrop"] / discharge_hours

        if "TotalBatteryLevelDrop" in m and "Download" in m and "Upload" in m:
            data_mb = (m.get("Download", 0) + m.get("Upload", 0)) / 1e6 + 1
            features["BatteryDrainPerMB"] = m["TotalBatteryLevelDrop"] / data_mb

        if "ChargePatternGoodCount" in m and "ChargePatternBadCount" in m:
            total = m["ChargePatternGoodCount"] + m["ChargePatternBadCount"] + 1
            features["ChargeQualityScore"] = m["ChargePatternGoodCount"] / total

        # Network efficiency
        if "Download" in m and "Upload" in m and "AvgSignalStrength" in m:
            signal_quality = m["AvgSignalStrength"] + 100  # Convert dBm to positive
            features["DataPerSignalQuality"] = (m["Download"] + m["Upload"]) / (signal_quality + 1)

        if "TotalDropCnt" in m and "TotalSignalReadings" in m:
            features["DropRate"] = m["TotalDropCnt"] / (m["TotalSignalReadings"] + 1)
            features["ConnectionStabilityScore"] = 1 - min(1, features["DropRate"])

        if "TotalDropCnt" in m and "AppForegroundTime" in m:
            active_hours = m["AppForegroundTime"] / 3600 + 1
            features["DropsPerActiveHour"] = m["TotalDropCnt"] / active_hours

        # Usage patterns
        if "CrashCount" in m and "AppVisitCount" in m:
            features["CrashRate"] = m["CrashCount"] / (m["AppVisitCount"] + 1)

        if "UniqueAppsUsed" in m and "AppVisitCount" in m:
            features["AppDiversity"] = m["UniqueAppsUsed"] / (m["AppVisitCount"] + 1)

        if "Upload" in m and "Download" in m:
            features["UploadToDownloadRatio"] = m["Upload"] / (m["Download"] + 1)

        # Storage
        if "AvailableStorage" in m and "TotalStorage" in m and m["TotalStorage"] > 0:
            features["StorageUtilization"] = 1 - (m["AvailableStorage"] / m["TotalStorage"])

        if "AvailableRAM" in m and "TotalRAM" in m and m["TotalRAM"] > 0:
            features["RAMPressure"] = 1 - (m["AvailableRAM"] / m["TotalRAM"])

    def _add_cross_domain_features(
        self,
        features: dict[str, float],
    ) -> None:
        """Add cross-domain correlation features."""
        # Device Health Score (composite)
        health_components = []

        if "ChargeQualityScore" in features:
            health_components.append(features["ChargeQualityScore"])
        if "StorageUtilization" in features:
            health_components.append(1 - features["StorageUtilization"])
        if "ConnectionStabilityScore" in features:
            health_components.append(features["ConnectionStabilityScore"])
        if "CrashRate" in features:
            health_components.append(1 - min(1, features["CrashRate"]))

        if health_components:
            features["DeviceHealthScore"] = sum(health_components) / len(health_components)

        # Anomaly Risk Score
        risk_components = []

        if "CrashRate" in features:
            risk_components.append(min(1, features["CrashRate"]))
        if "DropRate" in features:
            risk_components.append(min(1, features["DropRate"]))
        if "BatteryDrainPerHour" in features:
            # Normalize: assume >10%/hour is max concern
            risk_components.append(min(1, features["BatteryDrainPerHour"] / 10))

        if risk_components:
            features["AnomalyRiskScore"] = sum(risk_components) / len(risk_components)

        # Battery-Network Stress
        if "BatteryDrainPerHour" in features and "DropRate" in features:
            features["BatteryNetworkStress"] = features["BatteryDrainPerHour"] * (1 + features["DropRate"])

        # Usage-Storage Pressure
        if "StorageUtilization" in features and "AppDiversity" in features:
            features["UsageStoragePressure"] = features["StorageUtilization"] * features["AppDiversity"]

    def get_cohort_stats(self, cohort_id: str) -> Optional[dict]:
        """Get statistics for a cohort."""
        cohort = self._cohort_stats.get(cohort_id)
        if not cohort:
            return None

        return {
            "cohort_id": cohort_id,
            "device_count": cohort.device_count,
            "metrics": {
                name: stats.to_dict()
                for name, stats in cohort.metrics.items()
            },
            "last_update": cohort.last_update.isoformat(),
        }

    def get_all_cohort_ids(self) -> list[str]:
        """Get all known cohort IDs."""
        return list(self._cohort_stats.keys())
