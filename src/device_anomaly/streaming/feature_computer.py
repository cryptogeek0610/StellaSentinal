"""
Streaming Feature Computer - Incremental feature computation for real-time scoring.

Computes features incrementally as telemetry arrives, without needing
to reprocess the entire history.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from device_anomaly.config.feature_config import FeatureConfig
from device_anomaly.features.cohort_stats import CohortStatsStore, apply_cohort_stats
from device_anomaly.features.device_features import build_feature_builder, resolve_feature_spec
from device_anomaly.streaming.drift_monitor import (
    DriftMonitorConfig,
    StreamingDriftMonitor,
    resolve_drift_features,
)
from device_anomaly.streaming.engine import (
    MessageType,
    StreamingEngine,
    StreamMessage,
)
from device_anomaly.streaming.telemetry_stream import (
    DeviceBuffer,
    TelemetryBuffer,
    TelemetryEvent,
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
    last_update: datetime = field(default_factory=lambda: datetime.now(UTC))

    def update_metric(self, metric_name: str, value: float) -> None:
        """Update running stats for a metric."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = IncrementalStats()
        self.metrics[metric_name].update(value)
        self.last_update = datetime.now(UTC)

    def get_z_score(self, metric_name: str, value: float) -> float | None:
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

    Extended features (when enabled):
    - Location features (mobility, WiFi patterns)
    - Event features (crash rates, error trends)
    - System health features (CPU, RAM, thermal)

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
        cohort_stats: CohortStatsStore | None = None,
        feature_spec: dict[str, Any] | None = None,
        feature_norms: dict[str, float] | None = None,
        feature_mode: str | None = None,
        enable_extended_features: bool = False,
        enable_hourly_features: bool = False,
    ):
        self.engine = engine
        self.buffer = buffer
        self._cohort_stats: dict[str, CohortStats] = {}
        self._cohort_stats_store = cohort_stats
        self.feature_spec = resolve_feature_spec(
            {"feature_spec": feature_spec} if feature_spec else None
        )
        self.feature_norms = feature_norms or {}
        self.feature_mode = self._normalize_feature_mode(
            feature_mode or os.getenv("STREAMING_FEATURE_MODE", "canonical")
        )
        # Extended feature flags
        self.enable_extended_features = (
            enable_extended_features
            or os.getenv("STREAMING_EXTENDED_FEATURES", "false").lower() == "true"
        )
        self.enable_hourly_features = (
            enable_hourly_features
            or os.getenv("STREAMING_HOURLY_FEATURES", "false").lower() == "true"
        )

        if self.enable_extended_features:
            logger.info("Extended streaming features ENABLED (location, events, health)")
        if self.enable_hourly_features:
            logger.info("Hourly streaming features ENABLED")

        if self.feature_mode != "canonical":
            logger.warning(
                "Streaming feature mode '%s' reduces parity with batch features.",
                self.feature_mode,
            )
        self._compute_warn_ms = self._parse_float_env("STREAMING_FEATURE_COMPUTE_WARN_MS", 250.0)
        self._builder = build_feature_builder(
            feature_spec=self.feature_spec,
            feature_norms=self.feature_norms,
            compute_cohort=False,
        )
        self._history_required = max(
            [self.feature_spec.get("window", 14)]
            + list(self.feature_spec.get("rolling_windows", []))
        )
        self._warned_short_history: set[int] = set()
        self._warned_missing_norms = False
        self._drift_monitor = self._init_drift_monitor()
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

    @staticmethod
    def _parse_float_env(name: str, default: float) -> float:
        value = os.getenv(name)
        try:
            if value is None:
                return default
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_feature_mode(value: str) -> str:
        raw = (value or "").strip().lower()
        if raw in {"canonical", "canon", "batch"}:
            return "canonical"
        if raw in {"incremental", "inc", "legacy"}:
            return "incremental"
        logger.warning(
            "Unrecognized STREAMING_FEATURE_MODE='%s'; defaulting to incremental.", value
        )
        return "incremental"

    def _init_drift_monitor(self) -> StreamingDriftMonitor | None:
        enabled = os.getenv("STREAMING_DRIFT_ENABLED", "true").lower() == "true"
        if not enabled:
            return None

        default_features = list(FeatureConfig.rolling_feature_candidates)
        default_features.extend(
            ["BatteryDrainPerHour", "DropRate", "StorageUtilization", "AnomalyRiskScore"]
        )

        # Add extended feature names for drift monitoring when enabled
        if self.enable_extended_features:
            # Location features
            default_features.extend(
                [
                    "mobility_score",
                    "distance_traveled",
                    "location_entropy",
                    "unique_locations",
                    "wifi_signal_strength",
                    "dead_zone_ratio",
                ]
            )
            # Event features
            default_features.extend(
                ["crash_rate", "error_rate", "alert_severity_score", "event_frequency", "anr_rate"]
            )
            # System health features
            default_features.extend(
                [
                    "cpu_usage_avg",
                    "ram_pressure",
                    "storage_pressure",
                    "thermal_score",
                    "health_score",
                ]
            )
            logger.info("Extended features added to drift monitoring")

        features = resolve_drift_features(default_features)
        if not features:
            return None

        config = DriftMonitorConfig.from_env()
        return StreamingDriftMonitor(features, config)

    def _maybe_log_drift(
        self,
        features: dict[str, float],
        short_history: bool,
        missing_norms: bool,
    ) -> None:
        if self._drift_monitor is None:
            return
        metrics = self._drift_monitor.update(
            features,
            short_history=short_history,
            missing_norms=missing_norms,
        )
        if metrics:
            logger.info("streaming_drift_metrics %s", metrics)

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
            await self.engine.publish(
                StreamMessage(
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
                )
            )

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
        if self.feature_mode == "canonical":
            return self._compute_features_canonical(event, buffer)
        if self.feature_mode == "incremental":
            return self._compute_features_incremental(event, buffer)

        logger.warning("Unknown feature mode '%s', falling back to incremental.", self.feature_mode)
        return self._compute_features_incremental(event, buffer)

    def _compute_features_incremental(
        self,
        event: TelemetryEvent,
        buffer: DeviceBuffer,
    ) -> dict[str, float]:
        """Compute features using incremental streaming logic."""
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

    def _compute_features_canonical(
        self,
        event: TelemetryEvent,
        buffer: DeviceBuffer,
    ) -> dict[str, float]:
        """Compute features using the canonical batch feature builder."""
        start_time = time.perf_counter()
        short_history = len(buffer.events) < self._history_required
        missing_norms = not self.feature_norms

        if not self.feature_norms and not self._warned_missing_norms:
            logger.warning(
                "Feature norms missing; cross-domain normalization will use buffer quantiles."
            )
            self._warned_missing_norms = True

        if short_history and event.device_id not in self._warned_short_history:
            logger.warning(
                "Device %d has %d events (<%d); rolling features may be partial.",
                event.device_id,
                len(buffer.events),
                self._history_required,
            )
            self._warned_short_history.add(event.device_id)

        df = self._buffer_to_dataframe(buffer)
        if df.empty:
            return {}

        df_feat = self._builder.transform(df)
        df_feat = apply_cohort_stats(df_feat, self._cohort_stats_store)
        if df_feat.empty:
            return {}

        row = df_feat.iloc[-1]
        if "Timestamp" in df_feat.columns:
            import pandas as pd

            event_ts = pd.Timestamp(event.timestamp)
            mask = df_feat["Timestamp"] == event_ts
            if mask.any():
                row = df_feat.loc[mask].iloc[-1]
        features = self._row_to_features(row)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "streaming_feature_compute feature_compute_ms=%.2f device_id=%s event_count=%d mode=%s",
            elapsed_ms,
            event.device_id,
            len(buffer.events),
            self.feature_mode,
        )
        if self._compute_warn_ms and elapsed_ms > self._compute_warn_ms:
            logger.warning(
                "Streaming feature compute exceeded threshold (%.2f ms > %.2f ms)",
                elapsed_ms,
                self._compute_warn_ms,
            )

        self._maybe_log_drift(features, short_history=short_history, missing_norms=missing_norms)
        return features

    def _buffer_to_dataframe(self, buffer: DeviceBuffer):
        import pandas as pd

        rows = []
        for evt in buffer.events:
            firmware_version = evt.firmware_version
            if firmware_version is not None:
                with contextlib.suppress(TypeError, ValueError):
                    firmware_version = float(firmware_version)
            row = {
                "DeviceId": evt.device_id,
                "Timestamp": evt.timestamp,
                "ManufacturerId": evt.manufacturer_id,
                "ModelId": evt.model_id,
                "OsVersionId": evt.os_version_id,
                "FirmwareVersion": firmware_version,
                "tenant_id": evt.tenant_id,
            }
            # Sanitize metrics: skip non-finite values
            for metric, value in evt.metrics.items():
                if value is not None and isinstance(value, (int, float)) and np.isfinite(value):
                    row[metric] = value
                # Skip NaN/inf to avoid contaminating rolling stats
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            # CRITICAL: Sort by timestamp to ensure correct rolling window alignment
            # when events arrive out of order. Batch processing sorts by timestamp,
            # so streaming must do the same for feature parity.
            df = df.sort_values("Timestamp").reset_index(drop=True)
        return df

    @staticmethod
    def _row_to_features(row) -> dict[str, float]:
        features: dict[str, float] = {}
        for col, value in row.items():
            if col in FeatureConfig.excluded_columns:
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    continue
                features[col] = float(value)
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

        if self._cohort_stats_store is not None:
            for metric, value in event.metrics.items():
                z_score = self._cohort_stats_store.get_z_score(cohort_id, metric, value)
                if z_score is not None:
                    features[f"{metric}_cohort_z"] = z_score
            return

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
            features["BatteryNetworkStress"] = features["BatteryDrainPerHour"] * (
                1 + features["DropRate"]
            )

        # Usage-Storage Pressure
        if "StorageUtilization" in features and "AppDiversity" in features:
            features["UsageStoragePressure"] = (
                features["StorageUtilization"] * features["AppDiversity"]
            )

    def get_stats(self) -> dict[str, Any]:
        """Get feature computer status."""
        return {
            "running": self._running,
            "feature_mode": self.feature_mode,
            "feature_spec_version": self.feature_spec.get("version"),
            "history_required": self._history_required,
            "feature_norms_loaded": bool(self.feature_norms),
            "cohort_stats_loaded": self._cohort_stats_store is not None,
            "drift_monitor": self._drift_monitor.get_stats()
            if self._drift_monitor
            else {"enabled": False},
        }

    def get_cohort_stats(self, cohort_id: str) -> dict | None:
        """Get statistics for a cohort."""
        if self._cohort_stats_store is not None:
            return self._cohort_stats_store.get_cohort_stats(cohort_id)

        cohort = self._cohort_stats.get(cohort_id)
        if not cohort:
            return None

        return {
            "cohort_id": cohort_id,
            "device_count": cohort.device_count,
            "metrics": {name: stats.to_dict() for name, stats in cohort.metrics.items()},
            "last_update": cohort.last_update.isoformat(),
        }

    def get_all_cohort_ids(self) -> list[str]:
        """Get all known cohort IDs."""
        if self._cohort_stats_store is not None:
            return self._cohort_stats_store.get_all_cohort_ids()
        return list(self._cohort_stats.keys())
