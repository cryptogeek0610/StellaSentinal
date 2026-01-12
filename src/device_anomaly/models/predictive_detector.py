"""
Predictive Anomaly Detection.

This module predicts future anomalies using time-series forecasting:
- Battery exhaustion before shift end
- Storage exhaustion within N days
- Network degradation trends
- Performance degradation trajectory
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PredictiveConfig:
    """Configuration for predictive anomaly detection."""

    # Battery prediction
    battery_critical_level: float = 10.0  # Percent
    default_shift_hours: float = 8.0

    # Storage prediction
    storage_critical_threshold: float = 0.95  # 95% utilized
    storage_warning_days: int = 7

    # Trend detection
    min_history_points: int = 5
    trend_window_days: int = 7

    # Confidence thresholds
    min_confidence: float = 0.5


@dataclass
class PredictiveResult:
    """Result of a predictive analysis."""

    prediction_type: str
    device_id: Any
    will_occur: bool
    predicted_value: Optional[float]
    time_until: Optional[float]  # Hours or days depending on type
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


class PredictiveAnomalyDetector:
    """
    Predicts future anomalies using time-series forecasting.

    Predictions include:
    1. Battery exhaustion before shift end
    2. Storage full within N days
    3. Network degradation trend
    4. Performance degradation trajectory
    """

    def __init__(self, config: Optional[PredictiveConfig] = None):
        self.config = config or PredictiveConfig()

    def predict_battery_failure(
        self,
        drain_history: np.ndarray,
        current_level: float,
        shift_duration_hours: Optional[float] = None,
        device_id: Any = None,
    ) -> PredictiveResult:
        """
        Predict if battery will fail during shift.

        Args:
            drain_history: Historical drain rate per hour
            current_level: Current battery percentage
            shift_duration_hours: Remaining shift time
            device_id: Device identifier

        Returns:
            PredictiveResult with failure prediction
        """
        shift_hours = shift_duration_hours or self.config.default_shift_hours

        if len(drain_history) < 2:
            return PredictiveResult(
                prediction_type="battery_failure",
                device_id=device_id,
                will_occur=False,
                predicted_value=None,
                time_until=None,
                confidence=0.0,
                details={"error": "Insufficient history"},
            )

        try:
            # Use exponential smoothing for trend estimation
            if len(drain_history) >= self.config.min_history_points:
                # Simple exponential smoothing
                alpha = 0.3
                smoothed = [drain_history[0]]
                for i in range(1, len(drain_history)):
                    smoothed.append(alpha * drain_history[i] + (1 - alpha) * smoothed[-1])
                avg_drain_per_hour = smoothed[-1]
            else:
                avg_drain_per_hour = np.mean(drain_history)

            # Predict end level
            predicted_end = current_level - (avg_drain_per_hour * shift_hours)

            # Hours until critical
            if avg_drain_per_hour > 0:
                hours_until_critical = (current_level - self.config.battery_critical_level) / avg_drain_per_hour
            else:
                hours_until_critical = float('inf')

            # Calculate confidence based on history length and variance
            history_factor = min(1.0, len(drain_history) / 14)  # More history = more confidence
            variance = np.std(drain_history) if len(drain_history) > 1 else 0
            variance_factor = 1.0 / (1.0 + variance / (avg_drain_per_hour + 0.1))
            confidence = min(0.95, history_factor * variance_factor * 0.9)

            will_fail = predicted_end < self.config.battery_critical_level

            recommendation = ""
            if will_fail and hours_until_critical < 2:
                recommendation = "Critical: Charge immediately"
            elif will_fail:
                recommendation = f"Charge within {hours_until_critical:.1f} hours"
            elif predicted_end < 20:
                recommendation = "Consider charging before end of shift"

            return PredictiveResult(
                prediction_type="battery_failure",
                device_id=device_id,
                will_occur=will_fail,
                predicted_value=max(0, predicted_end),
                time_until=hours_until_critical if hours_until_critical < float('inf') else None,
                confidence=confidence,
                details={
                    "current_level": current_level,
                    "avg_drain_per_hour": avg_drain_per_hour,
                    "shift_hours_remaining": shift_hours,
                    "predicted_end_level": max(0, predicted_end),
                },
                recommendation=recommendation,
            )

        except Exception as e:
            logger.error(f"Battery prediction failed: {e}")
            return PredictiveResult(
                prediction_type="battery_failure",
                device_id=device_id,
                will_occur=False,
                predicted_value=None,
                time_until=None,
                confidence=0.0,
                details={"error": str(e)},
            )

    def predict_storage_exhaustion(
        self,
        storage_history: np.ndarray,
        total_storage: float,
        device_id: Any = None,
    ) -> PredictiveResult:
        """
        Predict days until storage is full.

        Args:
            storage_history: Historical available storage values
            total_storage: Total device storage
            device_id: Device identifier

        Returns:
            PredictiveResult with exhaustion prediction
        """
        if len(storage_history) < self.config.min_history_points:
            return PredictiveResult(
                prediction_type="storage_exhaustion",
                device_id=device_id,
                will_occur=False,
                predicted_value=None,
                time_until=None,
                confidence=0.0,
                details={"error": "Insufficient history"},
            )

        try:
            # Linear regression for trend
            x = np.arange(len(storage_history))
            slope, intercept = np.polyfit(x, storage_history, 1)

            current = storage_history[-1]
            current_pct = current / total_storage * 100 if total_storage > 0 else 0

            if slope >= 0:
                # Storage increasing or stable
                return PredictiveResult(
                    prediction_type="storage_exhaustion",
                    device_id=device_id,
                    will_occur=False,
                    predicted_value=None,
                    time_until=None,
                    confidence=min(0.9, len(storage_history) / 30),
                    details={
                        "current_pct": current_pct,
                        "trend": "stable_or_increasing",
                        "trend_gb_per_day": slope / (1024**3) if slope > 0 else 0,
                    },
                    recommendation="Storage is stable",
                )

            # Days until zero
            days_until_zero = -current / slope if slope < 0 else float('inf')

            # Calculate confidence
            r_squared = 1 - (np.var(storage_history - (slope * x + intercept)) / np.var(storage_history))
            history_factor = min(1.0, len(storage_history) / 30)
            confidence = min(0.9, r_squared * history_factor)

            will_exhaust = days_until_zero < self.config.storage_warning_days

            recommendation = ""
            if days_until_zero < 1:
                recommendation = "Critical: Free storage immediately"
            elif days_until_zero < 3:
                recommendation = f"Warning: Storage will fill in ~{days_until_zero:.0f} days"
            elif will_exhaust:
                recommendation = f"Consider clearing storage within {days_until_zero:.0f} days"

            return PredictiveResult(
                prediction_type="storage_exhaustion",
                device_id=device_id,
                will_occur=will_exhaust,
                predicted_value=current_pct,
                time_until=days_until_zero,
                confidence=confidence,
                details={
                    "current_available": current,
                    "current_pct": current_pct,
                    "trend_bytes_per_day": abs(slope),
                    "trend_gb_per_day": abs(slope) / (1024**3),
                },
                recommendation=recommendation,
            )

        except Exception as e:
            logger.error(f"Storage prediction failed: {e}")
            return PredictiveResult(
                prediction_type="storage_exhaustion",
                device_id=device_id,
                will_occur=False,
                predicted_value=None,
                time_until=None,
                confidence=0.0,
                details={"error": str(e)},
            )

    def predict_network_degradation(
        self,
        signal_history: np.ndarray,
        drop_rate_history: np.ndarray,
        device_id: Any = None,
    ) -> PredictiveResult:
        """
        Predict network quality degradation trend.

        Args:
            signal_history: Historical signal strength (dBm)
            drop_rate_history: Historical connection drop rates
            device_id: Device identifier

        Returns:
            PredictiveResult with degradation prediction
        """
        if len(signal_history) < self.config.min_history_points:
            return PredictiveResult(
                prediction_type="network_degradation",
                device_id=device_id,
                will_occur=False,
                predicted_value=None,
                time_until=None,
                confidence=0.0,
                details={"error": "Insufficient history"},
            )

        try:
            # Analyze signal trend
            x = np.arange(len(signal_history))
            signal_slope, _ = np.polyfit(x, signal_history, 1)

            # Analyze drop rate trend
            drop_slope = 0.0
            if len(drop_rate_history) >= self.config.min_history_points:
                drop_slope, _ = np.polyfit(x[:len(drop_rate_history)], drop_rate_history, 1)

            # Degradation indicators
            signal_degrading = signal_slope < -0.5  # Losing 0.5 dBm per period
            drops_increasing = drop_slope > 0.01   # Drop rate increasing

            # Current quality
            current_signal = signal_history[-1]
            current_drops = drop_rate_history[-1] if len(drop_rate_history) > 0 else 0

            # Predict future signal
            periods_ahead = self.config.trend_window_days
            predicted_signal = current_signal + (signal_slope * periods_ahead)

            will_degrade = signal_degrading or drops_increasing

            # Confidence based on trend clarity
            signal_r2 = 1 - (np.var(signal_history - (signal_slope * x + signal_history[0])) / (np.var(signal_history) + 1e-6))
            confidence = min(0.85, abs(signal_r2) * 0.8 + 0.2)

            recommendation = ""
            if current_signal < -90:
                recommendation = "Poor signal quality - consider location change"
            elif signal_degrading:
                recommendation = "Signal degrading - monitor connectivity"
            elif drops_increasing:
                recommendation = "Connection stability declining"

            return PredictiveResult(
                prediction_type="network_degradation",
                device_id=device_id,
                will_occur=will_degrade,
                predicted_value=predicted_signal,
                time_until=None,
                confidence=confidence,
                details={
                    "current_signal_dbm": current_signal,
                    "current_drop_rate": current_drops,
                    "signal_trend_per_day": signal_slope,
                    "drop_rate_trend": drop_slope,
                    "predicted_signal_7d": predicted_signal,
                },
                recommendation=recommendation,
            )

        except Exception as e:
            logger.error(f"Network prediction failed: {e}")
            return PredictiveResult(
                prediction_type="network_degradation",
                device_id=device_id,
                will_occur=False,
                predicted_value=None,
                time_until=None,
                confidence=0.0,
                details={"error": str(e)},
            )

    def predict_device_dataframe(
        self,
        df: pd.DataFrame,
        battery_col: str = "BatteryLevel",
        battery_drain_col: str = "BatteryDrainPerHour",
        storage_col: str = "AvailableStorage",
        total_storage_col: str = "TotalStorage",
        signal_col: str = "AvgSignalStrength",
        drop_rate_col: str = "DropRate",
    ) -> pd.DataFrame:
        """
        Generate predictions for all devices in a dataframe.

        Args:
            df: DataFrame with device metrics
            battery_col: Battery level column
            battery_drain_col: Battery drain rate column
            storage_col: Available storage column
            total_storage_col: Total storage column
            signal_col: Signal strength column
            drop_rate_col: Drop rate column

        Returns:
            DataFrame with prediction columns added
        """
        if "DeviceId" not in df.columns:
            logger.warning("No DeviceId column, returning unchanged")
            return df

        df = df.copy()

        # Initialize prediction columns
        df["battery_will_fail"] = False
        df["battery_hours_remaining"] = None
        df["storage_will_exhaust"] = False
        df["storage_days_remaining"] = None
        df["network_degrading"] = False

        for device_id, grp in df.groupby("DeviceId"):
            grp = grp.sort_values("Timestamp") if "Timestamp" in grp.columns else grp

            # Battery prediction
            if battery_drain_col in grp.columns and battery_col in grp.columns:
                drain_history = grp[battery_drain_col].dropna().values
                current_level = grp[battery_col].iloc[-1] if not grp[battery_col].isna().all() else 100

                battery_pred = self.predict_battery_failure(
                    drain_history=drain_history,
                    current_level=current_level,
                    device_id=device_id,
                )

                mask = df["DeviceId"] == device_id
                df.loc[mask, "battery_will_fail"] = battery_pred.will_occur
                df.loc[mask, "battery_hours_remaining"] = battery_pred.time_until

            # Storage prediction
            if storage_col in grp.columns and total_storage_col in grp.columns:
                storage_history = grp[storage_col].dropna().values
                total = grp[total_storage_col].iloc[-1] if not grp[total_storage_col].isna().all() else 1

                storage_pred = self.predict_storage_exhaustion(
                    storage_history=storage_history,
                    total_storage=total,
                    device_id=device_id,
                )

                mask = df["DeviceId"] == device_id
                df.loc[mask, "storage_will_exhaust"] = storage_pred.will_occur
                df.loc[mask, "storage_days_remaining"] = storage_pred.time_until

            # Network prediction
            if signal_col in grp.columns:
                signal_history = grp[signal_col].dropna().values
                drop_history = grp[drop_rate_col].dropna().values if drop_rate_col in grp.columns else np.array([])

                network_pred = self.predict_network_degradation(
                    signal_history=signal_history,
                    drop_rate_history=drop_history,
                    device_id=device_id,
                )

                mask = df["DeviceId"] == device_id
                df.loc[mask, "network_degrading"] = network_pred.will_occur

        return df


def create_predictive_detector(
    battery_critical_level: float = 10.0,
    storage_warning_days: int = 7,
) -> PredictiveAnomalyDetector:
    """
    Create a predictive anomaly detector.

    Args:
        battery_critical_level: Battery % considered critical
        storage_warning_days: Days ahead to warn about storage

    Returns:
        Configured PredictiveAnomalyDetector
    """
    config = PredictiveConfig(
        battery_critical_level=battery_critical_level,
        storage_warning_days=storage_warning_days,
    )
    return PredictiveAnomalyDetector(config=config)
