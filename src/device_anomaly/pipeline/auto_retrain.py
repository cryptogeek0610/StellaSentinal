"""
Automatic Model Retraining Pipeline.

This module orchestrates automatic retraining based on:
- Feature drift detection (PSI thresholds)
- Anomaly rate deviations from baseline
- Time-based retraining schedules
- Manual triggers
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrainingConfig:
    """Configuration for automatic retraining."""

    # Drift thresholds
    psi_trigger_threshold: float = 0.25
    feature_drift_ratio_trigger: float = 0.30
    min_features_drifted: int = 5

    # Performance thresholds
    anomaly_rate_deviation_trigger: float = 0.50

    # Timing constraints
    min_days_between_retraining: int = 7
    max_days_without_retraining: int = 30

    # Data requirements
    min_training_samples: int = 10000
    training_lookback_days: int = 90

    # Model output
    models_dir: Path = field(default_factory=lambda: Path("models/production"))


@dataclass
class RetrainingTrigger:
    """Result of evaluating retraining triggers."""

    should_retrain: bool
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)
    priority: str = "normal"  # low, normal, high, critical


class AutoRetrainingOrchestrator:
    """
    Orchestrates automatic model retraining based on drift detection.

    Monitors drift metrics and triggers retraining when thresholds are exceeded.
    Maintains state about last retraining and baseline metrics.
    """

    def __init__(self, config: Optional[RetrainingConfig] = None):
        self.config = config or RetrainingConfig()
        self.last_retrain_date: Optional[datetime] = None
        self.baseline_anomaly_rate: Optional[float] = None
        self.baseline_feature_distributions: dict[str, dict] = {}
        self._pending_retrain: bool = False

    def evaluate_triggers(
        self,
        drift_metrics: dict[str, Any],
        current_anomaly_rate: float,
        current_date: Optional[datetime] = None,
    ) -> RetrainingTrigger:
        """
        Evaluate if retraining is needed.

        Args:
            drift_metrics: Drift metrics from DriftMonitor (PSI scores, etc.)
            current_anomaly_rate: Current anomaly detection rate
            current_date: Current date (defaults to now)

        Returns:
            RetrainingTrigger with decision and reason
        """
        current_date = current_date or datetime.now(timezone.utc)

        # Check time-based triggers first
        time_trigger = self._check_time_triggers(current_date)
        if time_trigger.should_retrain:
            return time_trigger

        # Check drift-based triggers
        drift_trigger = self._check_drift_triggers(drift_metrics)
        if drift_trigger.should_retrain:
            return drift_trigger

        # Check anomaly rate triggers
        rate_trigger = self._check_anomaly_rate_triggers(current_anomaly_rate)
        if rate_trigger.should_retrain:
            return rate_trigger

        return RetrainingTrigger(
            should_retrain=False,
            reason="No triggers met",
            metrics={
                "days_since_retrain": self._days_since_retrain(current_date),
                "anomaly_rate": current_anomaly_rate,
                "drift_metrics": drift_metrics,
            }
        )

    def _days_since_retrain(self, current_date: datetime) -> Optional[int]:
        """Calculate days since last retraining."""
        if self.last_retrain_date is None:
            return None
        delta = current_date - self.last_retrain_date
        return delta.days

    def _check_time_triggers(self, current_date: datetime) -> RetrainingTrigger:
        """Check time-based retraining triggers."""
        if self.last_retrain_date is None:
            return RetrainingTrigger(
                should_retrain=True,
                reason="No previous training found",
                priority="high"
            )

        days_since = self._days_since_retrain(current_date)

        if days_since < self.config.min_days_between_retraining:
            return RetrainingTrigger(
                should_retrain=False,
                reason=f"Too soon since last retraining ({days_since} days)",
            )

        if days_since >= self.config.max_days_without_retraining:
            return RetrainingTrigger(
                should_retrain=True,
                reason=f"Max interval reached ({days_since} days)",
                priority="normal",
                metrics={"days_since_retrain": days_since}
            )

        return RetrainingTrigger(
            should_retrain=False,
            reason=f"Within normal retraining window ({days_since} days)"
        )

    def _check_drift_triggers(
        self,
        drift_metrics: dict[str, Any],
    ) -> RetrainingTrigger:
        """Check drift-based retraining triggers."""
        if not drift_metrics:
            return RetrainingTrigger(
                should_retrain=False,
                reason="No drift metrics available"
            )

        # Check PSI values
        psi_scores = drift_metrics.get("psi", {})
        high_psi_features = [
            f for f, score in psi_scores.items()
            if score > self.config.psi_trigger_threshold
        ]

        if len(high_psi_features) >= self.config.min_features_drifted:
            return RetrainingTrigger(
                should_retrain=True,
                reason=f"High PSI in {len(high_psi_features)} features",
                priority="high",
                metrics={
                    "high_psi_features": high_psi_features[:10],
                    "max_psi": max(psi_scores.values()) if psi_scores else 0,
                }
            )

        # Check feature drift ratio
        warn_features = drift_metrics.get("warn_features", [])
        total_features = drift_metrics.get("feature_count", 1)

        if total_features > 0:
            drift_ratio = len(warn_features) / total_features
            if drift_ratio >= self.config.feature_drift_ratio_trigger:
                return RetrainingTrigger(
                    should_retrain=True,
                    reason=f"Feature drift threshold exceeded ({drift_ratio:.1%})",
                    priority="high",
                    metrics={
                        "drift_ratio": drift_ratio,
                        "drifted_features": warn_features[:10],
                    }
                )

        return RetrainingTrigger(
            should_retrain=False,
            reason="Drift within acceptable limits"
        )

    def _check_anomaly_rate_triggers(
        self,
        current_rate: float,
    ) -> RetrainingTrigger:
        """Check anomaly rate deviation triggers."""
        if self.baseline_anomaly_rate is None:
            # Set baseline if not set
            self.baseline_anomaly_rate = current_rate
            return RetrainingTrigger(
                should_retrain=False,
                reason="Setting initial anomaly rate baseline"
            )

        if self.baseline_anomaly_rate == 0:
            return RetrainingTrigger(
                should_retrain=False,
                reason="Baseline rate is zero, skipping rate check"
            )

        rate_deviation = abs(current_rate - self.baseline_anomaly_rate) / self.baseline_anomaly_rate

        if rate_deviation >= self.config.anomaly_rate_deviation_trigger:
            return RetrainingTrigger(
                should_retrain=True,
                reason=f"Anomaly rate deviation ({rate_deviation:.1%})",
                priority="normal",
                metrics={
                    "current_rate": current_rate,
                    "baseline_rate": self.baseline_anomaly_rate,
                    "deviation": rate_deviation,
                }
            )

        return RetrainingTrigger(
            should_retrain=False,
            reason="Anomaly rate within acceptable range"
        )

    async def trigger_retraining(
        self,
        reason: str,
        priority: str = "normal",
    ) -> Optional[str]:
        """
        Initiate the retraining pipeline.

        Args:
            reason: Reason for retraining (for logging)
            priority: Priority level (low, normal, high, critical)

        Returns:
            Job ID if retraining was queued, None otherwise
        """
        logger.info(f"Triggering retraining: {reason} (priority: {priority})")

        try:
            # Import here to avoid circular imports
            from device_anomaly.workers.ml_worker import MLWorker

            worker = MLWorker()

            # Calculate date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=self.config.training_lookback_days)

            job_id = await worker.queue_training_job(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                reason=reason,
                priority=priority,
            )

            logger.info(f"Retraining job queued: {job_id}")
            self._pending_retrain = True
            return job_id

        except Exception as e:
            logger.error(f"Failed to queue retraining job: {e}")
            return None

    def on_training_complete(
        self,
        success: bool,
        metrics: Optional[dict] = None,
    ) -> None:
        """
        Callback when training completes.

        Args:
            success: Whether training was successful
            metrics: Training metrics (anomaly_rate, etc.)
        """
        self._pending_retrain = False

        if success:
            self.last_retrain_date = datetime.now(timezone.utc)
            if metrics and "anomaly_rate" in metrics:
                self.baseline_anomaly_rate = metrics["anomaly_rate"]
            logger.info(f"Retraining completed successfully at {self.last_retrain_date}")
        else:
            logger.warning("Retraining failed")

    def get_status(self) -> dict[str, Any]:
        """Get current retraining status."""
        return {
            "last_retrain_date": self.last_retrain_date.isoformat() if self.last_retrain_date else None,
            "baseline_anomaly_rate": self.baseline_anomaly_rate,
            "pending_retrain": self._pending_retrain,
            "config": {
                "psi_threshold": self.config.psi_trigger_threshold,
                "max_days_interval": self.config.max_days_without_retraining,
                "min_days_interval": self.config.min_days_between_retraining,
            }
        }

    def update_baseline(
        self,
        anomaly_rate: Optional[float] = None,
        feature_distributions: Optional[dict[str, dict]] = None,
    ) -> None:
        """
        Update baseline metrics.

        Args:
            anomaly_rate: New baseline anomaly rate
            feature_distributions: New baseline feature distributions
        """
        if anomaly_rate is not None:
            self.baseline_anomaly_rate = anomaly_rate
            logger.info(f"Updated baseline anomaly rate to {anomaly_rate:.4f}")

        if feature_distributions is not None:
            self.baseline_feature_distributions = feature_distributions
            logger.info(f"Updated baseline distributions for {len(feature_distributions)} features")


def create_auto_retraining_orchestrator(
    psi_threshold: float = 0.25,
    max_days_interval: int = 30,
    min_days_interval: int = 7,
) -> AutoRetrainingOrchestrator:
    """
    Create an auto-retraining orchestrator.

    Args:
        psi_threshold: PSI threshold for drift detection
        max_days_interval: Maximum days between retraining
        min_days_interval: Minimum days between retraining

    Returns:
        Configured AutoRetrainingOrchestrator
    """
    config = RetrainingConfig(
        psi_trigger_threshold=psi_threshold,
        max_days_without_retraining=max_days_interval,
        min_days_between_retraining=min_days_interval,
    )
    return AutoRetrainingOrchestrator(config=config)
