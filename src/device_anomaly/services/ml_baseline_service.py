"""
ML Baseline Service.

High-level service that integrates the ultra-advanced ML baseline engine
with the existing baseline management system. Provides unified API for:

1. Training ML-enhanced baselines from all data sources
2. Scoring devices with ensemble anomaly detection
3. Real-time baseline updates via online learning
4. Drift detection and automatic retraining triggers
5. Causal correlation insights
6. Integration with existing baseline storage

Usage:
    from device_anomaly.services.ml_baseline_service import MLBaselineService

    service = MLBaselineService()

    # Train from all available data
    await service.train_from_all_sources(tenant_id="customer_1")

    # Score new telemetry
    results = await service.score_telemetry(df)

    # Get insights
    insights = service.get_correlation_insights()
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from device_anomaly.models.ml_baseline_engine import (
    MLBaselineConfig,
    MLBaselineEngine,
)
from device_anomaly.services.correlation_service import (
    METRIC_DOMAINS,
    CorrelationService,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class MLBaselineServiceConfig:
    """Configuration for the ML baseline service."""

    # Data sources
    enable_xsight: bool = True
    enable_mobicontrol: bool = True
    enable_custom_telemetry: bool = True

    # ML engine settings
    ensemble_weights: dict[str, float] = None
    contamination: float = 0.05
    min_training_samples: int = 100

    # Baseline management
    auto_save_interval_hours: int = 24
    baseline_version: str = "ml_ultra_v1"
    checkpoint_dir: str = "models/ml_baselines"

    # Drift detection
    drift_check_interval_hours: int = 6
    auto_retrain_on_drift: bool = True
    drift_threshold: float = 0.15

    # Real-time updates
    enable_online_learning: bool = True
    online_batch_size: int = 100

    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                "isolation_forest": 0.35,
                "lof": 0.25,
                "autoencoder": 0.25,
                "dbscan": 0.15,
            }


# =============================================================================
# ML BASELINE SERVICE
# =============================================================================


class MLBaselineService:
    """
    High-level service for ML-enhanced baseline management.

    Integrates with:
    - XSight data warehouse (telemetry)
    - MobiControl database (device metadata)
    - Custom telemetry sources
    - Existing baseline storage system
    """

    def __init__(
        self,
        config: MLBaselineServiceConfig | None = None,
        ml_config: MLBaselineConfig | None = None,
    ):
        self.config = config or MLBaselineServiceConfig()
        self.ml_config = ml_config or MLBaselineConfig(
            anomaly_contamination=self.config.contamination,
            min_samples_for_baseline=self.config.min_training_samples,
            drift_threshold_psi=self.config.drift_threshold,
            online_batch_size=self.config.online_batch_size,
        )

        self.engine = MLBaselineEngine(self.ml_config)
        self.correlation_service = CorrelationService()

        self._is_initialized = False
        self._last_train_time: datetime | None = None
        self._last_drift_check: datetime | None = None

        # Performance tracking
        self._training_history: list[dict[str, Any]] = []
        self._drift_history: list[dict[str, Any]] = []

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    async def load_training_data(
        self,
        tenant_id: str | None = None,
        lookback_days: int = 90,
        sources: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load training data from all configured sources.

        Args:
            tenant_id: Optional tenant filter
            lookback_days: Days of historical data to load
            sources: Specific sources to load (default: all enabled)

        Returns:
            Unified DataFrame with data from all sources
        """
        sources = sources or []
        if not sources:
            if self.config.enable_xsight:
                sources.append("xsight")
            if self.config.enable_mobicontrol:
                sources.append("mobicontrol")
            if self.config.enable_custom_telemetry:
                sources.append("custom")

        logger.info(f"Loading training data from sources: {sources}")

        dfs = {}

        # Load from each source
        for source in sources:
            try:
                if source == "xsight":
                    df = await self._load_xsight_data(tenant_id, lookback_days)
                elif source == "mobicontrol":
                    df = await self._load_mobicontrol_data(tenant_id, lookback_days)
                elif source == "custom":
                    df = await self._load_custom_telemetry(tenant_id, lookback_days)
                else:
                    logger.warning(f"Unknown source: {source}")
                    continue

                if df is not None and not df.empty:
                    dfs[source] = df
                    logger.info(f"Loaded {len(df)} rows from {source}")

            except Exception as e:
                logger.error(f"Failed to load data from {source}: {e}")

        # Fuse data sources
        if len(dfs) == 0:
            logger.warning("No data loaded from any source")
            return pd.DataFrame()

        if len(dfs) == 1:
            return list(dfs.values())[0]

        # Use multi-source fuser
        fused = self.engine.data_fuser.fuse(dfs)
        fused = self.engine.data_fuser.impute_missing(fused, method="knn")

        logger.info(f"Fused dataset: {len(fused)} rows, {len(fused.columns)} columns")

        return fused

    async def _load_xsight_data(
        self,
        tenant_id: str | None,
        lookback_days: int,
    ) -> pd.DataFrame | None:
        """Load data from XSight data warehouse."""
        try:
            from device_anomaly.data_access.unified_loader import UnifiedTelemetryLoader

            loader = UnifiedTelemetryLoader()
            cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

            df = await loader.load_telemetry(
                tenant_id=tenant_id,
                start_date=cutoff,
                end_date=datetime.now(UTC),
            )

            return df

        except ImportError:
            logger.warning("UnifiedTelemetryLoader not available")
            return None
        except Exception as e:
            logger.error(f"XSight data load failed: {e}")
            return None

    async def _load_mobicontrol_data(
        self,
        tenant_id: str | None,
        lookback_days: int,
    ) -> pd.DataFrame | None:
        """Load data from MobiControl database."""
        try:
            from device_anomaly.data_access.mc_timeseries_loader import MCTimeseriesLoader

            loader = MCTimeseriesLoader()
            cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

            df = await loader.load_timeseries(
                tenant_id=tenant_id,
                start_date=cutoff,
            )

            return df

        except ImportError:
            logger.warning("MCTimeseriesLoader not available")
            return None
        except Exception as e:
            logger.error(f"MobiControl data load failed: {e}")
            return None

    async def _load_custom_telemetry(
        self,
        tenant_id: str | None,
        lookback_days: int,
    ) -> pd.DataFrame | None:
        """Load custom telemetry data."""
        # Placeholder for custom data sources
        return None

    # =========================================================================
    # TRAINING
    # =========================================================================

    async def train(
        self,
        df: pd.DataFrame | None = None,
        tenant_id: str | None = None,
        lookback_days: int = 90,
        feature_cols: list[str] | None = None,
        metric_cols: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Train the ML baseline engine.

        Args:
            df: Training data (if None, loads from sources)
            tenant_id: Tenant filter
            lookback_days: Days of data to use
            feature_cols: Features for anomaly detection
            metric_cols: Metrics for baseline tracking

        Returns:
            Training results dictionary
        """
        start_time = datetime.now(UTC)

        # Load data if not provided
        if df is None or df.empty:
            df = await self.load_training_data(tenant_id, lookback_days)

        if df.empty:
            return {
                "success": False,
                "error": "No training data available",
                "timestamp": start_time.isoformat(),
            }

        logger.info(f"Training ML baseline engine on {len(df)} samples")

        try:
            # Train the engine
            self.engine.fit(df, feature_cols, metric_cols)

            # Save checkpoint
            checkpoint_path = self._get_checkpoint_path(tenant_id)
            self.engine.export_state(checkpoint_path)

            # Update tracking
            self._last_train_time = datetime.now(UTC)
            self._is_initialized = True

            duration = (datetime.now(UTC) - start_time).total_seconds()

            result = {
                "success": True,
                "samples_trained": len(df),
                "feature_count": len(self.engine._feature_cols),
                "metric_count": len(self.engine._metric_cols),
                "duration_seconds": duration,
                "timestamp": start_time.isoformat(),
                "checkpoint_path": str(checkpoint_path),
            }

            self._training_history.append(result)

            return result

        except Exception as e:
            logger.exception("Training failed")
            return {
                "success": False,
                "error": str(e),
                "timestamp": start_time.isoformat(),
            }

    async def train_from_all_sources(
        self,
        tenant_id: str | None = None,
        lookback_days: int = 90,
    ) -> dict[str, Any]:
        """
        Convenience method to train from all available data sources.

        This is the main entry point for "ultra" baseline training.
        """
        return await self.train(
            tenant_id=tenant_id,
            lookback_days=lookback_days,
        )

    # =========================================================================
    # SCORING
    # =========================================================================

    async def score_telemetry(
        self,
        df: pd.DataFrame,
        include_bayesian: bool = True,
        include_correlations: bool = True,
    ) -> pd.DataFrame:
        """
        Score telemetry data using the ML engine.

        Returns DataFrame with:
        - ensemble_score: Combined anomaly score (0-1)
        - ensemble_anomaly: Binary anomaly label
        - Per-metric anomaly probabilities
        - Correlation-based insights
        """
        if not self._is_initialized:
            logger.warning("Engine not initialized, using default scoring")
            return df

        # Score with ensemble
        result = self.engine.score(df)

        # Add correlation-based features
        if include_correlations and len(df) >= 30:
            try:
                corr_result = self.correlation_service.compute_correlation_matrix(df)
                strong_corrs = corr_result.get("strong_correlations", [])

                # Add correlation anomaly indicator
                result["correlation_anomaly_count"] = 0
                for corr in strong_corrs:
                    if corr.get("is_significant") and abs(corr["correlation"]) > 0.7:
                        # Check if correlation pattern deviates from expected
                        # This is a simplified check
                        result["correlation_anomaly_count"] += 1

            except Exception as e:
                logger.warning(f"Correlation scoring failed: {e}")

        return result

    def score_single_device(
        self,
        device_data: dict[str, float],
    ) -> dict[str, Any]:
        """
        Score a single device's metrics.

        Returns anomaly assessment for the device.
        """
        if not self._is_initialized:
            return {"error": "Engine not initialized"}

        result = {
            "device_id": device_data.get("DeviceId"),
            "timestamp": datetime.now(UTC).isoformat(),
            "metrics": {},
            "overall_anomaly_score": 0.0,
            "is_anomaly": False,
            "anomaly_type": "normal",
        }

        anomaly_scores = []

        for metric, value in device_data.items():
            if metric in ["DeviceId", "Timestamp"]:
                continue

            if not isinstance(value, (int, float)) or not np.isfinite(value):
                continue

            # Get Bayesian probability
            prob, severity = self.engine.bayesian_adapter.get_anomaly_probability(
                metric, float(value)
            )

            # Get baseline
            baseline = self.engine.online_baseline.get_baseline(metric)

            result["metrics"][metric] = {
                "value": value,
                "anomaly_probability": prob,
                "severity": severity,
                "baseline_mean": baseline["mean"] if baseline else None,
                "baseline_std": baseline["std"] if baseline else None,
            }

            if prob > 0.8:  # High anomaly probability
                anomaly_scores.append(prob)

        # Compute overall score
        if anomaly_scores:
            result["overall_anomaly_score"] = float(np.mean(anomaly_scores))
            result["is_anomaly"] = result["overall_anomaly_score"] > 0.8
            result["anomaly_type"] = self._classify_anomaly(result["metrics"])

        return result

    def _classify_anomaly(self, metrics: dict[str, dict]) -> str:
        """Classify the type of anomaly based on affected metrics."""
        anomalous_metrics = [
            m for m, data in metrics.items()
            if data.get("severity") in ("critical", "warning")
        ]

        if not anomalous_metrics:
            return "normal"

        # Check domains
        domains = set()
        for metric in anomalous_metrics:
            for domain, domain_metrics in METRIC_DOMAINS.items():
                if metric in domain_metrics:
                    domains.add(domain)
                    break

        if "battery" in domains:
            return "battery_anomaly"
        elif "rf" in domains:
            return "connectivity_anomaly"
        elif "usage" in domains:
            return "usage_anomaly"
        elif "storage" in domains:
            return "storage_anomaly"
        else:
            return "multi_domain_anomaly"

    # =========================================================================
    # ONLINE UPDATES
    # =========================================================================

    async def update_baselines(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "Timestamp",
    ) -> dict[str, Any]:
        """
        Update baselines with new streaming data.

        Uses online learning for real-time adaptation.
        """
        if not self._is_initialized:
            logger.warning("Engine not initialized, skipping update")
            return {"error": "Engine not initialized"}

        if not self.config.enable_online_learning:
            return {"skipped": True, "reason": "Online learning disabled"}

        return self.engine.update_online(df, timestamp_col)

    # =========================================================================
    # DRIFT DETECTION
    # =========================================================================

    async def check_drift(
        self,
        df: pd.DataFrame | None = None,
        tenant_id: str | None = None,
        lookback_days: int = 7,
    ) -> dict[str, Any]:
        """
        Check for distribution drift in the data.

        If significant drift is detected and auto_retrain is enabled,
        triggers automatic retraining.
        """
        if not self._is_initialized:
            return {"error": "Engine not initialized"}

        # Load recent data if not provided
        if df is None:
            df = await self.load_training_data(tenant_id, lookback_days)

        if df.empty:
            return {"error": "No data for drift check"}

        # Run drift detection
        drift_report = self.engine.check_drift(df)

        # Track drift
        self._drift_history.append({
            "timestamp": datetime.now(UTC).isoformat(),
            "drift_rate": drift_report.get("drift_rate", 0),
            "metrics_drifted": drift_report.get("metrics_drifted", 0),
        })

        self._last_drift_check = datetime.now(UTC)

        # Auto-retrain if significant drift
        if (self.config.auto_retrain_on_drift and
            drift_report.get("drift_rate", 0) > 0.3):
            logger.warning("Significant drift detected, triggering retraining")
            retrain_result = await self.train(tenant_id=tenant_id, lookback_days=lookback_days)
            drift_report["auto_retrained"] = retrain_result.get("success", False)

        return drift_report

    # =========================================================================
    # INSIGHTS & CORRELATIONS
    # =========================================================================

    def get_correlation_insights(self) -> list[dict[str, Any]]:
        """Get discovered causal relationships and correlation insights."""
        if not self._is_initialized:
            return []

        return self.engine.get_causal_insights()

    def get_baseline_suggestions(
        self,
        df: pd.DataFrame,
        z_threshold: float = 3.0,
    ) -> list[dict[str, Any]]:
        """
        Get baseline adjustment suggestions based on recent anomalies.

        Combines ML-based detection with statistical analysis.
        """
        suggestions = []

        for metric in self.engine._metric_cols:
            if metric not in df.columns:
                continue

            # Get current baseline
            baseline = self.engine.online_baseline.get_baseline(metric)
            if baseline is None:
                continue

            # Analyze recent values
            values = df[metric].dropna()
            if len(values) < 10:
                continue

            current_median = float(values.median())
            baseline_mean = baseline["mean"]
            baseline_std = baseline["std"]

            # Check for significant drift
            z = abs(current_median - baseline_mean) / (baseline_std + 1e-10)

            if z >= z_threshold:
                # Get Bayesian stats
                bayesian_stats = self.engine.bayesian_adapter.metric_stats.get(metric)

                suggestion = {
                    "metric": metric,
                    "current_baseline_mean": baseline_mean,
                    "current_baseline_std": baseline_std,
                    "observed_median": current_median,
                    "z_score": float(z),
                    "proposed_adjustment": float((current_median - baseline_mean) * 0.5),
                    "confidence": 1.0 - (1.0 / (z + 1)),
                    "rationale": f"Metric drifted {z:.1f} standard deviations from baseline",
                }

                if bayesian_stats:
                    suggestion["bayesian_uncertainty"] = bayesian_stats.uncertainty
                    suggestion["credible_interval"] = [
                        bayesian_stats.credible_interval_lower,
                        bayesian_stats.credible_interval_upper,
                    ]

                suggestions.append(suggestion)

        # Sort by z-score (most significant first)
        suggestions.sort(key=lambda x: x["z_score"], reverse=True)

        return suggestions

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _get_checkpoint_path(self, tenant_id: str | None = None) -> Path:
        """Get path for checkpoint file."""
        base_dir = Path(self.config.checkpoint_dir)
        if tenant_id:
            return base_dir / tenant_id / "ml_baseline_state.json"
        return base_dir / "ml_baseline_state.json"

    async def load_checkpoint(
        self,
        tenant_id: str | None = None,
    ) -> bool:
        """Load engine state from checkpoint."""
        path = self._get_checkpoint_path(tenant_id)

        if not path.exists():
            logger.info(f"No checkpoint found at {path}")
            return False

        try:
            self.engine.import_state(path)
            self._is_initialized = True
            logger.info(f"Loaded checkpoint from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    async def save_checkpoint(
        self,
        tenant_id: str | None = None,
    ) -> bool:
        """Save engine state to checkpoint."""
        if not self._is_initialized:
            logger.warning("Engine not initialized, nothing to save")
            return False

        try:
            path = self._get_checkpoint_path(tenant_id)
            self.engine.export_state(path)
            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    # =========================================================================
    # STATUS & METRICS
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """Get current service status."""
        return {
            "initialized": self._is_initialized,
            "last_train_time": self._last_train_time.isoformat() if self._last_train_time else None,
            "last_drift_check": self._last_drift_check.isoformat() if self._last_drift_check else None,
            "feature_count": len(self.engine._feature_cols) if self._is_initialized else 0,
            "metric_count": len(self.engine._metric_cols) if self._is_initialized else 0,
            "training_history_count": len(self._training_history),
            "drift_history_count": len(self._drift_history),
            "config": asdict(self.config),
        }

    def get_training_history(self) -> list[dict[str, Any]]:
        """Get training history."""
        return self._training_history

    def get_drift_history(self) -> list[dict[str, Any]]:
        """Get drift detection history."""
        return self._drift_history


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def create_ml_baseline_service(
    tenant_id: str | None = None,
    auto_train: bool = True,
    lookback_days: int = 90,
) -> MLBaselineService:
    """
    Factory function to create and optionally train the ML baseline service.

    Args:
        tenant_id: Tenant filter
        auto_train: Whether to automatically train on creation
        lookback_days: Days of historical data for training

    Returns:
        Initialized MLBaselineService
    """
    service = MLBaselineService()

    # Try to load existing checkpoint
    loaded = await service.load_checkpoint(tenant_id)

    # Train if no checkpoint or auto_train is True
    if auto_train and not loaded:
        await service.train_from_all_sources(tenant_id, lookback_days)

    return service


async def score_device_telemetry(
    df: pd.DataFrame,
    tenant_id: str | None = None,
) -> pd.DataFrame:
    """
    Convenience function to score device telemetry.

    Creates service, trains if needed, and scores data.
    """
    service = await create_ml_baseline_service(tenant_id, auto_train=True)
    return await service.score_telemetry(df)
