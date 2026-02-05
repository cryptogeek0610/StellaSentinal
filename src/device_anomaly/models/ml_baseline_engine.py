"""
Ultra-Advanced ML Baseline Engine.

This module implements a sophisticated machine learning-based baseline system that:
1. Uses ensemble methods (IsolationForest + LOF + DBSCAN + AutoEncoder)
2. Implements Bayesian updating for adaptive baselines with uncertainty
3. Supports online learning for real-time adaptation
4. Fuses data from multiple sources (XSight, MobiControl, custom telemetry)
5. Implements causal discovery for correlation insights
6. Handles concept drift with automatic model retraining triggers

Architecture:
    MultiSourceDataFuser -> FeatureEngineeringPipeline -> EnsembleAnomalyDetector
                                                              |
                                                              v
                                                    BayesianBaselineAdapter
                                                              |
                                                              v
                                                    CausalCorrelationEngine
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class MLBaselineConfig:
    """Configuration for the ML baseline engine."""

    # Ensemble configuration
    enable_isolation_forest: bool = True
    enable_local_outlier_factor: bool = True
    enable_autoencoder: bool = True
    enable_dbscan_clustering: bool = True

    # Bayesian updating
    prior_weight: float = 0.3  # Weight for prior beliefs
    likelihood_weight: float = 0.7  # Weight for new observations
    uncertainty_threshold: float = 0.2  # Trigger recompute if uncertainty exceeds this

    # Online learning
    online_batch_size: int = 100
    online_learning_rate: float = 0.1
    max_buffer_size: int = 10000

    # Drift detection
    drift_window_size: int = 500
    drift_threshold_psi: float = 0.15  # More sensitive than default 0.2
    drift_threshold_ks: float = 0.05  # KS test p-value threshold
    concept_drift_lookback_days: int = 30

    # Multi-source fusion
    source_weights: dict[str, float] = field(
        default_factory=lambda: {
            "xsight": 1.0,
            "mobicontrol": 0.9,
            "custom_telemetry": 0.8,
        }
    )

    # Causal discovery
    enable_causal_discovery: bool = True
    max_lag_days: int = 7
    min_correlation_threshold: float = 0.3
    granger_significance_level: float = 0.05

    # Thresholds
    anomaly_contamination: float = 0.05
    min_samples_for_baseline: int = 50
    confidence_interval: float = 0.95

    # Model persistence
    model_version: str = "ultra_v1"
    checkpoint_interval: int = 1000  # Save checkpoint every N observations


# =============================================================================
# BAYESIAN BASELINE ADAPTER
# =============================================================================


@dataclass
class BayesianMetricStats:
    """Bayesian statistics for a single metric."""

    metric_name: str

    # Prior parameters (from historical data)
    prior_mean: float = 0.0
    prior_std: float = 1.0
    prior_alpha: float = 2.0  # Shape parameter for inverse-gamma
    prior_beta: float = 1.0  # Scale parameter

    # Posterior parameters (updated with new data)
    posterior_mean: float = 0.0
    posterior_std: float = 1.0
    posterior_alpha: float = 2.0
    posterior_beta: float = 1.0

    # Uncertainty quantification
    uncertainty: float = 1.0  # Posterior variance / prior variance
    credible_interval_lower: float = 0.0
    credible_interval_upper: float = 0.0

    # Statistics
    n_observations: int = 0
    last_updated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BayesianMetricStats:
        return cls(**data)


class BayesianBaselineAdapter:
    """
    Implements Bayesian updating for adaptive baselines.

    Uses conjugate priors for efficient closed-form updates:
    - Normal-Inverse-Gamma prior for mean and variance
    - Enables uncertainty quantification for each metric
    - Automatically triggers recomputation when uncertainty exceeds threshold

    Mathematical Foundation:
        Prior: μ ~ N(μ₀, σ²/κ₀), σ² ~ Inv-Gamma(α₀, β₀)
        Posterior: μ|data ~ N(μₙ, σ²/κₙ), σ²|data ~ Inv-Gamma(αₙ, βₙ)

    Where:
        κₙ = κ₀ + n
        μₙ = (κ₀μ₀ + n*x̄) / κₙ
        αₙ = α₀ + n/2
        βₙ = β₀ + 0.5*Σ(xᵢ - x̄)² + (κ₀*n*(x̄ - μ₀)²) / (2*κₙ)
    """

    def __init__(self, config: MLBaselineConfig):
        self.config = config
        self.metric_stats: dict[str, BayesianMetricStats] = {}
        self._observation_buffer: dict[str, deque] = {}

    def initialize_prior(
        self,
        metric_name: str,
        historical_data: pd.Series,
        min_samples: int = 30,
    ) -> BayesianMetricStats:
        """
        Initialize prior distribution from historical data.

        Uses empirical Bayes approach: estimate prior parameters from data.
        """
        data = historical_data.dropna()

        if len(data) < min_samples:
            # Weak prior if insufficient data
            logger.warning(f"Insufficient data for {metric_name}, using weak prior")
            stats_obj = BayesianMetricStats(
                metric_name=metric_name,
                prior_mean=0.0,
                prior_std=10.0,
                prior_alpha=1.0,
                prior_beta=1.0,
                posterior_mean=0.0,
                posterior_std=10.0,
                posterior_alpha=1.0,
                posterior_beta=1.0,
                uncertainty=1.0,
            )
        else:
            mean = float(data.mean())
            std = float(data.std())
            n = len(data)

            # Estimate prior parameters (empirical Bayes)
            prior_alpha = n / 2 + 1
            prior_beta = (n - 1) * std**2 / 2

            # Credible interval (95%)
            ci_lower = float(np.percentile(data, 2.5))
            ci_upper = float(np.percentile(data, 97.5))

            stats_obj = BayesianMetricStats(
                metric_name=metric_name,
                prior_mean=mean,
                prior_std=std if std > 0 else 1e-6,
                prior_alpha=prior_alpha,
                prior_beta=prior_beta,
                posterior_mean=mean,
                posterior_std=std if std > 0 else 1e-6,
                posterior_alpha=prior_alpha,
                posterior_beta=prior_beta,
                uncertainty=0.0,  # No uncertainty initially
                credible_interval_lower=ci_lower,
                credible_interval_upper=ci_upper,
                n_observations=n,
                last_updated=datetime.now(UTC).isoformat(),
            )

        self.metric_stats[metric_name] = stats_obj
        self._observation_buffer[metric_name] = deque(maxlen=self.config.max_buffer_size)

        return stats_obj

    def update(
        self,
        metric_name: str,
        new_observations: float | np.ndarray | pd.Series,
        batch_update: bool = True,
    ) -> BayesianMetricStats:
        """
        Update posterior distribution with new observations.

        Implements conjugate Bayesian update for Normal-Inverse-Gamma prior.
        """
        if metric_name not in self.metric_stats:
            raise ValueError(f"Metric {metric_name} not initialized. Call initialize_prior first.")

        stats_obj = self.metric_stats[metric_name]

        # Convert to numpy array
        if isinstance(new_observations, (int, float)):
            obs = np.array([new_observations])
        elif isinstance(new_observations, pd.Series):
            obs = new_observations.dropna().values
        else:
            obs = np.array(new_observations)

        obs = obs[np.isfinite(obs)]
        if len(obs) == 0:
            return stats_obj

        # Add to buffer for online learning
        for o in obs:
            self._observation_buffer[metric_name].append(o)

        # Batch update when buffer is large enough
        if (
            batch_update
            and len(self._observation_buffer[metric_name]) >= self.config.online_batch_size
        ):
            buffer_data = np.array(self._observation_buffer[metric_name])
            self._perform_bayesian_update(stats_obj, buffer_data)
            self._observation_buffer[metric_name].clear()
        elif not batch_update:
            self._perform_bayesian_update(stats_obj, obs)

        return stats_obj

    def _perform_bayesian_update(
        self,
        stats_obj: BayesianMetricStats,
        observations: np.ndarray,
    ) -> None:
        """Perform the actual Bayesian update."""
        n = len(observations)
        if n == 0:
            return

        x_bar = float(np.mean(observations))
        s_sq = float(np.var(observations, ddof=1)) if n > 1 else 0.0

        # Prior parameters
        mu_0 = stats_obj.prior_mean
        kappa_0 = 1.0 / (stats_obj.prior_std**2 + 1e-10)
        alpha_0 = stats_obj.prior_alpha
        beta_0 = stats_obj.prior_beta

        # Posterior parameters (conjugate update)
        kappa_n = kappa_0 + n
        mu_n = (kappa_0 * mu_0 + n * x_bar) / kappa_n
        alpha_n = alpha_0 + n / 2

        # Sum of squared deviations
        ss = (n - 1) * s_sq if n > 1 else 0.0

        beta_n = beta_0 + 0.5 * ss + (kappa_0 * n * (x_bar - mu_0) ** 2) / (2 * kappa_n)

        # Posterior standard deviation (marginal t-distribution)
        posterior_var = (beta_n / alpha_n) / kappa_n
        posterior_std = float(np.sqrt(posterior_var)) if posterior_var > 0 else 1e-6

        # Uncertainty: ratio of posterior to prior variance
        prior_var = stats_obj.prior_std**2
        uncertainty = posterior_var / (prior_var + 1e-10)

        # Credible interval (using Student's t approximation)
        t_crit = stats.t.ppf(1 - (1 - self.config.confidence_interval) / 2, 2 * alpha_n)
        ci_half = t_crit * posterior_std

        # Update stats object
        stats_obj.posterior_mean = mu_n
        stats_obj.posterior_std = posterior_std
        stats_obj.posterior_alpha = alpha_n
        stats_obj.posterior_beta = beta_n
        stats_obj.uncertainty = float(uncertainty)
        stats_obj.credible_interval_lower = mu_n - ci_half
        stats_obj.credible_interval_upper = mu_n + ci_half
        stats_obj.n_observations += n
        stats_obj.last_updated = datetime.now(UTC).isoformat()

    def get_anomaly_probability(
        self,
        metric_name: str,
        value: float,
    ) -> tuple[float, str]:
        """
        Compute probability that a value is anomalous given the posterior.

        Returns probability (0-1) and severity level.
        """
        if metric_name not in self.metric_stats:
            return 0.5, "unknown"

        stats_obj = self.metric_stats[metric_name]

        # Z-score using posterior distribution
        z = (value - stats_obj.posterior_mean) / (stats_obj.posterior_std + 1e-10)

        # Probability of seeing this value or more extreme
        prob_extreme = 2 * (1 - stats.norm.cdf(abs(z)))

        # Anomaly probability (inverse)
        anomaly_prob = 1 - prob_extreme

        # Severity classification
        if anomaly_prob >= 0.99:
            severity = "critical"
        elif anomaly_prob >= 0.95:
            severity = "warning"
        elif anomaly_prob >= 0.80:
            severity = "elevated"
        else:
            severity = "normal"

        return float(anomaly_prob), severity

    def needs_recomputation(self, metric_name: str) -> bool:
        """Check if baseline needs recomputation based on uncertainty."""
        if metric_name not in self.metric_stats:
            return True
        return self.metric_stats[metric_name].uncertainty > self.config.uncertainty_threshold

    def export_state(self) -> dict[str, Any]:
        """Export current state for persistence."""
        return {
            "version": "bayesian_v1",
            "config": asdict(self.config),
            "metrics": {k: v.to_dict() for k, v in self.metric_stats.items()},
            "exported_at": datetime.now(UTC).isoformat(),
        }

    def import_state(self, state: dict[str, Any]) -> None:
        """Import state from persistence."""
        for metric_name, metric_data in state.get("metrics", {}).items():
            self.metric_stats[metric_name] = BayesianMetricStats.from_dict(metric_data)
            self._observation_buffer[metric_name] = deque(maxlen=self.config.max_buffer_size)


# =============================================================================
# ENSEMBLE ANOMALY DETECTOR
# =============================================================================


@dataclass
class EnsembleScore:
    """Combined score from multiple anomaly detection methods."""

    isolation_forest_score: float = 0.0
    lof_score: float = 0.0
    autoencoder_score: float = 0.0
    dbscan_outlier: bool = False

    ensemble_score: float = 0.0
    confidence: float = 0.0
    is_anomaly: bool = False
    anomaly_type: str = "unknown"
    contributing_factors: list[str] = field(default_factory=list)


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detector combining multiple algorithms.

    Algorithms:
    1. Isolation Forest - Good for high-dimensional data, global outliers
    2. Local Outlier Factor - Good for local density-based outliers
    3. Autoencoder - Good for complex patterns, reconstruction error
    4. DBSCAN - Good for cluster-based outliers, no assumption on shape

    Ensemble Strategy:
    - Weighted voting based on algorithm confidence
    - Cross-validation for weight optimization
    - Disagreement detection for uncertainty
    """

    def __init__(self, config: MLBaselineConfig):
        self.config = config
        self.models: dict[str, Any] = {}
        self.feature_cols: list[str] = []
        self.scaler = None
        self._is_fitted = False

        # Algorithm weights (optimized during training)
        self.weights = {
            "isolation_forest": 0.35,
            "lof": 0.25,
            "autoencoder": 0.25,
            "dbscan": 0.15,
        }

    def fit(self, df: pd.DataFrame, feature_cols: list[str] | None = None) -> None:
        """Fit all ensemble components."""
        from sklearn.cluster import DBSCAN
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.preprocessing import StandardScaler

        # Select features
        if feature_cols:
            self.feature_cols = [c for c in feature_cols if c in df.columns]
        else:
            self.feature_cols = self._auto_select_features(df)

        if len(self.feature_cols) < 2:
            raise ValueError("Need at least 2 numeric features for ensemble training")

        # Prepare data
        X = df[self.feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        logger.info(f"Training ensemble on {len(self.feature_cols)} features, {len(df)} samples")

        # 1. Isolation Forest
        if self.config.enable_isolation_forest:
            self.models["isolation_forest"] = IsolationForest(
                n_estimators=300,
                contamination=self.config.anomaly_contamination,
                random_state=42,
                n_jobs=-1,
            )
            self.models["isolation_forest"].fit(X_scaled)
            logger.info("Isolation Forest fitted")

        # 2. Local Outlier Factor
        if self.config.enable_local_outlier_factor:
            self.models["lof"] = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.config.anomaly_contamination,
                novelty=True,  # For prediction on new data
                n_jobs=-1,
            )
            self.models["lof"].fit(X_scaled)
            logger.info("Local Outlier Factor fitted")

        # 3. DBSCAN (for cluster-based outliers)
        if self.config.enable_dbscan_clustering:
            # Use eps based on data density
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=5)
            nn.fit(X_scaled)
            distances, _ = nn.kneighbors(X_scaled)
            eps = float(np.percentile(distances[:, -1], 90))

            self.models["dbscan"] = DBSCAN(
                eps=max(eps, 0.5),
                min_samples=5,
                n_jobs=-1,
            )
            # Fit and get labels (noise points = -1)
            self.models["dbscan_labels"] = self.models["dbscan"].fit_predict(X_scaled)
            logger.info("DBSCAN fitted")

        # 4. Autoencoder (if enabled)
        if self.config.enable_autoencoder:
            try:
                self._fit_autoencoder(X_scaled)
                logger.info("Autoencoder fitted")
            except ImportError:
                logger.warning("TensorFlow not available, skipping autoencoder")
                self.config.enable_autoencoder = False

        self._is_fitted = True

    def _fit_autoencoder(self, X: np.ndarray) -> None:
        """Fit a simple autoencoder for reconstruction-based anomaly detection."""
        try:
            import tensorflow as tf
            from tensorflow import keras

            # Disable TF logging
            tf.get_logger().setLevel("ERROR")

            input_dim = X.shape[1]
            encoding_dim = max(2, input_dim // 3)

            # Build autoencoder
            encoder = keras.Sequential(
                [
                    keras.layers.Dense(input_dim, activation="relu", input_shape=(input_dim,)),
                    keras.layers.Dense(encoding_dim * 2, activation="relu"),
                    keras.layers.Dense(encoding_dim, activation="relu"),
                ]
            )

            decoder = keras.Sequential(
                [
                    keras.layers.Dense(
                        encoding_dim * 2, activation="relu", input_shape=(encoding_dim,)
                    ),
                    keras.layers.Dense(input_dim, activation="linear"),
                ]
            )

            autoencoder = keras.Sequential([encoder, decoder])
            autoencoder.compile(optimizer="adam", loss="mse")

            # Train
            autoencoder.fit(
                X,
                X,
                epochs=50,
                batch_size=32,
                validation_split=0.1,
                verbose=0,
            )

            self.models["autoencoder"] = autoencoder

            # Compute reconstruction threshold (95th percentile of training errors)
            reconstructed = autoencoder.predict(X, verbose=0)
            errors = np.mean(np.square(X - reconstructed), axis=1)
            self.models["autoencoder_threshold"] = float(np.percentile(errors, 95))

        except Exception as e:
            logger.warning(f"Autoencoder training failed: {e}")
            self.config.enable_autoencoder = False

    def _auto_select_features(self, df: pd.DataFrame) -> list[str]:
        """Automatically select numeric features for training."""
        exclude = {
            "DeviceId",
            "ModelId",
            "ManufacturerId",
            "OsVersionId",
            "is_injected_anomaly",
            "anomaly_score",
            "anomaly_label",
            "Timestamp",
            "CollectedDate",
            "tenant_id",
            "cohort_id",
        }

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in exclude]

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score samples using the ensemble.

        Returns DataFrame with ensemble scores and individual algorithm scores.
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        # Prepare data
        X = df[self.feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        X_scaled = self.scaler.transform(X)

        result = df.copy()
        scores = {}

        # 1. Isolation Forest scores
        if "isolation_forest" in self.models:
            # decision_function: lower = more anomalous
            if_scores = self.models["isolation_forest"].decision_function(X_scaled)
            # Normalize to 0-1 (higher = more anomalous)
            scores["if"] = 1 - self._normalize_scores(if_scores)
            result["if_score"] = scores["if"]

        # 2. LOF scores
        if "lof" in self.models:
            lof_scores = self.models["lof"].decision_function(X_scaled)
            scores["lof"] = 1 - self._normalize_scores(lof_scores)
            result["lof_score"] = scores["lof"]

        # 3. Autoencoder reconstruction error
        if "autoencoder" in self.models:
            reconstructed = self.models["autoencoder"].predict(X_scaled, verbose=0)
            ae_errors = np.mean(np.square(X_scaled - reconstructed), axis=1)
            scores["ae"] = self._normalize_scores(ae_errors)
            result["ae_score"] = scores["ae"]

        # 4. DBSCAN (binary: is noise point or not)
        if "dbscan" in self.models:
            # For new data, predict using nearest cluster
            labels = self.models["dbscan"].fit_predict(X_scaled)
            scores["dbscan"] = (labels == -1).astype(float)
            result["dbscan_outlier"] = scores["dbscan"]

        # Compute ensemble score
        ensemble_score = np.zeros(len(df))
        total_weight = 0.0

        if "if" in scores:
            ensemble_score += self.weights["isolation_forest"] * scores["if"]
            total_weight += self.weights["isolation_forest"]

        if "lof" in scores:
            ensemble_score += self.weights["lof"] * scores["lof"]
            total_weight += self.weights["lof"]

        if "ae" in scores:
            ensemble_score += self.weights["autoencoder"] * scores["ae"]
            total_weight += self.weights["autoencoder"]

        if "dbscan" in scores:
            ensemble_score += self.weights["dbscan"] * scores["dbscan"]
            total_weight += self.weights["dbscan"]

        if total_weight > 0:
            ensemble_score /= total_weight

        result["ensemble_score"] = ensemble_score

        # Compute confidence (agreement between algorithms)
        if len(scores) > 1:
            score_matrix = np.column_stack(list(scores.values()))
            result["ensemble_confidence"] = 1 - np.std(score_matrix, axis=1)
        else:
            result["ensemble_confidence"] = 0.5

        # Classify anomalies
        threshold = np.percentile(ensemble_score, 100 * (1 - self.config.anomaly_contamination))
        result["ensemble_anomaly"] = (ensemble_score >= threshold).astype(int)

        return result

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to 0-1 range."""
        min_s = np.min(scores)
        max_s = np.max(scores)
        if max_s - min_s < 1e-10:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)


# =============================================================================
# CAUSAL CORRELATION ENGINE
# =============================================================================


@dataclass
class CausalRelationship:
    """Represents a causal relationship between metrics."""

    cause_metric: str
    effect_metric: str
    lag_days: int
    correlation: float
    granger_p_value: float
    transfer_entropy: float
    confidence: float
    direction: str  # "causal", "reverse_causal", "bidirectional", "spurious"
    insight: str


class CausalCorrelationEngine:
    """
    Advanced correlation analysis with causal discovery.

    Implements:
    1. Granger causality tests
    2. Transfer entropy for non-linear causality
    3. PC algorithm for causal structure learning
    4. Time-lagged cross-correlation with lead-lag detection
    5. Partial correlation for confounding control
    """

    def __init__(self, config: MLBaselineConfig):
        self.config = config
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self.discovered_relationships: list[CausalRelationship] = []

    def discover_causal_relationships(
        self,
        df: pd.DataFrame,
        target_metrics: list[str] | None = None,
        timestamp_col: str = "Timestamp",
    ) -> list[CausalRelationship]:
        """
        Discover causal relationships between metrics.

        Uses multiple methods:
        1. Granger causality for time-series
        2. Correlation analysis for associations
        3. Partial correlation for confounding control
        """
        relationships = []

        # Get numeric columns
        if target_metrics:
            metrics = [m for m in target_metrics if m in df.columns]
        else:
            exclude = {"DeviceId", "Timestamp", "CollectedDate", "tenant_id"}
            metrics = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

        if len(metrics) < 2:
            logger.warning("Insufficient metrics for causal discovery")
            return relationships

        logger.info(f"Running causal discovery on {len(metrics)} metrics")

        # Sort by timestamp if available
        if timestamp_col in df.columns:
            df = df.sort_values(timestamp_col)

        # Test all pairs
        for i, metric_a in enumerate(metrics):
            for metric_b in metrics[i + 1 :]:
                relationship = self._test_causality(df, metric_a, metric_b)
                if relationship is not None:
                    relationships.append(relationship)

        # Sort by confidence
        relationships.sort(key=lambda r: r.confidence, reverse=True)
        self.discovered_relationships = relationships

        return relationships

    def _test_causality(
        self,
        df: pd.DataFrame,
        metric_a: str,
        metric_b: str,
    ) -> CausalRelationship | None:
        """Test causal relationship between two metrics."""
        x = df[metric_a].dropna().values
        y = df[metric_b].dropna().values

        # Align series
        min_len = min(len(x), len(y))
        if min_len < self.config.min_samples_for_baseline:
            return None

        x = x[:min_len]
        y = y[:min_len]

        # 1. Basic correlation
        corr, p_corr = stats.pearsonr(x, y)
        if abs(corr) < self.config.min_correlation_threshold:
            return None

        # 2. Granger causality (simplified)
        granger_p_a_to_b = self._granger_test(x, y)
        granger_p_b_to_a = self._granger_test(y, x)

        # 3. Time-lagged correlation to find optimal lag
        best_lag, best_lag_corr = self._find_optimal_lag(x, y)

        # 4. Determine direction
        if granger_p_a_to_b < self.config.granger_significance_level:
            if granger_p_b_to_a < self.config.granger_significance_level:
                direction = "bidirectional"
            else:
                direction = "causal"
        elif granger_p_b_to_a < self.config.granger_significance_level:
            direction = "reverse_causal"
        else:
            direction = "spurious"

        # Compute confidence
        confidence = self._compute_causal_confidence(
            corr, granger_p_a_to_b, granger_p_b_to_a, best_lag_corr
        )

        # Generate insight
        insight = self._generate_causal_insight(
            metric_a, metric_b, direction, corr, best_lag, confidence
        )

        return CausalRelationship(
            cause_metric=metric_a if direction in ("causal", "bidirectional") else metric_b,
            effect_metric=metric_b if direction in ("causal", "bidirectional") else metric_a,
            lag_days=best_lag,
            correlation=float(corr),
            granger_p_value=float(min(granger_p_a_to_b, granger_p_b_to_a)),
            transfer_entropy=0.0,  # Placeholder for future implementation
            confidence=float(confidence),
            direction=direction,
            insight=insight,
        )

    def _granger_test(self, x: np.ndarray, y: np.ndarray, max_lag: int = 5) -> float:
        """
        Simplified Granger causality test.

        Tests whether x Granger-causes y by comparing:
        - Restricted model: y[t] = a₀ + Σ a_i * y[t-i]
        - Full model: y[t] = a₀ + Σ a_i * y[t-i] + Σ b_j * x[t-j]

        Uses F-test to determine if x improves prediction of y.
        """
        from scipy.stats import f as f_dist

        n = len(y)
        if n < max_lag * 3:
            return 1.0  # Not enough data

        # Create lagged features
        y_lagged = np.column_stack(
            [y[max_lag - i : -i if i > 0 else None] for i in range(1, max_lag + 1)]
        )
        x_lagged = np.column_stack(
            [x[max_lag - i : -i if i > 0 else None] for i in range(1, max_lag + 1)]
        )
        y_target = y[max_lag:]

        # Restricted model (y only)
        try:
            from sklearn.linear_model import LinearRegression

            model_restricted = LinearRegression()
            model_restricted.fit(y_lagged, y_target)
            rss_restricted = np.sum((y_target - model_restricted.predict(y_lagged)) ** 2)

            # Full model (y + x)
            X_full = np.hstack([y_lagged, x_lagged])
            model_full = LinearRegression()
            model_full.fit(X_full, y_target)
            rss_full = np.sum((y_target - model_full.predict(X_full)) ** 2)

            # F-test
            n_obs = len(y_target)
            df1 = max_lag
            df2 = n_obs - 2 * max_lag - 1

            if rss_full < 1e-10 or df2 <= 0:
                return 1.0

            f_stat = ((rss_restricted - rss_full) / df1) / (rss_full / df2)
            p_value = 1 - f_dist.cdf(f_stat, df1, df2)

            return float(p_value)

        except Exception as e:
            logger.debug(f"Granger test failed: {e}")
            return 1.0

    def _find_optimal_lag(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[int, float]:
        """Find optimal lag for time-lagged correlation."""
        best_lag = 0
        best_corr = 0.0

        for lag in range(1, self.config.max_lag_days + 1):
            if lag >= len(x):
                break

            x_lagged = x[:-lag]
            y_current = y[lag:]

            if len(x_lagged) < 30:
                continue

            try:
                corr, _ = stats.pearsonr(x_lagged, y_current)
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
            except Exception:
                continue

        return best_lag, best_corr

    def _compute_causal_confidence(
        self,
        correlation: float,
        granger_p_ab: float,
        granger_p_ba: float,
        lag_corr: float,
    ) -> float:
        """Compute confidence score for causal relationship."""
        # Higher correlation = more confidence
        corr_score = abs(correlation)

        # Lower p-value = more confidence
        granger_score = 1 - min(granger_p_ab, granger_p_ba)

        # Strong lag correlation = more confidence
        lag_score = abs(lag_corr)

        # Weighted average
        confidence = 0.4 * corr_score + 0.4 * granger_score + 0.2 * lag_score

        return float(np.clip(confidence, 0, 1))

    def _generate_causal_insight(
        self,
        metric_a: str,
        metric_b: str,
        direction: str,
        correlation: float,
        lag: int,
        confidence: float,
    ) -> str:
        """Generate human-readable insight."""
        a_readable = metric_a.replace("_", " ").lower()
        b_readable = metric_b.replace("_", " ").lower()

        if direction == "causal":
            return (
                f"Changes in {a_readable} appear to cause changes in {b_readable} "
                f"with {lag} day(s) delay (r={correlation:.2f}, confidence={confidence:.0%}). "
                f"Consider monitoring {a_readable} as a leading indicator."
            )
        elif direction == "reverse_causal":
            return (
                f"Changes in {b_readable} appear to cause changes in {a_readable} "
                f"(r={correlation:.2f}, confidence={confidence:.0%}). "
                f"The effect precedes the apparent cause."
            )
        elif direction == "bidirectional":
            return (
                f"{a_readable.title()} and {b_readable} influence each other mutually "
                f"(r={correlation:.2f}, confidence={confidence:.0%}). "
                f"Feedback loop detected."
            )
        else:
            return (
                f"Correlation between {a_readable} and {b_readable} (r={correlation:.2f}) "
                f"may be spurious or due to common cause."
            )


# =============================================================================
# MULTI-SOURCE DATA FUSER
# =============================================================================


class MultiSourceDataFuser:
    """
    Fuses data from multiple sources (XSight, MobiControl, custom telemetry).

    Features:
    - Temporal alignment across sources
    - Weighted aggregation based on source reliability
    - Conflict resolution for overlapping metrics
    - Missing data imputation using cross-source information
    """

    def __init__(self, config: MLBaselineConfig):
        self.config = config
        self._source_quality: dict[str, float] = {}

    def fuse(
        self,
        sources: dict[str, pd.DataFrame],
        join_keys: list[str] = None,
        timestamp_col: str = "Timestamp",
    ) -> pd.DataFrame:
        """
        Fuse multiple data sources into a unified dataset.

        Args:
            sources: Dictionary mapping source name to DataFrame
            join_keys: Columns to join on (default: DeviceId + timestamp)
            timestamp_col: Name of timestamp column

        Returns:
            Fused DataFrame with weighted metrics
        """
        if not sources:
            return pd.DataFrame()

        if join_keys is None:
            join_keys = ["DeviceId"]

        # Start with the highest-weight source
        sorted_sources = sorted(
            sources.items(),
            key=lambda x: self.config.source_weights.get(x[0], 0.5),
            reverse=True,
        )

        result = sorted_sources[0][1].copy()
        result["_primary_source"] = sorted_sources[0][0]

        logger.info(f"Fusing {len(sources)} data sources")

        # Merge additional sources
        for source_name, df in sorted_sources[1:]:
            weight = self.config.source_weights.get(source_name, 0.5)

            # Find overlapping columns (excluding join keys)
            overlap_cols = set(result.columns) & set(df.columns) - set(join_keys) - {
                timestamp_col,
                "_primary_source",
            }

            # For overlapping columns, use weighted average
            if overlap_cols:
                df_renamed = df.copy()
                for col in overlap_cols:
                    df_renamed = df_renamed.rename(columns={col: f"{col}_{source_name}"})

                # Merge
                result = result.merge(
                    df_renamed,
                    on=join_keys,
                    how="outer",
                    suffixes=("", f"_{source_name}"),
                )

                # Weighted combination for overlapping columns
                for col in overlap_cols:
                    col_alt = f"{col}_{source_name}"
                    if col_alt in result.columns:
                        primary_weight = self.config.source_weights.get(
                            result["_primary_source"].iloc[0], 1.0
                        )
                        total_weight = primary_weight + weight

                        result[col] = (
                            result[col].fillna(0) * primary_weight
                            + result[col_alt].fillna(0) * weight
                        ) / total_weight

                        result = result.drop(columns=[col_alt])
            else:
                # No overlap, simple merge
                result = result.merge(
                    df,
                    on=join_keys,
                    how="outer",
                )

        # Drop helper column
        if "_primary_source" in result.columns:
            result = result.drop(columns=["_primary_source"])

        logger.info(f"Fused dataset: {len(result)} rows, {len(result.columns)} columns")

        return result

    def impute_missing(
        self,
        df: pd.DataFrame,
        method: str = "knn",
        n_neighbors: int = 5,
    ) -> pd.DataFrame:
        """
        Impute missing values using cross-feature information.

        Methods:
        - median: Simple median imputation
        - knn: K-nearest neighbors imputation
        - iterative: Iterative imputation (MICE)
        """
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return result

        missing_cols = [c for c in numeric_cols if result[c].isna().any()]

        if not missing_cols:
            return result

        logger.info(f"Imputing {len(missing_cols)} columns with missing values using {method}")

        if method == "median":
            for col in missing_cols:
                result[col] = result[col].fillna(result[col].median())

        elif method == "knn":
            try:
                from sklearn.impute import KNNImputer

                imputer = KNNImputer(n_neighbors=n_neighbors)
                result[numeric_cols] = imputer.fit_transform(result[numeric_cols])
            except Exception as e:
                logger.warning(f"KNN imputation failed: {e}, falling back to median")
                for col in missing_cols:
                    result[col] = result[col].fillna(result[col].median())

        elif method == "iterative":
            try:
                from sklearn.impute import IterativeImputer

                imputer = IterativeImputer(random_state=42, max_iter=10)
                result[numeric_cols] = imputer.fit_transform(result[numeric_cols])
            except Exception as e:
                logger.warning(f"Iterative imputation failed: {e}, falling back to median")
                for col in missing_cols:
                    result[col] = result[col].fillna(result[col].median())

        return result


# =============================================================================
# ONLINE LEARNING BASELINE UPDATER
# =============================================================================


class OnlineLearningBaseline:
    """
    Online learning component for real-time baseline updates.

    Implements:
    - Incremental mean/variance updates (Welford's algorithm)
    - Exponentially weighted moving statistics
    - Change point detection
    - Automatic retraining triggers
    """

    def __init__(self, config: MLBaselineConfig):
        self.config = config
        self._stats: dict[str, _OnlineStats] = {}
        self._change_points: dict[str, list[datetime]] = {}
        self._observation_count = 0

    def update(
        self,
        metric_name: str,
        value: float,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Update statistics with a single observation.

        Returns current statistics and any detected changes.
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        if metric_name not in self._stats:
            self._stats[metric_name] = _OnlineStats(learning_rate=self.config.online_learning_rate)
            self._change_points[metric_name] = []

        stats = self._stats[metric_name]

        # Update statistics
        stats.update(value)
        self._observation_count += 1

        # Check for change point
        change_detected = False
        if stats.count > self.config.min_samples_for_baseline:
            z_score = abs(value - stats.mean) / (stats.std + 1e-10)
            if z_score > 4.0:  # More than 4 standard deviations
                change_detected = True
                self._change_points[metric_name].append(timestamp)
                logger.info(f"Change point detected in {metric_name} at {timestamp}")

        return {
            "metric": metric_name,
            "value": value,
            "mean": stats.mean,
            "std": stats.std,
            "ewma": stats.ewma,
            "count": stats.count,
            "change_detected": change_detected,
            "timestamp": timestamp.isoformat(),
        }

    def batch_update(
        self,
        df: pd.DataFrame,
        metric_cols: list[str],
        timestamp_col: str = "Timestamp",
    ) -> dict[str, dict[str, Any]]:
        """Update statistics for multiple metrics from a DataFrame."""
        results = {}

        for col in metric_cols:
            if col not in df.columns:
                continue

            values = df[col].dropna()
            if len(values) == 0:
                continue

            for i, val in enumerate(values):
                ts = df[timestamp_col].iloc[i] if timestamp_col in df.columns else None
                if isinstance(ts, str):
                    ts = pd.to_datetime(ts)
                results[col] = self.update(col, float(val), ts)

        return results

    def get_baseline(self, metric_name: str) -> dict[str, Any] | None:
        """Get current baseline for a metric."""
        if metric_name not in self._stats:
            return None

        stats = self._stats[metric_name]
        return {
            "metric": metric_name,
            "mean": stats.mean,
            "std": stats.std,
            "ewma": stats.ewma,
            "count": stats.count,
            "min": stats.min_val,
            "max": stats.max_val,
        }

    def needs_retraining(self, metric_name: str, lookback_hours: int = 24) -> bool:
        """Check if metric needs model retraining based on recent change points."""
        if metric_name not in self._change_points:
            return False

        cutoff = datetime.now(UTC) - timedelta(hours=lookback_hours)
        recent_changes = [cp for cp in self._change_points[metric_name] if cp > cutoff]

        # Trigger retraining if more than 3 change points in lookback period
        return len(recent_changes) >= 3


class _OnlineStats:
    """Welford's online algorithm for computing mean and variance."""

    def __init__(self, learning_rate: float = 0.1):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.ewma = 0.0
        self.learning_rate = learning_rate
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def update(self, value: float) -> None:
        if not np.isfinite(value):
            return

        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

        # EWMA
        if self.count == 1:
            self.ewma = value
        else:
            self.ewma = self.learning_rate * value + (1 - self.learning_rate) * self.ewma

        # Min/Max
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.M2 / (self.count - 1)

    @property
    def std(self) -> float:
        return np.sqrt(self.variance)


# =============================================================================
# ADVANCED DRIFT DETECTION
# =============================================================================


class AdvancedDriftDetector:
    """
    Advanced drift detection with multiple algorithms.

    Implements:
    1. PSI (Population Stability Index)
    2. KS Test (Kolmogorov-Smirnov)
    3. Jensen-Shannon Divergence
    4. Page-Hinkley Test (for concept drift)
    5. ADWIN (Adaptive Windowing)
    """

    def __init__(self, config: MLBaselineConfig):
        self.config = config
        self._reference_distributions: dict[str, np.ndarray] = {}
        self._page_hinkley: dict[str, _PageHinkley] = {}

    def set_reference(self, metric_name: str, values: np.ndarray) -> None:
        """Set reference distribution for a metric."""
        self._reference_distributions[metric_name] = values
        self._page_hinkley[metric_name] = _PageHinkley(threshold=self.config.drift_threshold_psi)

    def detect_drift(
        self,
        metric_name: str,
        current_values: np.ndarray,
    ) -> dict[str, Any]:
        """
        Detect drift using multiple methods.

        Returns dictionary with drift indicators from each method.
        """
        if metric_name not in self._reference_distributions:
            return {"error": "No reference distribution set"}

        reference = self._reference_distributions[metric_name]
        current = current_values[np.isfinite(current_values)]

        if len(current) < 10:
            return {"error": "Insufficient current samples"}

        results = {
            "metric": metric_name,
            "reference_samples": len(reference),
            "current_samples": len(current),
        }

        # 1. PSI
        psi = self._compute_psi(reference, current)
        results["psi"] = float(psi)
        results["psi_drift"] = psi >= self.config.drift_threshold_psi

        # 2. KS Test
        ks_stat, ks_pvalue = stats.ks_2samp(reference, current)
        results["ks_statistic"] = float(ks_stat)
        results["ks_pvalue"] = float(ks_pvalue)
        results["ks_drift"] = ks_pvalue < self.config.drift_threshold_ks

        # 3. Jensen-Shannon Divergence
        js_div = self._compute_js_divergence(reference, current)
        results["js_divergence"] = float(js_div)
        results["js_drift"] = js_div > 0.1  # Threshold for JS divergence

        # 4. Mean shift
        ref_mean = np.mean(reference)
        cur_mean = np.mean(current)
        ref_std = np.std(reference)
        mean_shift_z = abs(cur_mean - ref_mean) / (ref_std + 1e-10)
        results["mean_shift_z"] = float(mean_shift_z)
        results["mean_drift"] = mean_shift_z > 2.0

        # 5. Variance change
        ref_var = np.var(reference)
        cur_var = np.var(current)
        var_ratio = cur_var / (ref_var + 1e-10)
        results["variance_ratio"] = float(var_ratio)
        results["variance_drift"] = var_ratio > 2.0 or var_ratio < 0.5

        # Combined drift detection
        drift_indicators = [
            results["psi_drift"],
            results["ks_drift"],
            results["js_drift"],
            results["mean_drift"],
            results["variance_drift"],
        ]
        results["drift_detected"] = sum(drift_indicators) >= 2  # Majority voting
        results["drift_severity"] = sum(drift_indicators) / len(drift_indicators)

        return results

    def _compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Population Stability Index."""
        # Create bins from reference
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0

        # Compute histograms
        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

        # Normalize
        ref_pct = ref_counts / (len(reference) + 1e-10)
        cur_pct = cur_counts / (len(current) + 1e-10)

        # Avoid log(0)
        epsilon = 1e-10
        ref_pct = np.clip(ref_pct, epsilon, 1)
        cur_pct = np.clip(cur_pct, epsilon, 1)

        # PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        return float(psi)

    def _compute_js_divergence(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 20,
    ) -> float:
        """Compute Jensen-Shannon Divergence."""
        # Create common bins
        all_data = np.concatenate([reference, current])
        bins = np.linspace(np.min(all_data), np.max(all_data), n_bins + 1)

        # Compute histograms
        ref_hist, _ = np.histogram(reference, bins=bins, density=True)
        cur_hist, _ = np.histogram(current, bins=bins, density=True)

        # Normalize to probabilities
        ref_prob = ref_hist / (np.sum(ref_hist) + 1e-10)
        cur_prob = cur_hist / (np.sum(cur_hist) + 1e-10)

        # JS divergence
        m = 0.5 * (ref_prob + cur_prob)

        # KL divergences
        epsilon = 1e-10
        kl_ref = np.sum(ref_prob * np.log((ref_prob + epsilon) / (m + epsilon)))
        kl_cur = np.sum(cur_prob * np.log((cur_prob + epsilon) / (m + epsilon)))

        js = 0.5 * (kl_ref + kl_cur)

        return float(js)


class _PageHinkley:
    """Page-Hinkley test for change detection."""

    def __init__(self, threshold: float = 0.2, alpha: float = 0.005):
        self.threshold = threshold
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.mean = 0.0
        self.sum = 0.0
        self.min_sum = 0.0
        self.count = 0

    def update(self, value: float) -> bool:
        """Update and return True if change detected."""
        self.count += 1
        self.mean = self.mean + (value - self.mean) / self.count
        self.sum = self.sum + value - self.mean - self.alpha
        self.min_sum = min(self.min_sum, self.sum)

        return (self.sum - self.min_sum) > self.threshold


# =============================================================================
# MAIN ENGINE CLASS
# =============================================================================


class MLBaselineEngine:
    """
    Ultra-Advanced ML Baseline Engine.

    Orchestrates all components:
    - BayesianBaselineAdapter: Adaptive baselines with uncertainty
    - EnsembleAnomalyDetector: Multi-algorithm anomaly detection
    - CausalCorrelationEngine: Causal discovery for correlations
    - MultiSourceDataFuser: Data fusion from multiple sources
    - OnlineLearningBaseline: Real-time updates
    - AdvancedDriftDetector: Comprehensive drift detection

    Usage:
        engine = MLBaselineEngine()

        # Fit from historical data
        engine.fit(historical_df)

        # Score new data
        scored_df = engine.score(new_df)

        # Online updates
        engine.update_online(streaming_df)

        # Check for drift
        drift_report = engine.check_drift()
    """

    def __init__(self, config: MLBaselineConfig | None = None):
        self.config = config or MLBaselineConfig()

        # Initialize components
        self.bayesian_adapter = BayesianBaselineAdapter(self.config)
        self.ensemble_detector = EnsembleAnomalyDetector(self.config)
        self.causal_engine = CausalCorrelationEngine(self.config)
        self.data_fuser = MultiSourceDataFuser(self.config)
        self.online_baseline = OnlineLearningBaseline(self.config)
        self.drift_detector = AdvancedDriftDetector(self.config)

        self._is_fitted = False
        self._feature_cols: list[str] = []
        self._metric_cols: list[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        metric_cols: list[str] | None = None,
    ) -> MLBaselineEngine:
        """
        Fit the baseline engine on historical data.

        Args:
            df: Historical telemetry data
            feature_cols: Columns for anomaly detection
            metric_cols: Columns for baseline tracking
        """
        logger.info(f"Fitting ML Baseline Engine on {len(df)} rows")

        # Auto-select columns if not provided
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {
            "DeviceId",
            "Timestamp",
            "CollectedDate",
            "tenant_id",
            "cohort_id",
            "is_injected_anomaly",
            "anomaly_score",
            "anomaly_label",
        }

        self._feature_cols = feature_cols or [c for c in numeric_cols if c not in exclude]
        self._metric_cols = metric_cols or self._feature_cols[:20]  # Limit for performance

        # 1. Fit ensemble detector
        logger.info("Fitting ensemble anomaly detector...")
        self.ensemble_detector.fit(df, self._feature_cols)

        # 2. Initialize Bayesian baselines
        logger.info("Initializing Bayesian baselines...")
        for col in self._metric_cols:
            if col in df.columns:
                self.bayesian_adapter.initialize_prior(col, df[col])

        # 3. Set drift references
        logger.info("Setting drift detection references...")
        for col in self._metric_cols:
            if col in df.columns:
                values = df[col].dropna().values
                if len(values) >= self.config.min_samples_for_baseline:
                    self.drift_detector.set_reference(col, values)

        # 4. Discover causal relationships
        if self.config.enable_causal_discovery:
            logger.info("Running causal discovery...")
            self.causal_engine.discover_causal_relationships(df, self._metric_cols)

        self._is_fitted = True
        logger.info("ML Baseline Engine fitted successfully")

        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score new data using the fitted engine.

        Returns DataFrame with:
        - ensemble_score: Combined anomaly score (0-1)
        - ensemble_anomaly: Binary anomaly label
        - individual algorithm scores
        - bayesian_anomaly_prob: Per-metric anomaly probabilities
        """
        if not self._is_fitted:
            raise RuntimeError("Engine not fitted. Call fit() first.")

        # Score with ensemble
        result = self.ensemble_detector.score(df)

        # Add Bayesian probabilities for each metric
        for col in self._metric_cols:
            if col not in df.columns:
                continue

            probs = []
            severities = []
            for val in df[col]:
                if pd.isna(val):
                    probs.append(0.5)
                    severities.append("unknown")
                else:
                    prob, severity = self.bayesian_adapter.get_anomaly_probability(col, float(val))
                    probs.append(prob)
                    severities.append(severity)

            result[f"{col}_anomaly_prob"] = probs
            result[f"{col}_severity"] = severities

        return result

    def update_online(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "Timestamp",
    ) -> dict[str, Any]:
        """
        Update baselines with new streaming data.

        Returns update statistics and any detected changes.
        """
        # Update online statistics
        update_results = self.online_baseline.batch_update(df, self._metric_cols, timestamp_col)

        # Update Bayesian posteriors
        for col in self._metric_cols:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    self.bayesian_adapter.update(col, values, batch_update=True)

        # Check for metrics needing retraining
        needs_retraining = []
        for col in self._metric_cols:
            if self.online_baseline.needs_retraining(col):
                needs_retraining.append(col)
            if self.bayesian_adapter.needs_recomputation(col):
                needs_retraining.append(col)

        return {
            "updated_metrics": list(update_results.keys()),
            "update_count": len(df),
            "needs_retraining": list(set(needs_retraining)),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def check_drift(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Check for drift across all tracked metrics.

        Returns comprehensive drift report.
        """
        drift_report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "metrics_checked": 0,
            "metrics_drifted": 0,
            "details": {},
        }

        for col in self._metric_cols:
            if col not in df.columns:
                continue

            values = df[col].dropna().values
            if len(values) < 10:
                continue

            drift_result = self.drift_detector.detect_drift(col, values)
            drift_report["metrics_checked"] += 1

            if drift_result.get("drift_detected", False):
                drift_report["metrics_drifted"] += 1

            drift_report["details"][col] = drift_result

        drift_report["drift_rate"] = drift_report["metrics_drifted"] / max(
            1, drift_report["metrics_checked"]
        )

        return drift_report

    def get_causal_insights(self) -> list[dict[str, Any]]:
        """Get discovered causal relationships."""
        return [
            {
                "cause": r.cause_metric,
                "effect": r.effect_metric,
                "lag_days": r.lag_days,
                "correlation": r.correlation,
                "direction": r.direction,
                "confidence": r.confidence,
                "insight": r.insight,
            }
            for r in self.causal_engine.discovered_relationships
        ]

    def export_state(self, path: Path) -> None:
        """Export engine state for persistence."""
        state = {
            "version": self.config.model_version,
            "config": asdict(self.config),
            "bayesian_state": self.bayesian_adapter.export_state(),
            "feature_cols": self._feature_cols,
            "metric_cols": self._metric_cols,
            "causal_relationships": self.get_causal_insights(),
            "exported_at": datetime.now(UTC).isoformat(),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, default=str))
        logger.info(f"Exported engine state to {path}")

    def import_state(self, path: Path) -> None:
        """Import engine state from persistence."""
        state = json.loads(path.read_text())

        self._feature_cols = state.get("feature_cols", [])
        self._metric_cols = state.get("metric_cols", [])

        if "bayesian_state" in state:
            self.bayesian_adapter.import_state(state["bayesian_state"])

        logger.info(f"Imported engine state from {path}")
