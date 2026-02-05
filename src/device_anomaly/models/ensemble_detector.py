"""
Ensemble Anomaly Detector combining multiple algorithms.

This module provides a multi-algorithm ensemble for robust anomaly detection:
- Isolation Forest (tree-based, fast)
- Local Outlier Factor (density-based)
- One-Class SVM (boundary-based)

The ensemble combines scores using weighted averaging for improved
detection accuracy and reduced false positives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble anomaly detector."""

    contamination: float = 0.05
    random_state: int = 42
    min_variance: float = 1e-6
    scale_features: bool = True

    # Algorithm-specific parameters
    if_n_estimators: int = 300
    lof_n_neighbors: int = 20
    ocsvm_kernel: str = "rbf"
    ocsvm_nu: float = 0.05

    # Ensemble weights
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "isolation_forest": 0.5,
            "lof": 0.3,
            "ocsvm": 0.2,
        }
    )

    # Enabled algorithms
    enable_isolation_forest: bool = True
    enable_lof: bool = True
    enable_ocsvm: bool = True

    # Adaptive thresholding
    use_adaptive_threshold: bool = True
    adaptive_percentile: float = 5.0


class EnsembleAnomalyDetector:
    """
    Multi-algorithm ensemble for robust anomaly detection.

    Combines:
    1. Isolation Forest - Tree-based isolation, fast training and inference
    2. Local Outlier Factor - Density-based, good for clustered normal data
    3. One-Class SVM - Boundary-based, captures complex decision boundaries

    Usage:
        detector = EnsembleAnomalyDetector()
        detector.fit(df_features)
        df_scored = detector.score_dataframe(df_features)
    """

    def __init__(
        self,
        config: EnsembleConfig | None = None,
        feature_overrides: list[str] | None = None,
    ):
        self.config = config or EnsembleConfig()
        self.feature_overrides = feature_overrides

        self.models: dict[str, Any] = {}
        self.feature_cols: list[str] = []
        self.impute_values: pd.Series | None = None
        self.scaler: StandardScaler | None = None
        self.feature_weights: dict[str, float] = {}

        self._init_models()

    def _init_models(self) -> None:
        """Initialize ensemble models based on config."""
        if self.config.enable_isolation_forest:
            self.models["isolation_forest"] = IsolationForest(
                n_estimators=self.config.if_n_estimators,
                contamination=self.config.contamination,
                random_state=self.config.random_state,
                n_jobs=-1,
            )

        if self.config.enable_lof:
            self.models["lof"] = LocalOutlierFactor(
                n_neighbors=self.config.lof_n_neighbors,
                contamination=self.config.contamination,
                novelty=True,  # Enable predict on new data
                n_jobs=-1,
            )

        if self.config.enable_ocsvm:
            self.models["ocsvm"] = OneClassSVM(
                kernel=self.config.ocsvm_kernel,
                nu=self.config.ocsvm_nu,
            )

        if not self.models:
            raise ValueError("At least one algorithm must be enabled")

    def _select_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Select numeric feature columns for training."""
        if self.feature_overrides:
            overrides = [c for c in self.feature_overrides if c in df.columns]
            if not overrides:
                raise ValueError(
                    f"None of the specified feature overrides exist: {self.feature_overrides}"
                )
            return [c for c in overrides if np.issubdtype(df[c].dtype, np.number)]

        import pandas.api.types as ptypes

        numeric_cols = [c for c in df.columns if ptypes.is_numeric_dtype(df[c])]

        exclude = {
            "DeviceId",
            "ModelId",
            "ManufacturerId",
            "OsVersionId",
            "is_injected_anomaly",
            "anomaly_score",
            "anomaly_label",
            "ensemble_score",
            "ensemble_label",
        }

        candidates = [c for c in numeric_cols if c not in exclude]

        # Prefer cohort-normalized features
        cohort_cols = [c for c in candidates if c.endswith("_cohort_z")]
        baseline_cols = [c for c in candidates if "_z_" in c]

        feature_cols = cohort_cols or baseline_cols or candidates

        if not feature_cols:
            raise ValueError("No feature columns found to train on.")

        return feature_cols

    def _prepare_training_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix for training."""
        feature_df = df[self.feature_cols].copy()
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        medians = feature_df.median().fillna(0.0)
        self.impute_values = medians
        feature_df = feature_df.fillna(self.impute_values)

        # Apply feature weighting
        self.feature_weights = self._resolve_feature_weights(feature_df.columns.tolist())
        feature_df = self._apply_feature_weights(feature_df)

        # Remove near-constant features
        variances = feature_df.var(ddof=0).fillna(0.0)
        keep_mask = variances > self.config.min_variance
        kept_cols = list(variances[keep_mask].index)

        if not kept_cols:
            raise ValueError("All feature columns have near-zero variance.")

        dropped = sorted(set(self.feature_cols) - set(kept_cols))
        if dropped:
            logger.info(f"Dropping {len(dropped)} near-constant features")

        self.feature_cols = kept_cols
        self.impute_values = self.impute_values[self.feature_cols]
        feature_df = feature_df[self.feature_cols]

        if self.config.scale_features:
            self.scaler = StandardScaler()
            matrix = self.scaler.fit_transform(feature_df.values)
        else:
            self.scaler = None
            matrix = feature_df.values

        return matrix

    def _prepare_inference_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix for inference."""
        if not self.feature_cols:
            raise RuntimeError("Model has not been fit yet.")

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        feature_df = df[self.feature_cols].copy()
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        if self.impute_values is None:
            raise RuntimeError("Imputation values missing")

        feature_df = feature_df.fillna(self.impute_values)
        feature_df = self._apply_feature_weights(feature_df)

        if self.config.scale_features and self.scaler is not None:
            return self.scaler.transform(feature_df.values)

        return feature_df.values

    def _resolve_feature_weights(self, columns: list[str]) -> dict[str, float]:
        """Resolve feature weights by domain."""
        try:
            from device_anomaly.config.feature_config import FeatureConfig

            domains = FeatureConfig.feature_domains
            weights = FeatureConfig.domain_weights
        except ImportError:
            return dict.fromkeys(columns, 1.0)

        resolved: dict[str, float] = {}
        for col in columns:
            root = self._root_feature(col)
            domain = domains.get(root)
            resolved[col] = weights.get(domain, 1.0) if domain else 1.0

        return resolved

    @staticmethod
    def _root_feature(col: str) -> str:
        """Extract root feature name from derived feature."""
        suffixes = ["_roll_mean", "_roll_std", "_delta", "_cohort_z", "_trend", "_residual"]
        for suf in suffixes:
            if col.endswith(suf):
                return col[: -len(suf)]
        if "_z_" in col:
            return col.split("_z_")[0]
        return col

    def _apply_feature_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature weights to dataframe."""
        if not self.feature_weights:
            return df
        weight_series = pd.Series(self.feature_weights)
        aligned = weight_series.reindex(df.columns).fillna(1.0)
        return df.mul(aligned, axis=1)

    def fit(self, df: pd.DataFrame) -> None:
        """Fit all ensemble models on the training data."""
        self.feature_cols = self._select_feature_columns(df)
        logger.info(f"Training ensemble on {len(self.feature_cols)} features")

        X = self._prepare_training_matrix(df)

        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            try:
                model.fit(X)
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                raise

        logger.info("Ensemble training complete")

    def score(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Score data using all ensemble models.

        Returns dict of model_name -> scores array.
        Note: Lower scores = more anomalous for all algorithms.
        """
        X = self._prepare_inference_matrix(df)
        scores = {}

        for name, model in self.models.items():
            try:
                if name == "isolation_forest":
                    scores[name] = model.decision_function(X)
                elif name == "lof":
                    # LOF returns negative scores for outliers
                    scores[name] = model.decision_function(X)
                elif name == "ocsvm":
                    scores[name] = model.decision_function(X)
            except Exception as e:
                logger.warning(f"Scoring failed for {name}: {e}")
                scores[name] = np.zeros(len(X))

        return scores

    def ensemble_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute weighted ensemble score.

        Returns normalized ensemble score where lower = more anomalous.
        """
        raw_scores = self.score(df)

        # Normalize each algorithm's scores to [0, 1] range
        normalized_scores = {}
        for name, scores in raw_scores.items():
            min_s, max_s = scores.min(), scores.max()
            if max_s - min_s > 1e-8:
                normalized_scores[name] = (scores - min_s) / (max_s - min_s)
            else:
                normalized_scores[name] = np.ones_like(scores) * 0.5

        # Compute weighted average
        weights = self.config.weights
        total_weight = sum(weights.get(name, 0) for name in normalized_scores)

        if total_weight < 1e-8:
            # Equal weights if no weights specified
            return np.mean(list(normalized_scores.values()), axis=0)

        ensemble = np.zeros(len(df))
        for name, scores in normalized_scores.items():
            w = weights.get(name, 0)
            ensemble += scores * (w / total_weight)

        return ensemble

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly labels using ensemble.

        Returns: 1 for normal, -1 for anomaly
        """
        scores = self.ensemble_score(df)
        threshold = np.percentile(scores, self.config.adaptive_percentile)
        return np.where(scores <= threshold, -1, 1)

    def score_dataframe(
        self,
        df: pd.DataFrame,
        include_individual: bool = False,
    ) -> pd.DataFrame:
        """
        Score dataframe and return with ensemble scores and labels.

        Args:
            df: Input dataframe with features
            include_individual: Include individual model scores

        Returns:
            DataFrame with ensemble_score and ensemble_label columns.
        """
        df_scored = df.copy()

        # Get individual scores
        raw_scores = self.score(df)

        # Compute ensemble score
        ensemble = self.ensemble_score(df)
        df_scored["ensemble_score"] = ensemble

        # Determine labels
        threshold = np.percentile(ensemble, self.config.adaptive_percentile)
        labels = np.where(ensemble <= threshold, -1, 1)
        df_scored["ensemble_label"] = labels

        # Add individual model scores if requested
        if include_individual:
            for name, scores in raw_scores.items():
                df_scored[f"{name}_score"] = scores

        anomaly_count = int((labels == -1).sum())
        logger.info(f"Ensemble detection: {anomaly_count}/{len(df)} anomalies")

        return df_scored

    def get_model_agreement(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate agreement between ensemble models.

        Returns dataframe with agreement metrics per sample.
        """
        raw_scores = self.score(df)
        n_models = len(raw_scores)

        # Get individual predictions
        predictions = {}
        for name, scores in raw_scores.items():
            threshold = np.percentile(scores, self.config.adaptive_percentile)
            predictions[name] = np.where(scores <= threshold, -1, 1)

        # Count agreements
        pred_matrix = np.array(list(predictions.values()))
        anomaly_votes = (pred_matrix == -1).sum(axis=0)
        normal_votes = n_models - anomaly_votes

        agreement = pd.DataFrame(
            {
                "anomaly_votes": anomaly_votes,
                "normal_votes": normal_votes,
                "agreement_ratio": np.maximum(anomaly_votes, normal_votes) / n_models,
                "unanimous_anomaly": anomaly_votes == n_models,
                "unanimous_normal": normal_votes == n_models,
            }
        )

        return agreement

    def save_model(self, output_path: str | Path) -> dict[str, Path]:
        """Save ensemble model to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pkl_path = output_path.with_suffix(".pkl")

        state = {
            "models": self.models,
            "feature_cols": self.feature_cols,
            "impute_values": self.impute_values.to_dict()
            if self.impute_values is not None
            else None,
            "scaler": self.scaler,
            "config": self.config,
            "feature_weights": self.feature_weights,
        }

        joblib.dump(state, pkl_path)
        logger.info(f"Saved ensemble model to {pkl_path}")

        return {"sklearn": pkl_path}

    @classmethod
    def load_model(cls, model_path: str | Path) -> EnsembleAnomalyDetector:
        """Load ensemble model from disk."""
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        loaded = joblib.load(model_path)

        instance = cls.__new__(cls)
        instance.models = loaded["models"]
        instance.feature_cols = loaded.get("feature_cols", [])

        if loaded.get("impute_values"):
            instance.impute_values = pd.Series(loaded["impute_values"])
        else:
            instance.impute_values = None

        instance.scaler = loaded.get("scaler")
        instance.config = loaded.get("config", EnsembleConfig())
        instance.feature_weights = loaded.get("feature_weights", {})
        instance.feature_overrides = None

        logger.info(f"Loaded ensemble model from {model_path}")
        return instance


def create_ensemble_detector(
    contamination: float = 0.05,
    weights: dict[str, float] | None = None,
) -> EnsembleAnomalyDetector:
    """
    Create an ensemble detector with default configuration.

    Args:
        contamination: Expected proportion of anomalies
        weights: Algorithm weights (default: IF=0.5, LOF=0.3, OCSVM=0.2)

    Returns:
        Configured EnsembleAnomalyDetector
    """
    config = EnsembleConfig(
        contamination=contamination,
        weights=weights or {"isolation_forest": 0.5, "lof": 0.3, "ocsvm": 0.2},
    )
    return EnsembleAnomalyDetector(config=config)
