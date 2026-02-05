"""
SHAP-Based Explainability for Anomaly Detection.

This module provides feature-level explanations for why devices were
flagged as anomalies using SHAP (SHapley Additive exPlanations) values.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExplanationConfig:
    """Configuration for anomaly explanations."""

    background_samples: int = 100
    top_features: int = 5
    use_tree_explainer: bool = True  # Use TreeExplainer for tree-based models
    enable_caching: bool = True


@dataclass
class AnomalyExplanation:
    """Explanation for a single anomaly."""

    device_id: Any
    anomaly_score: float
    is_anomaly: bool
    top_contributors: list[dict[str, Any]]
    shap_sum: float
    confidence: float


class AnomalyExplainer:
    """
    SHAP-based explainability for anomaly detection.

    Provides feature-level explanations for why each device was flagged.
    Uses TreeExplainer for Isolation Forest (fast) or KernelExplainer
    for other models.

    Usage:
        explainer = AnomalyExplainer(model, feature_names)
        explainer.fit(X_background)
        explanations = explainer.explain(X_new)
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        config: ExplanationConfig | None = None,
    ):
        """
        Initialize the explainer.

        Args:
            model: Trained anomaly detection model (IsolationForest or similar)
            feature_names: List of feature column names
            config: Explanation configuration
        """
        self.model = model
        self.feature_names = feature_names
        self.config = config or ExplanationConfig()
        self._explainer: Any = None
        self._is_fitted = False

    def fit(self, X_background: np.ndarray) -> None:
        """
        Fit the SHAP explainer with background data.

        Args:
            X_background: Background dataset for SHAP calculations
        """
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not installed. Install with: pip install shap")
            return

        # Sample background data if needed
        if len(X_background) > self.config.background_samples:
            idx = np.random.choice(
                len(X_background),
                self.config.background_samples,
                replace=False
            )
            X_background = X_background[idx]

        logger.info(f"Fitting SHAP explainer with {len(X_background)} background samples")

        try:
            # Try TreeExplainer first (faster for tree-based models)
            if self.config.use_tree_explainer and hasattr(self.model, "estimators_"):
                self._explainer = shap.TreeExplainer(
                    self.model,
                    X_background,
                    feature_perturbation="interventional",
                )
                logger.info("Using TreeExplainer for Isolation Forest")
            else:
                # Fall back to KernelExplainer
                self._explainer = shap.KernelExplainer(
                    self._decision_function,
                    X_background,
                )
                logger.info("Using KernelExplainer")

            self._is_fitted = True

        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            self._is_fitted = False

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """Wrapper for model's decision function."""
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(X)
        elif hasattr(self.model, "score_samples"):
            return self.model.score_samples(X)
        else:
            raise ValueError("Model must have decision_function or score_samples method")

    def explain(
        self,
        X: np.ndarray,
        scores: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        device_ids: list | None = None,
    ) -> list[AnomalyExplanation]:
        """
        Generate explanations for samples.

        Args:
            X: Feature matrix to explain
            scores: Pre-computed anomaly scores (optional)
            labels: Pre-computed labels (optional, -1=anomaly, 1=normal)
            device_ids: Device identifiers (optional)

        Returns:
            List of AnomalyExplanation objects
        """
        if not self._is_fitted or self._explainer is None:
            logger.warning("Explainer not fitted, returning empty explanations")
            return self._fallback_explanations(X, scores, labels, device_ids)

        try:
            import shap
        except ImportError:
            return self._fallback_explanations(X, scores, labels, device_ids)

        logger.info(f"Computing SHAP explanations for {len(X)} samples")

        try:
            shap_values = self._explainer.shap_values(X)
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            return self._fallback_explanations(X, scores, labels, device_ids)

        # Compute scores if not provided
        if scores is None:
            scores = self._decision_function(X)

        if labels is None:
            threshold = np.percentile(scores, 5)
            labels = np.where(scores <= threshold, -1, 1)

        if device_ids is None:
            device_ids = list(range(len(X)))

        # Build explanations
        explanations = []
        for i in range(len(X)):
            explanation = self._build_explanation(
                device_id=device_ids[i],
                shap_values=shap_values[i] if isinstance(shap_values, np.ndarray) else shap_values,
                score=scores[i],
                is_anomaly=(labels[i] == -1),
            )
            explanations.append(explanation)

        return explanations

    def _build_explanation(
        self,
        device_id: Any,
        shap_values: np.ndarray,
        score: float,
        is_anomaly: bool,
    ) -> AnomalyExplanation:
        """Build explanation object from SHAP values."""
        # Get feature contributions
        contributions = list(zip(self.feature_names, shap_values, strict=False))

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        # Get top contributors
        top_k = min(self.config.top_features, len(contributions))
        top_contributors = [
            {
                "feature": name,
                "contribution": float(value),
                "direction": "increases_anomaly" if value < 0 else "decreases_anomaly",
            }
            for name, value in contributions[:top_k]
        ]

        # Calculate confidence from score distribution
        # Further from threshold = higher confidence
        confidence = min(1.0, abs(score) / (abs(score) + 0.1))

        return AnomalyExplanation(
            device_id=device_id,
            anomaly_score=float(score),
            is_anomaly=is_anomaly,
            top_contributors=top_contributors,
            shap_sum=float(sum(shap_values)),
            confidence=confidence,
        )

    def _fallback_explanations(
        self,
        X: np.ndarray,
        scores: np.ndarray | None,
        labels: np.ndarray | None,
        device_ids: list | None,
    ) -> list[AnomalyExplanation]:
        """Generate fallback explanations without SHAP."""
        if scores is None:
            scores = self._decision_function(X)

        if labels is None:
            threshold = np.percentile(scores, 5)
            labels = np.where(scores <= threshold, -1, 1)

        if device_ids is None:
            device_ids = list(range(len(X)))

        explanations = []
        for i in range(len(X)):
            # Use feature values as proxy for contribution
            feature_values = X[i]
            contributions = list(zip(self.feature_names, feature_values, strict=False))

            # Sort by absolute value (extreme values more likely contributors)
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)

            top_k = min(self.config.top_features, len(contributions))
            top_contributors = [
                {
                    "feature": name,
                    "value": float(value),
                    "note": "SHAP unavailable, showing extreme values",
                }
                for name, value in contributions[:top_k]
            ]

            explanations.append(AnomalyExplanation(
                device_id=device_ids[i],
                anomaly_score=float(scores[i]),
                is_anomaly=(labels[i] == -1),
                top_contributors=top_contributors,
                shap_sum=0.0,
                confidence=0.5,
            ))

        return explanations

    def explain_dataframe(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        score_col: str = "anomaly_score",
        label_col: str = "anomaly_label",
        device_id_col: str = "DeviceId",
    ) -> pd.DataFrame:
        """
        Explain anomalies in a scored dataframe.

        Args:
            df: Scored dataframe with features and anomaly labels
            feature_cols: List of feature columns used for detection
            score_col: Column containing anomaly scores
            label_col: Column containing anomaly labels
            device_id_col: Column containing device identifiers

        Returns:
            DataFrame with explanation columns added
        """
        X = df[feature_cols].values
        scores = df[score_col].values if score_col in df.columns else None
        labels = df[label_col].values if label_col in df.columns else None
        device_ids = df[device_id_col].tolist() if device_id_col in df.columns else None

        explanations = self.explain(X, scores, labels, device_ids)

        # Add explanation columns to dataframe
        df_out = df.copy()
        df_out["explanation_top_features"] = [
            ", ".join(e["feature"] for e in exp.top_contributors)
            for exp in explanations
        ]
        df_out["explanation_confidence"] = [exp.confidence for exp in explanations]
        df_out["shap_sum"] = [exp.shap_sum for exp in explanations]

        # Add detailed explanations as JSON
        df_out["explanation_details"] = [
            exp.top_contributors for exp in explanations
        ]

        return df_out


def create_explainer(
    model: Any,
    feature_names: list[str],
    background_samples: int = 100,
) -> AnomalyExplainer:
    """
    Create an anomaly explainer.

    Args:
        model: Trained anomaly detection model
        feature_names: List of feature column names
        background_samples: Number of background samples for SHAP

    Returns:
        Configured AnomalyExplainer
    """
    config = ExplanationConfig(background_samples=background_samples)
    return AnomalyExplainer(model, feature_names, config)


def explain_anomalies(
    model: Any,
    df: pd.DataFrame,
    feature_cols: list[str],
    X_background: np.ndarray | None = None,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Convenience function to explain anomalies in a dataframe.

    Args:
        model: Trained anomaly detection model
        df: Dataframe with features and anomaly scores
        feature_cols: Feature columns used for detection
        X_background: Background data for SHAP (uses training data if None)
        top_k: Number of top contributing features to show

    Returns:
        DataFrame with explanation columns added
    """
    config = ExplanationConfig(top_features=top_k)
    explainer = AnomalyExplainer(model, feature_cols, config)

    if X_background is None:
        X_background = df[feature_cols].values

    explainer.fit(X_background)
    return explainer.explain_dataframe(df, feature_cols)
