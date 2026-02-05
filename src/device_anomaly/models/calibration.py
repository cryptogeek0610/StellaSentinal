from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from device_anomaly.config.feature_config import FeatureConfig


@dataclass
class IsoScoreCalibratorConfig:
    model_path: Path = Path("artifacts/iso_calibrator.pkl")
    min_rows: int = 200
    feature_limit: int = 10


class IsoScoreCalibrator:
    """
    Lightweight supervised calibrator that learns to map anomaly scores + top metrics
    to probabilities using synthetic ground truth (is_injected_anomaly).
    """

    def __init__(self, config: IsoScoreCalibratorConfig | None = None):
        self.config = config or IsoScoreCalibratorConfig()
        self.model: GradientBoostingClassifier | None = None
        self.features: list[str] = []

    def fit(self, df_scored: pd.DataFrame) -> np.ndarray | None:
        if "is_injected_anomaly" not in df_scored.columns:
            return None

        labels = df_scored["is_injected_anomaly"].dropna()
        if labels.empty or len(labels) < self.config.min_rows:
            return None

        y = labels.astype(int)
        aligned = df_scored.loc[labels.index]

        candidate_features = [
            "anomaly_score",
            "iforest_score",
            "temporal_score",
            "hybrid_score",
        ] + FeatureConfig.genericFeatures

        self.features = [
            col
            for col in candidate_features
            if col in aligned.columns and np.issubdtype(aligned[col].dtype, np.number)
        ][: self.config.feature_limit]

        if not self.features:
            return None

        X = aligned[self.features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.model = GradientBoostingClassifier(random_state=42)
        self.model.fit(X, y)

        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.features}, self.config.model_path)

        probs = self.model.predict_proba(X)[:, 1]
        df_scored.loc[labels.index, "calibrated_probability"] = probs
        return probs

    def predict(self, df_scored: pd.DataFrame) -> np.ndarray | None:
        bundle = self._load_if_needed()
        if bundle is None:
            return None
        model = bundle["model"]
        features = bundle["features"]
        available = [f for f in features if f in df_scored.columns]
        if not available:
            return None

        X = df_scored[available].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        probs = model.predict_proba(X)[:, 1]
        df_scored["calibrated_probability"] = probs
        return probs

    def _load_if_needed(self) -> dict | None:
        if self.model is not None:
            return {"model": self.model, "features": self.features}
        if not self.config.model_path.exists():
            return None
        bundle = joblib.load(self.config.model_path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        return bundle
