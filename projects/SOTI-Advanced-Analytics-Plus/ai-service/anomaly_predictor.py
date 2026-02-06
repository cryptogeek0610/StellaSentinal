import logging
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError

logger = logging.getLogger("SAAP.ML")


class SAAPAnomalyPredictor:
    """
    Production-grade Anomaly Detection for SAAP Plus.
    Uses Isolation Forest for multivariate outlier detection.
    """
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self._is_fitted = False

    def train(self, feature_matrix):
        """Train on nominal fleet behavior."""
        feature_matrix = np.asarray(feature_matrix)
        if feature_matrix.size == 0:
            raise ValueError("Cannot train on empty feature matrix.")
        if np.any(np.isnan(feature_matrix)) or np.any(np.isinf(feature_matrix)):
            raise ValueError("Feature matrix contains NaN or Inf values.")
        self.model.fit(feature_matrix)
        self._is_fitted = True
        logger.info("Anomaly model retrained on %d samples.", feature_matrix.shape[0])

    def predict(self, current_data):
        """Predict anomalies in current fleet telemetry.

        Returns (predictions, scores) where predictions are -1 for anomaly, 1 for normal.
        """
        if not self._is_fitted:
            raise NotFittedError(
                "SAAPAnomalyPredictor has not been trained yet. Call train() first."
            )
        current_data = np.asarray(current_data)
        if current_data.size == 0:
            raise ValueError("Cannot predict on empty data.")
        preds = self.model.predict(current_data)
        scores = self.model.decision_function(current_data)
        return preds, scores

    def save_model(self, path):
        """Persist trained model to disk via joblib."""
        import joblib
        if not self._is_fitted:
            raise NotFittedError("Cannot save an untrained model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("Model saved to %s", path)

    def load_model(self, path):
        """Load a previously persisted model from disk."""
        import joblib
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(path)
        self._is_fitted = True
        logger.info("Model loaded from %s", path)


if __name__ == "__main__":
    predictor = SAAPAnomalyPredictor()
    print("SAAP Anomaly Predictor Initialized.")
