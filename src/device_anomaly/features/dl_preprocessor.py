"""Deep Learning feature preprocessor.

This module provides preprocessing utilities specifically designed for
DL models. It handles:
- Feature selection and filtering
- Missing value imputation
- Normalization/scaling
- Outlier clipping
- Data type conversion for PyTorch
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


@dataclass
class DLPreprocessorConfig:
    """Configuration for DL feature preprocessing.

    Attributes:
        normalize_method: Scaling method ('standard', 'minmax', 'robust', 'none')
        handle_missing: Missing value strategy ('impute_median', 'impute_mean', 'impute_zero', 'drop')
        clip_outliers: Whether to clip extreme outlier values
        outlier_std: Number of standard deviations for clipping threshold
        min_variance: Minimum variance threshold for feature selection
        exclude_patterns: Column name patterns to exclude
        prefer_cohort_features: Prefer *_cohort_z features if available
        max_features: Maximum number of features to keep (None for no limit)
    """
    normalize_method: str = "standard"
    handle_missing: str = "impute_median"
    clip_outliers: bool = True
    outlier_std: float = 5.0
    min_variance: float = 1e-6
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "DeviceId", "ModelId", "ManufacturerId", "OsVersionId",
        "is_injected_anomaly", "anomaly_score", "anomaly_label",
    ])
    prefer_cohort_features: bool = True
    max_features: Optional[int] = None


class DLFeaturePreprocessor:
    """Preprocessor for preparing features for Deep Learning models.

    This class provides a sklearn-like fit/transform interface for
    preprocessing tabular data for DL anomaly detection models.

    Example:
        preprocessor = DLFeaturePreprocessor()
        X_train = preprocessor.fit_transform(train_df)
        X_test = preprocessor.transform(test_df)

        # Use with VAE detector
        detector = VAEDetector(config)
        detector.fit(train_df)  # Uses internal preprocessing
    """

    def __init__(self, config: Optional[DLPreprocessorConfig] = None):
        """Initialize the preprocessor.

        Args:
            config: DLPreprocessorConfig with preprocessing settings
        """
        self.config = config or DLPreprocessorConfig()
        self._feature_cols: List[str] = []
        self._impute_values: Optional[pd.Series] = None
        self._scaler: Optional[Any] = None
        self._clip_bounds: Optional[Dict[str, Tuple[float, float]]] = None
        self._is_fitted = False

    @property
    def feature_cols(self) -> List[str]:
        """Get the selected feature columns."""
        return self._feature_cols

    @property
    def is_fitted(self) -> bool:
        """Check if preprocessor has been fitted."""
        return self._is_fitted

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """Select numeric feature columns for training.

        Args:
            df: Input DataFrame

        Returns:
            List of selected feature column names
        """
        import pandas.api.types as ptypes

        # Get all numeric columns
        numeric_cols = [
            c for c in df.columns
            if ptypes.is_numeric_dtype(df[c])
        ]

        # Exclude specified patterns
        candidates = [
            c for c in numeric_cols
            if c not in self.config.exclude_patterns
        ]

        # Prefer cohort-normalized features if configured
        if self.config.prefer_cohort_features:
            cohort_cols = [c for c in candidates if c.endswith("_cohort_z")]
            baseline_cols = [c for c in candidates if "_z_" in c]
            if cohort_cols:
                candidates = cohort_cols
            elif baseline_cols:
                candidates = baseline_cols

        if not candidates:
            raise ValueError("No feature columns found after selection")

        return candidates

    def _create_scaler(self):
        """Create the appropriate scaler based on config."""
        method = self.config.normalize_method.lower()
        if method == "standard":
            return StandardScaler()
        elif method == "minmax":
            return MinMaxScaler()
        elif method == "robust":
            return RobustScaler()
        elif method == "none":
            return None
        else:
            raise ValueError(f"Unknown normalize_method: {method}")

    def fit(self, df: pd.DataFrame) -> "DLFeaturePreprocessor":
        """Fit the preprocessor on training data.

        Args:
            df: Training DataFrame

        Returns:
            self for method chaining
        """
        # Select features
        self._feature_cols = self._select_features(df)

        # Get feature data
        feature_df = df[self._feature_cols].copy()

        # Replace infinities with NaN
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Compute imputation values
        if self.config.handle_missing == "impute_median":
            self._impute_values = feature_df.median().fillna(0.0)
        elif self.config.handle_missing == "impute_mean":
            self._impute_values = feature_df.mean().fillna(0.0)
        elif self.config.handle_missing == "impute_zero":
            self._impute_values = pd.Series(0.0, index=self._feature_cols)
        else:
            self._impute_values = pd.Series(0.0, index=self._feature_cols)

        # Apply imputation for variance calculation
        feature_df = feature_df.fillna(self._impute_values)

        # Remove near-constant features
        variances = feature_df.var(ddof=0).fillna(0.0)
        keep_mask = variances > self.config.min_variance
        kept_cols = list(variances[keep_mask].index)

        if not kept_cols:
            raise ValueError("All feature columns have near-zero variance")

        self._feature_cols = kept_cols
        self._impute_values = self._impute_values[kept_cols]
        feature_df = feature_df[kept_cols]

        # Limit features if configured
        if self.config.max_features and len(self._feature_cols) > self.config.max_features:
            # Keep top features by variance
            var_sorted = variances[kept_cols].sort_values(ascending=False)
            self._feature_cols = list(var_sorted.head(self.config.max_features).index)
            self._impute_values = self._impute_values[self._feature_cols]
            feature_df = feature_df[self._feature_cols]

        # Compute clipping bounds before scaling
        if self.config.clip_outliers:
            self._clip_bounds = {}
            for col in self._feature_cols:
                mean = feature_df[col].mean()
                std = feature_df[col].std()
                if std > 0:
                    lower = mean - self.config.outlier_std * std
                    upper = mean + self.config.outlier_std * std
                    self._clip_bounds[col] = (lower, upper)

        # Apply clipping
        if self.config.clip_outliers and self._clip_bounds:
            for col, (lower, upper) in self._clip_bounds.items():
                feature_df[col] = feature_df[col].clip(lower, upper)

        # Fit scaler
        self._scaler = self._create_scaler()
        if self._scaler is not None:
            self._scaler.fit(feature_df.values)

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor.

        Args:
            df: DataFrame to transform

        Returns:
            Numpy array of shape (n_samples, n_features), dtype float32
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor has not been fitted")

        # Check for missing columns
        missing = [c for c in self._feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Get feature data
        feature_df = df[self._feature_cols].copy()

        # Replace infinities
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Impute missing values
        feature_df = feature_df.fillna(self._impute_values)

        # Apply clipping
        if self.config.clip_outliers and self._clip_bounds:
            for col, (lower, upper) in self._clip_bounds.items():
                if col in feature_df.columns:
                    feature_df[col] = feature_df[col].clip(lower, upper)

        # Scale
        if self._scaler is not None:
            data = self._scaler.transform(feature_df.values)
        else:
            data = feature_df.values

        return data.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            df: Training DataFrame

        Returns:
            Transformed numpy array
        """
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        """Inverse transform scaled data back to original scale.

        Args:
            data: Scaled numpy array

        Returns:
            DataFrame with original scale values
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor has not been fitted")

        if self._scaler is not None:
            unscaled = self._scaler.inverse_transform(data)
        else:
            unscaled = data

        return pd.DataFrame(unscaled, columns=self._feature_cols)

    def get_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each feature.

        Returns:
            Dict mapping feature name to stats (mean, std, etc.)
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor has not been fitted")

        stats = {}
        for i, col in enumerate(self._feature_cols):
            feature_stats = {
                "impute_value": float(self._impute_values[col]),
            }

            if self._scaler is not None:
                if hasattr(self._scaler, 'mean_'):
                    feature_stats["scaler_mean"] = float(self._scaler.mean_[i])
                if hasattr(self._scaler, 'scale_'):
                    feature_stats["scaler_scale"] = float(self._scaler.scale_[i])

            if self._clip_bounds and col in self._clip_bounds:
                feature_stats["clip_lower"] = float(self._clip_bounds[col][0])
                feature_stats["clip_upper"] = float(self._clip_bounds[col][1])

            stats[col] = feature_stats

        return stats

    def save(self, path: str) -> None:
        """Save preprocessor state to file.

        Args:
            path: Path to save file
        """
        import joblib

        state = {
            "config": self.config,
            "feature_cols": self._feature_cols,
            "impute_values": self._impute_values.to_dict() if self._impute_values is not None else None,
            "scaler": self._scaler,
            "clip_bounds": self._clip_bounds,
            "is_fitted": self._is_fitted,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> "DLFeaturePreprocessor":
        """Load preprocessor from file.

        Args:
            path: Path to saved file

        Returns:
            Loaded preprocessor instance
        """
        import joblib

        state = joblib.load(path)

        instance = cls(config=state.get("config"))
        instance._feature_cols = state.get("feature_cols", [])
        if state.get("impute_values"):
            instance._impute_values = pd.Series(state["impute_values"])
        instance._scaler = state.get("scaler")
        instance._clip_bounds = state.get("clip_bounds")
        instance._is_fitted = state.get("is_fitted", False)

        return instance

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        n_features = len(self._feature_cols) if self._feature_cols else 0
        return f"<DLFeaturePreprocessor({status}, {n_features} features)>"


def create_dl_preprocessor(
    normalize_method: str = "standard",
    clip_outliers: bool = True,
    prefer_cohort_features: bool = True,
    **kwargs
) -> DLFeaturePreprocessor:
    """Create a DL preprocessor with common configuration.

    Args:
        normalize_method: Scaling method
        clip_outliers: Whether to clip outliers
        prefer_cohort_features: Prefer cohort-normalized features
        **kwargs: Additional config parameters

    Returns:
        Configured DLFeaturePreprocessor
    """
    config = DLPreprocessorConfig(
        normalize_method=normalize_method,
        clip_outliers=clip_outliers,
        prefer_cohort_features=prefer_cohort_features,
        **kwargs
    )
    return DLFeaturePreprocessor(config)
