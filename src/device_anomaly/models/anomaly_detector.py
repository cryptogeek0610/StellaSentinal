from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class AnomalyDetectorConfig:
    contamination: float = 0.05  # Increased default: 5% expected anomaly rate
    n_estimators: int = 300
    random_state: int = 42
    min_variance: float = 1e-6
    scale_features: bool = True
    feature_domain_weights: dict[str, float] | None = None
    # Adaptive scoring: use percentile-based thresholding when model threshold fails
    use_adaptive_threshold: bool = True
    adaptive_percentile: float = 5.0  # Bottom 5% of scores = anomalies


class AnomalyDetectorIsolationForest:
    """
    Thin wrapper around IsolationForest to keep things tidy.

    Usage:
        detector = AnomalyDetectorIsolationForest()
        detector.fit(df_features)
        df_scored = detector.score_dataframe(df_features)
    """

    def __init__(self, config: Optional[AnomalyDetectorConfig] = None, feature_overrides: Optional[List[str]] = None):
        self.config = config or AnomalyDetectorConfig()
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        self.feature_cols: List[str] = []
        self.impute_values: Optional[pd.Series] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_overrides = feature_overrides
        self.feature_weights: dict[str, float] = {}
        self.model_path: Optional[Path] = None
        self.onnx_engine = None

    def _select_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Select numeric feature columns for training.

        Strategy:
        - Prefer cohort-normalized features (*_cohort_z) if they exist.
        - Otherwise fall back to all numeric telemetry columns,
          excluding IDs / labels / flags.
        """
        if self.feature_overrides:
            overrides = [c for c in self.feature_overrides if c in df.columns]
            if not overrides:
                raise ValueError(
                    f"None of the specified feature overrides exist in dataframe: {self.feature_overrides}"
                )
            numeric_overrides = [
                c for c in overrides
                if np.issubdtype(df[c].dtype, np.number)
            ]
            missing_numeric = sorted(set(overrides) - set(numeric_overrides))
            if missing_numeric:
                raise ValueError(
                    f"Feature overrides must be numeric columns. Non-numeric: {missing_numeric}"
                )
            return numeric_overrides

        # 1) All numeric columns (use pandas.api.types for robust dtype checking)
        import pandas.api.types as ptypes

        numeric_cols: list[str] = [
            c
            for c in df.columns
            if ptypes.is_numeric_dtype(df[c])
        ]

        # 2) Exclude IDs / labels / targets
        exclude = {
            "DeviceId",
            "ModelId",
            "ManufacturerId",
            "OsVersionId",
            "is_injected_anomaly",  # synthetic ground truth flag
            "anomaly_score",
            "anomaly_label",
        }

        candidates = [
            c for c in numeric_cols
            if c not in exclude
        ]

        # 3) Prefer baseline/cohort normalized columns if present (DW with cohorts)
        baseline_cols = [c for c in candidates if "_z_" in c]
        cohort_cols = [c for c in candidates if c.endswith("_cohort_z")]

        feature_cols = baseline_cols or cohort_cols or candidates

        if not feature_cols:
            raise ValueError("No feature columns found to train on.")

        return feature_cols

    def _prepare_training_matrix(self, df: pd.DataFrame) -> np.ndarray:
        feature_df = df[self.feature_cols].copy()
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        medians = feature_df.median()
        medians = medians.fillna(0.0)
        self.impute_values = medians
        feature_df = feature_df.fillna(self.impute_values)

        # Apply domain-based feature weighting to stop any single category (e.g. battery) from dominating.
        self.feature_weights = self._resolve_feature_weights(feature_df.columns)
        feature_df = self._apply_feature_weights(feature_df)

        variances = feature_df.var(ddof=0).fillna(0.0)
        keep_mask = variances > self.config.min_variance
        kept_cols = list(variances[keep_mask].index)
        if not kept_cols:
            raise ValueError("All feature columns have near-zero variance.")

        dropped = sorted(set(self.feature_cols) - set(kept_cols))
        if dropped:
            logging.getLogger(__name__).info(
                "Dropping %d near-constant features: %s",
                len(dropped),
                ", ".join(dropped[:20]),
            )

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
        if not self.feature_cols:
            raise RuntimeError("Model has not been fit yet.")

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Input dataframe is missing required feature columns: {missing}"
            )

        feature_df = df[self.feature_cols].copy()
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        if self.impute_values is None:
            raise RuntimeError("Imputation values missing; has the model been fit?")

        feature_df = feature_df.fillna(self.impute_values)
        feature_df = self._apply_feature_weights(feature_df)

        if self.config.scale_features and self.scaler is not None:
            return self.scaler.transform(feature_df.values)

        return feature_df.values

    def fit(self, df: pd.DataFrame) -> None:
        self.feature_cols = self._select_feature_columns(df)
        logger = logging.getLogger(__name__)
        logger.info(
            "Training IsolationForest on %d features: %s",
            len(self.feature_cols),
            ", ".join(self.feature_cols[:20]),
        )
        X = self._prepare_training_matrix(df)
        self.model.fit(X)

    def save_model(self, output_path: str | Path, export_onnx: bool = False) -> dict[str, Path]:
        """
        Save trained model to disk.

        Args:
            output_path: Base path for model files (without extension)
            export_onnx: If True, also export to ONNX format

        Returns:
            Dictionary with paths to saved models {'sklearn': path, 'onnx': path}

        Example:
            detector.fit(df)
            paths = detector.save_model("models/isolation_forest", export_onnx=True)
            # Saves: models/isolation_forest.pkl and models/isolation_forest.onnx
        """
        logger = logging.getLogger(__name__)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        # Save full detector state (model + preprocessing) with joblib
        sklearn_path = output_path.with_suffix(".pkl")
        detector_state = {
            "model": self.model,
            "feature_cols": self.feature_cols,
            "impute_values": self.impute_values.to_dict() if self.impute_values is not None else None,
            "scaler": self.scaler,
            "config": self.config,
            "feature_weights": self.feature_weights,
        }
        joblib.dump(detector_state, sklearn_path)
        self.model_path = sklearn_path
        saved_paths["sklearn"] = sklearn_path
        logger.info("Saved detector state to %s", sklearn_path)

        # Export to ONNX if requested
        if export_onnx:
            try:
                from device_anomaly.models.onnx_exporter import ONNXModelExporter, ONNXExportConfig

                onnx_path = output_path.with_suffix(".onnx")
                exporter = ONNXModelExporter(ONNXExportConfig())

                # Export with metadata
                metadata = {
                    "n_estimators": self.config.n_estimators,
                    "contamination": self.config.contamination,
                    "n_features": len(self.feature_cols),
                    "feature_names": ",".join(self.feature_cols[:50]),  # Limit metadata size
                }

                exporter.export_with_metadata(
                    model=self.model,
                    feature_count=len(self.feature_cols),
                    output_path=onnx_path,
                    metadata=metadata,
                )

                saved_paths["onnx"] = onnx_path
                logger.info("Exported ONNX model to %s", onnx_path)

                # Also create quantized version if requested
                from device_anomaly.config.onnx_config import get_onnx_config

                onnx_config = get_onnx_config()
                if onnx_config.export.export_quantized:
                    from device_anomaly.models.onnx_exporter import ONNXQuantizer

                    quantized_path = output_path.with_name(output_path.stem + "_int8").with_suffix(".onnx")
                    ONNXQuantizer.quantize_dynamic(onnx_path, quantized_path)
                    saved_paths["onnx_quantized"] = quantized_path
                    logger.info("Created quantized ONNX model at %s", quantized_path)

            except Exception as e:
                logger.warning("Failed to export ONNX model: %s", e, exc_info=True)
                logger.warning("Continuing with sklearn-only model...")

        return saved_paths

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare_inference_matrix(df)
        return self._score_matrix(X)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare_inference_matrix(df)
        return self._predict_matrix(X)

    def score_dataframe(self, df: pd.DataFrame, use_adaptive: bool | None = None) -> pd.DataFrame:
        """
        Score a dataframe and return with anomaly scores and labels.

        Args:
            df: Input dataframe with features
            use_adaptive: Override config's use_adaptive_threshold setting

        Returns:
            DataFrame with anomaly_score and anomaly_label columns added.

        Notes on Isolation Forest scoring:
        - decision_function returns scores where LOWER = more anomalous
        - The model's offset_ is the threshold: score < offset_ => anomaly
        - When data distribution shifts, the model threshold may not work well
        - Adaptive thresholding uses percentile-based approach as fallback
        """
        logger = logging.getLogger(__name__)
        df_scored = df.copy()
        X = self._prepare_inference_matrix(df_scored)
        scores = self._score_matrix(X)

        # Get model-based predictions first
        model_labels = self._predict_matrix(X)
        model_anomaly_count = int((model_labels == -1).sum())

        # Determine if we should use adaptive thresholding
        should_use_adaptive = use_adaptive if use_adaptive is not None else self.config.use_adaptive_threshold

        # Check if model threshold is producing reasonable results
        # If ALL scores are above threshold (no anomalies), the model may have distribution drift
        if should_use_adaptive and model_anomaly_count == 0 and len(scores) > 10:
            # All scores are "normal" - likely distribution drift
            # Use percentile-based adaptive threshold instead
            adaptive_threshold = np.percentile(scores, self.config.adaptive_percentile)
            adaptive_labels = np.where(scores <= adaptive_threshold, -1, 1)
            adaptive_anomaly_count = int((adaptive_labels == -1).sum())

            logger.info(
                f"Adaptive thresholding activated: model found 0 anomalies, "
                f"adaptive found {adaptive_anomaly_count} (bottom {self.config.adaptive_percentile}% of scores)"
            )
            logger.debug(
                f"Score stats: min={scores.min():.4f}, max={scores.max():.4f}, "
                f"model_offset={self.model.offset_:.4f}, adaptive_threshold={adaptive_threshold:.4f}"
            )

            labels = adaptive_labels
        else:
            labels = model_labels
            if model_anomaly_count > 0:
                logger.info(f"Model-based detection: {model_anomaly_count}/{len(scores)} anomalies")

        df_scored["anomaly_score"] = scores     # lower is more anomalous
        df_scored["anomaly_label"] = labels     # 1 normal, -1 anomaly
        return df_scored

    def _resolve_onnx_path(self) -> Optional[Path]:
        if self.model_path:
            candidate = self.model_path.with_suffix(".onnx")
            if candidate.exists():
                return candidate
        try:
            from device_anomaly.models.model_registry import resolve_model_artifacts, resolve_artifact_path

            artifacts = resolve_model_artifacts()
            if artifacts.metadata and artifacts.metadata.get("artifacts"):
                onnx_path = artifacts.metadata["artifacts"].get("onnx_path")
                resolved = resolve_artifact_path(artifacts.model_dir, onnx_path)
                if resolved and resolved.exists():
                    return resolved
        except Exception:
            return None
        return None

    def _get_onnx_engine(self):
        if self.onnx_engine is not None:
            return self.onnx_engine
        try:
            from device_anomaly.config.onnx_config import get_onnx_config
            from device_anomaly.models.inference_engine import EngineConfig, EngineType, ONNXInferenceEngine

            onnx_config = get_onnx_config()
            if onnx_config.inference.engine_type != EngineType.ONNX:
                return None

            onnx_path = self._resolve_onnx_path()
            if not onnx_path:
                return None

            engine_config = EngineConfig(
                engine_type=EngineType.ONNX,
                onnx_provider=onnx_config.inference.onnx_provider,
                intra_op_num_threads=onnx_config.inference.intra_op_num_threads,
                inter_op_num_threads=onnx_config.inference.inter_op_num_threads,
                enable_profiling=onnx_config.inference.enable_profiling,
                enable_graph_optimization=onnx_config.inference.enable_graph_optimization,
                collect_metrics=onnx_config.inference.collect_metrics,
            )
            self.onnx_engine = ONNXInferenceEngine(onnx_path, config=engine_config)
        except Exception as exc:
            logging.getLogger(__name__).warning("ONNX engine unavailable, falling back to sklearn: %s", exc)
            self.onnx_engine = None
        return self.onnx_engine

    def _score_matrix(self, X: np.ndarray) -> np.ndarray:
        engine = self._get_onnx_engine()
        if engine is not None:
            return engine.score_samples(X)
        return self.model.decision_function(X)

    def _predict_matrix(self, X: np.ndarray) -> np.ndarray:
        engine = self._get_onnx_engine()
        if engine is not None:
            return engine.predict(X)
        return self.model.predict(X)

    # ------------------------------------------------------------------ #
    # Feature weighting helpers
    # ------------------------------------------------------------------ #
    def _resolve_feature_weights(self, columns: list[str]) -> dict[str, float]:
        from device_anomaly.config.feature_config import FeatureConfig

        weights = self.config.feature_domain_weights or FeatureConfig.domain_weights
        domains = FeatureConfig.feature_domains
        resolved: dict[str, float] = {}

        for col in columns:
            root = self._root_feature(col)
            domain = domains.get(root)
            if domain and domain in weights:
                resolved[col] = weights[domain]
            else:
                resolved[col] = 1.0
        return resolved

    @staticmethod
    def _root_feature(col: str) -> str:
        suffixes = [
            "_roll_mean",
            "_roll_std",
            "_delta",
            "_cohort_z",
        ]
        for suf in suffixes:
            if col.endswith(suf):
                return col[: -len(suf)]
        if "_z_" in col:
            return col.split("_z_")[0]
        return col

    def _apply_feature_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_weights:
            return df
        weight_series = pd.Series(self.feature_weights)
        # align on columns; missing weights default to 1
        aligned = weight_series.reindex(df.columns).fillna(1.0)
        return df.mul(aligned, axis=1)

    @classmethod
    def load_model(cls, model_path: str | Path) -> "AnomalyDetectorIsolationForest":
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the saved .pkl model file

        Returns:
            Loaded AnomalyDetectorIsolationForest instance
        """
        logger = logging.getLogger(__name__)
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load the saved state
        loaded = joblib.load(model_path)

        # Create instance
        instance = cls()
        instance.model_path = model_path

        # Handle both new format (dict with full state) and old format (just model)
        if isinstance(loaded, dict) and "model" in loaded:
            # New format: full detector state
            instance.model = loaded["model"]
            instance.feature_cols = loaded.get("feature_cols", [])
            if loaded.get("impute_values"):
                instance.impute_values = pd.Series(loaded["impute_values"])
            instance.scaler = loaded.get("scaler")
            if loaded.get("config"):
                old_config = loaded["config"]
                # Merge old config with new defaults (for adaptive thresholding)
                instance.config = AnomalyDetectorConfig(
                    contamination=getattr(old_config, 'contamination', 0.05),
                    n_estimators=getattr(old_config, 'n_estimators', 300),
                    random_state=getattr(old_config, 'random_state', 42),
                    min_variance=getattr(old_config, 'min_variance', 1e-6),
                    scale_features=getattr(old_config, 'scale_features', True),
                    feature_domain_weights=getattr(old_config, 'feature_domain_weights', None),
                    use_adaptive_threshold=getattr(old_config, 'use_adaptive_threshold', True),
                    adaptive_percentile=getattr(old_config, 'adaptive_percentile', 5.0),
                )
            instance.feature_weights = loaded.get("feature_weights", {})
            logger.info("Loaded full detector state from %s with %d features", model_path, len(instance.feature_cols))
        else:
            # Old format: just the sklearn model
            instance.model = loaded

            # Try to load metadata for feature columns from companion file
            metadata_path = model_path.with_name("training_metadata.json")
            if metadata_path.exists():
                import json
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    # Check both possible keys for feature columns
                    if "feature_cols" in metadata:
                        instance.feature_cols = metadata["feature_cols"]
                    elif "feature_columns" in metadata:
                        instance.feature_cols = metadata["feature_columns"]
                    if "impute_values" in metadata:
                        instance.impute_values = pd.Series(metadata["impute_values"])
                    if "config" in metadata:
                        config_data = metadata["config"]
                        instance.config = AnomalyDetectorConfig(
                            contamination=config_data.get("contamination", 0.03),
                            n_estimators=config_data.get("n_estimators", 300),
                        )
            logger.info("Loaded legacy model from %s with %d features", model_path, len(instance.feature_cols))

        return instance

    @classmethod
    def load_latest(cls, models_dir: str | Path | None = None) -> Optional["AnomalyDetectorIsolationForest"]:
        """
        Load the most recently trained production model.

        Args:
            models_dir: Directory containing model files. Defaults to 'models/production'

        Returns:
            Loaded model instance, or None if no model found
        """
        logger = logging.getLogger(__name__)

        if models_dir is None:
            # Default to models/production relative to project root
            models_dir = Path(__file__).parent.parent.parent.parent / "models" / "production"

        models_dir = Path(models_dir)

        if not models_dir.exists():
            logger.warning("Models directory not found: %s", models_dir)
            return None

        # Find isolation forest model files specifically (not other .pkl files like cohort_detector.pkl)
        model_files = list(models_dir.glob("isolation_forest.pkl"))
        if not model_files:
            # Also check subdirectories (for versioned models)
            model_files = list(models_dir.glob("**/isolation_forest.pkl"))

        if not model_files:
            logger.warning("No model files found in %s", models_dir)
            return None

        # Sort by modification time, most recent first
        model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_model = model_files[0]

        logger.info("Loading latest model: %s", latest_model)
        return cls.load_model(latest_model)


# Alias for convenience
AnomalyDetector = AnomalyDetectorIsolationForest
