"""
Real Data Training Pipeline for Anomaly Detection.

This module provides a complete ML training pipeline that:
1. Loads real telemetry data from SQL Server (XSight DW)
2. Applies feature engineering with rolling windows
3. Computes data-driven baselines with percentiles
4. Performs time-based train/validation split
5. Trains IsolationForest model
6. Evaluates model on validation data
7. Exports artifacts (model, ONNX, baselines, metadata)
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve

from device_anomaly.models.anomaly_detector import (
    AnomalyDetectorConfig,
    AnomalyDetectorIsolationForest,
)
from device_anomaly.models.baseline import (
    compute_data_driven_baselines,
    save_data_driven_baselines,
)
from device_anomaly.pipeline.validation import (
    PipelineStage,
    PipelineTracker,
    ensure_min_rows,
    drop_all_nan_columns,
    save_model_metadata,
)

logger = logging.getLogger(__name__)


def _get_min_train_rows() -> int:
    """Get minimum training rows from environment or use default."""
    return int(os.getenv("MIN_TRAIN_ROWS", "1000"))


def _get_min_validation_rows() -> int:
    """Get minimum validation rows from environment or use default."""
    return int(os.getenv("MIN_VALIDATION_ROWS", "100"))


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""

    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    validation_days: int = 7  # Last N days used for validation
    contamination: float = 0.03
    n_estimators: int = 300
    random_state: int = 42
    export_onnx: bool = True
    min_train_rows: int = field(default_factory=_get_min_train_rows)
    min_validation_rows: int = field(default_factory=_get_min_validation_rows)
    device_type_col: Optional[str] = "ModelId"
    timestamp_col: str = "CollectedDate"
    row_limit: int = 1_000_000
    # Multi-source training: if True, loads from all configured training sources
    use_multi_source: bool = False
    row_limit_per_source: int = 500_000


@dataclass
class TrainingMetrics:
    """Metrics from model training and validation."""

    train_rows: int
    validation_rows: int
    feature_count: int
    anomaly_rate_train: float
    anomaly_rate_validation: float
    validation_auc: Optional[float] = None
    precision_at_recall_80: Optional[float] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingArtifacts:
    """Paths to training artifacts."""

    model_path: Path
    onnx_path: Optional[Path] = None
    baselines_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    feature_importance_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, str]:
        return {
            k: str(v) if v else None
            for k, v in {
                "model_path": self.model_path,
                "onnx_path": self.onnx_path,
                "baselines_path": self.baselines_path,
                "metadata_path": self.metadata_path,
                "feature_importance_path": self.feature_importance_path,
            }.items()
        }


@dataclass
class TrainingResult:
    """Complete result from a training run."""

    run_id: str
    model_version: str
    config: TrainingConfig
    metrics: TrainingMetrics
    artifacts: TrainingArtifacts
    started_at: str
    completed_at: str
    status: str = "completed"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_version": self.model_version,
            "config": asdict(self.config),
            "metrics": self.metrics.to_dict(),
            "artifacts": self.artifacts.to_dict(),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "error": self.error,
        }


class RealDataTrainingPipeline:
    """
    Orchestrates training on real production data from SQL Server.

    Example usage:
        config = TrainingConfig(
            start_date="2024-01-01",
            end_date="2024-12-31",
            validation_days=7,
        )
        pipeline = RealDataTrainingPipeline(config)
        result = pipeline.run(output_dir="models/production")
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tracker = PipelineTracker()
        self.run_id = str(uuid.uuid4())[:8]
        self._df_train: Optional[pd.DataFrame] = None
        self._df_val: Optional[pd.DataFrame] = None
        self._baselines: Optional[Dict] = None

    def load_training_data(self) -> pd.DataFrame:
        """
        Load and validate training data from the XSight Database.

        If use_multi_source is enabled in config, loads from all configured
        training data sources (multiple customer databases) for better model
        generalization.

        Returns:
            DataFrame with raw telemetry data
        """
        logger.info("Loading training data from DW...")
        logger.info(f"Date range: {self.config.start_date} to {self.config.end_date}")

        if self.config.use_multi_source:
            logger.info("Multi-source training enabled - loading from all configured sources")
            from device_anomaly.data_access.unified_loader import (
                load_multi_source_training_data,
                get_multi_source_summary,
            )

            df = load_multi_source_training_data(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                row_limit_per_source=self.config.row_limit_per_source,
            )

            if not df.empty:
                summary = get_multi_source_summary(df)
                logger.info(f"Multi-source summary: {summary.get('sources', {})}")
        else:
            from device_anomaly.data_access.unified_loader import load_unified_device_dataset

            df = load_unified_device_dataset(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                row_limit=self.config.row_limit,
                include_mc_labels=True,
            )

        if df.empty:
            raise ValueError("No data loaded from DW. Check database connection and date range.")

        logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
        self.tracker.advance(PipelineStage.INGESTION)
        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering: rolling windows, rate calculations, etc.

        Args:
            df: Raw telemetry DataFrame

        Returns:
            DataFrame with engineered features
        """
        logger.info("Applying feature engineering...")

        from device_anomaly.features.device_features import DeviceFeatureBuilder

        # Use existing feature builder
        builder = DeviceFeatureBuilder()
        df_features = builder.build_features(df)

        # Drop all-NaN columns
        df_features = drop_all_nan_columns(df_features, "feature_engineering")

        logger.info(f"Feature engineering complete: {len(df_features.columns)} columns")
        self.tracker.advance(PipelineStage.FEATURES)
        return df_features

    def compute_baselines(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute data-driven baselines from training data.

        Args:
            df: Feature-engineered DataFrame

        Returns:
            Dictionary of DataDrivenBaseline objects
        """
        logger.info("Computing data-driven baselines...")

        # Identify numeric feature columns (exclude IDs)
        exclude_cols = {
            "DeviceId", "ModelId", "ManufacturerId", "OsVersionId",
            "CollectedDate", "Timestamp", "is_injected_anomaly",
        }
        feature_cols = [
            col for col in df.columns
            if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
            and col not in exclude_cols
        ]

        baselines = compute_data_driven_baselines(
            df=df,
            feature_cols=feature_cols,
            timestamp_col=self.config.timestamp_col,
            device_type_col=self.config.device_type_col,
            include_temporal=True,
            min_samples=25,
        )

        logger.info(f"Computed baselines for {len(baselines)} metrics")
        self._baselines = baselines
        self.tracker.advance(PipelineStage.BASELINES)
        return baselines

    def train_validation_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform time-based train/validation split.

        The last `validation_days` of data are used for validation,
        the rest for training. This prevents data leakage.

        Args:
            df: Feature-engineered DataFrame

        Returns:
            Tuple of (train_df, validation_df)
        """
        logger.info("Performing time-based train/validation split...")

        # Ensure timestamp column exists
        ts_col = self.config.timestamp_col
        if ts_col not in df.columns:
            # Try alternate column names
            for alt in ["Timestamp", "timestamp", "Date", "date"]:
                if alt in df.columns:
                    ts_col = alt
                    break

        if ts_col not in df.columns:
            # If no timestamp, use random split as fallback
            logger.warning("No timestamp column found. Using random 80/20 split.")
            mask = np.random.rand(len(df)) < 0.8
            df_train = df[mask].copy()
            df_val = df[~mask].copy()
        else:
            df = df.copy()
            df[ts_col] = pd.to_datetime(df[ts_col])

            # Calculate split date
            max_date = df[ts_col].max()
            split_date = max_date - timedelta(days=self.config.validation_days)

            df_train = df[df[ts_col] < split_date].copy()
            df_val = df[df[ts_col] >= split_date].copy()

            logger.info(f"Split date: {split_date.date()}")

        # Validate splits
        ensure_min_rows(df_train, self.config.min_train_rows, "training_set")
        ensure_min_rows(df_val, self.config.min_validation_rows, "validation_set")

        logger.info(f"Train set: {len(df_train):,} rows, Validation set: {len(df_val):,} rows")

        self._df_train = df_train
        self._df_val = df_val
        return df_train, df_val

    def train_model(self, train_df: pd.DataFrame) -> AnomalyDetectorIsolationForest:
        """
        Train the IsolationForest anomaly detector.

        Args:
            train_df: Training DataFrame

        Returns:
            Trained AnomalyDetectorIsolationForest instance
        """
        logger.info("Training IsolationForest model...")

        detector_config = AnomalyDetectorConfig(
            contamination=self.config.contamination,
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
        )

        detector = AnomalyDetectorIsolationForest(config=detector_config)
        detector.fit(train_df)

        logger.info(f"Model trained on {len(detector.feature_cols)} features")
        self.tracker.advance(PipelineStage.TRAINING)
        return detector

    def evaluate_model(
        self,
        detector: AnomalyDetectorIsolationForest,
        val_df: pd.DataFrame,
    ) -> TrainingMetrics:
        """
        Evaluate model on validation data and compute metrics.

        Args:
            detector: Trained detector
            val_df: Validation DataFrame

        Returns:
            TrainingMetrics with validation results
        """
        logger.info("Evaluating model on validation set...")

        # Score validation data
        val_scored = detector.score_dataframe(val_df)

        # Calculate anomaly rates
        train_scored = detector.score_dataframe(self._df_train)
        train_anomaly_rate = (train_scored["anomaly_label"] == -1).mean()
        val_anomaly_rate = (val_scored["anomaly_label"] == -1).mean()

        # Calculate feature importance (based on isolation depth proxy)
        feature_importance = self._estimate_feature_importance(
            detector, self._df_train
        )

        metrics = TrainingMetrics(
            train_rows=len(self._df_train),
            validation_rows=len(val_df),
            feature_count=len(detector.feature_cols),
            anomaly_rate_train=float(train_anomaly_rate),
            anomaly_rate_validation=float(val_anomaly_rate),
            feature_importance=feature_importance,
        )

        # If we have ground truth labels, compute AUC
        if "is_injected_anomaly" in val_df.columns:
            y_true = val_df["is_injected_anomaly"].values
            y_scores = -val_scored["anomaly_score"].values  # Flip sign for AUC

            try:
                metrics.validation_auc = float(roc_auc_score(y_true, y_scores))
                logger.info(f"Validation AUC: {metrics.validation_auc:.4f}")

                # Precision at 80% recall
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                idx = np.argmin(np.abs(recall - 0.8))
                metrics.precision_at_recall_80 = float(precision[idx])
            except Exception as e:
                logger.warning(f"Could not compute AUC metrics: {e}")

        logger.info(f"Train anomaly rate: {train_anomaly_rate:.2%}")
        logger.info(f"Validation anomaly rate: {val_anomaly_rate:.2%}")

        self.tracker.advance(PipelineStage.SCORING)
        return metrics

    def _estimate_feature_importance(
        self,
        detector: AnomalyDetectorIsolationForest,
        df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Estimate feature importance by measuring score variance contribution.

        This is a simple proxy - features that contribute more to anomaly scores
        are considered more important.
        """
        importance = {}

        if not detector.feature_cols:
            return importance

        try:
            base_scores = detector.score(df)
            base_std = np.std(base_scores)

            for col in detector.feature_cols[:20]:  # Limit to top 20 for efficiency
                df_copy = df.copy()
                # Shuffle the feature
                df_copy[col] = np.random.permutation(df_copy[col].values)
                shuffled_scores = detector.score(df_copy)
                shuffled_std = np.std(shuffled_scores)

                # Features that cause larger score changes when shuffled are more important
                importance[col] = abs(base_std - shuffled_std) / (base_std + 1e-6)

            # Normalize to sum to 1
            total = sum(importance.values()) or 1
            importance = {k: v / total for k, v in importance.items()}

        except Exception as e:
            logger.warning(f"Could not estimate feature importance: {e}")

        return importance

    def export_artifacts(
        self,
        detector: AnomalyDetectorIsolationForest,
        metrics: TrainingMetrics,
        output_dir: str | Path,
    ) -> TrainingArtifacts:
        """
        Save all training artifacts to disk.

        Creates:
            - isolation_forest.pkl (sklearn model)
            - isolation_forest.onnx (ONNX model, optional)
            - baselines.json (data-driven baselines)
            - training_metadata.json (config, metrics, timestamps)
            - feature_importance.json (feature importance scores)

        Args:
            detector: Trained detector
            metrics: Training metrics
            output_dir: Directory for artifacts

        Returns:
            TrainingArtifacts with file paths
        """
        logger.info(f"Exporting artifacts to {output_dir}...")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate version from timestamp
        version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")

        # 1. Save sklearn model (and ONNX if configured)
        model_base = output_dir / "isolation_forest"
        saved_paths = detector.save_model(model_base, export_onnx=self.config.export_onnx)

        artifacts = TrainingArtifacts(
            model_path=saved_paths["sklearn"],
            onnx_path=saved_paths.get("onnx"),
        )

        # 2. Save data-driven baselines
        if self._baselines:
            baselines_path = output_dir / "baselines.json"
            save_data_driven_baselines(self._baselines, baselines_path)
            artifacts.baselines_path = baselines_path

        # 3. Save feature importance
        if metrics.feature_importance:
            importance_path = output_dir / "feature_importance.json"
            importance_path.write_text(json.dumps(metrics.feature_importance, indent=2))
            artifacts.feature_importance_path = importance_path

        # 4. Save training metadata
        metadata = {
            "run_id": self.run_id,
            "model_version": version,
            "config": asdict(self.config),
            "metrics": metrics.to_dict(),
            "feature_cols": detector.feature_cols,
            "artifacts": artifacts.to_dict(),
        }
        metadata_path = output_dir / "training_metadata.json"
        save_model_metadata(metadata_path, metadata)
        artifacts.metadata_path = metadata_path

        self.tracker.advance(PipelineStage.PERSISTENCE)
        logger.info("Artifact export complete")
        return artifacts

    def _persist_training_run(self, result: TrainingResult) -> None:
        """Persist training metadata to the results database."""
        from device_anomaly.api.dependencies import get_tenant_id
        from device_anomaly.database.connection import get_results_db_session
        from device_anomaly.database.schema import TrainingRun

        def _parse_dt(value: Optional[str]) -> Optional[datetime]:
            if not value:
                return None
            try:
                return datetime.fromisoformat(value)
            except Exception:
                return None

        tenant_id = get_tenant_id()
        session = get_results_db_session()
        try:
            run = TrainingRun(
                run_id=result.run_id,
                tenant_id=tenant_id,
                model_version=result.model_version,
                status=result.status,
                config_json=json.dumps(asdict(result.config), default=str),
                metrics_json=json.dumps(result.metrics.to_dict(), default=str),
                artifacts_json=json.dumps(result.artifacts.to_dict(), default=str),
                started_at=_parse_dt(result.started_at),
                completed_at=_parse_dt(result.completed_at),
                error=result.error,
            )
            session.add(run)
            session.commit()
        except Exception as exc:
            logger.warning("Failed to persist training run metadata: %s", exc)
            session.rollback()
        finally:
            session.close()

    def run(self, output_dir: str | Path = "models/production") -> TrainingResult:
        """
        Execute the complete training pipeline.

        Args:
            output_dir: Directory for model artifacts

        Returns:
            TrainingResult with complete run information
        """
        started_at = datetime.now(timezone.utc).isoformat()
        logger.info(f"=== Starting Training Pipeline (run_id: {self.run_id}) ===")

        try:
            # 1. Load data
            df = self.load_training_data()

            # 2. Feature engineering
            df_features = self.prepare_features(df)

            # 3. Train/validation split
            df_train, df_val = self.train_validation_split(df_features)

            # 4. Compute baselines (train-only to avoid leakage)
            self.compute_baselines(df_train)

            # 5. Train model
            detector = self.train_model(df_train)

            # 6. Evaluate
            metrics = self.evaluate_model(detector, df_val)

            # 7. Export artifacts
            artifacts = self.export_artifacts(detector, metrics, output_dir)

            completed_at = datetime.now(timezone.utc).isoformat()
            version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")

            result = TrainingResult(
                run_id=self.run_id,
                model_version=version,
                config=self.config,
                metrics=metrics,
                artifacts=artifacts,
                started_at=started_at,
                completed_at=completed_at,
                status="completed",
            )

            # Save result summary
            result_path = Path(output_dir) / "training_result.json"
            result_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))

            logger.info(f"=== Training Pipeline Complete (run_id: {self.run_id}) ===")
            self._persist_training_run(result)
            return result

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}", exc_info=True)
            completed_at = datetime.now(timezone.utc).isoformat()

            result = TrainingResult(
                run_id=self.run_id,
                model_version="",
                config=self.config,
                metrics=TrainingMetrics(0, 0, 0, 0, 0),
                artifacts=TrainingArtifacts(Path("")),
                started_at=started_at,
                completed_at=completed_at,
                status="failed",
                error=str(e),
            )
            self._persist_training_run(result)
            return result


def train_production_model(
    start_date: str,
    end_date: str,
    validation_days: int = 7,
    output_dir: str = "models/production",
    export_onnx: bool = True,
    contamination: float = 0.03,
    n_estimators: int = 300,
    use_multi_source: bool = False,
    row_limit_per_source: int = 500_000,
) -> TrainingResult:
    """
    Convenience function to train a production model.

    Args:
        start_date: Start date for training data (YYYY-MM-DD)
        end_date: End date for training data (YYYY-MM-DD)
        validation_days: Days to hold out for validation
        output_dir: Directory for model artifacts
        export_onnx: Whether to export ONNX model
        contamination: Expected anomaly rate (0-1)
        n_estimators: Number of trees in IsolationForest
        use_multi_source: If True, train on data from all configured sources
                          (TRAINING_DATA_SOURCES). This creates a more robust
                          model that generalizes across customer environments.
        row_limit_per_source: Max rows per source when use_multi_source=True

    Returns:
        TrainingResult with complete run information

    Example:
        # Single-source training (default)
        result = train_production_model("2024-01-01", "2024-12-31")

        # Multi-source training from BENELUX + PIBLIC databases
        result = train_production_model(
            "2024-01-01", "2024-12-31",
            use_multi_source=True,
        )
    """
    config = TrainingConfig(
        start_date=start_date,
        end_date=end_date,
        validation_days=validation_days,
        contamination=contamination,
        n_estimators=n_estimators,
        export_onnx=export_onnx,
        use_multi_source=use_multi_source,
        row_limit_per_source=row_limit_per_source,
    )

    pipeline = RealDataTrainingPipeline(config)
    return pipeline.run(output_dir=output_dir)
