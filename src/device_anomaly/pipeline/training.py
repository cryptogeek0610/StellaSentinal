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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score

from device_anomaly.features.cohort_stats import (
    CohortStatsStore,
    apply_cohort_stats,
    compute_cohort_stats,
    save_cohort_stats,
)
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
    drop_all_nan_columns,
    ensure_min_rows,
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
    device_type_col: str | None = "ModelId"
    timestamp_col: str = "Timestamp"
    row_limit: int = 1_000_000
    # Multi-source training: if True, loads from all configured training sources
    use_multi_source: bool = False
    row_limit_per_source: int = 500_000
    # Ensemble detector configuration
    use_ensemble: bool = False
    ensemble_contamination: float = 0.05
    ensemble_weights: dict[str, float] | None = None  # IF, LOF, OCSVM weights
    # SHAP explanations
    enable_shap: bool = False
    shap_background_samples: int = 100
    # Extended features
    use_extended_features: bool = False
    include_location_features: bool = True
    include_event_features: bool = True
    include_system_health_features: bool = True
    include_wifi_features: bool = True
    # Hourly granularity
    include_hourly_data: bool = False
    hourly_tables: list[str] = field(
        default_factory=lambda: ["cs_DataUsageByHour", "cs_BatteryLevelDrop", "cs_WifiHour"]
    )
    hourly_aggregation: str = "device_day"  # "hourly", "device_day", "device_hour"
    hourly_windows: list[int] = field(default_factory=lambda: [6, 12, 24, 48])
    hourly_max_days: int = 7  # Limit hourly data to recent N days
    # Auto-discovery
    use_auto_discovery: bool = False
    discovery_min_rows: int = 10000
    # Feature selection for high dimensionality
    max_features: int = 200
    enable_feature_selection: bool = True
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    # Memory management for large datasets
    chunk_size: int = 100_000
    # Enhanced evaluation
    compute_cohort_fairness: bool = True
    compute_score_stability: bool = True


@dataclass
class TrainingMetrics:
    """Metrics from model training and validation."""

    train_rows: int
    validation_rows: int
    feature_count: int
    anomaly_rate_train: float
    anomaly_rate_validation: float
    validation_auc: float | None = None
    precision_at_recall_80: float | None = None
    feature_importance: dict[str, float] = field(default_factory=dict)
    # Ensemble-specific metrics
    ensemble_algorithm_scores: dict[str, float] | None = None
    # Enhanced evaluation metrics
    score_stability: dict[str, float] | None = None
    cohort_fairness: dict[str, Any] | None = None
    temporal_stability: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingArtifacts:
    """Paths to training artifacts."""

    model_path: Path
    onnx_path: Path | None = None
    baselines_path: Path | None = None
    cohort_stats_path: Path | None = None
    cohort_detector_path: Path | None = None
    metadata_path: Path | None = None
    feature_importance_path: Path | None = None

    def to_dict(self) -> dict[str, str]:
        return {
            k: str(v) if v else None
            for k, v in {
                "model_path": self.model_path,
                "onnx_path": self.onnx_path,
                "baselines_path": self.baselines_path,
                "cohort_stats_path": self.cohort_stats_path,
                "cohort_detector_path": self.cohort_detector_path,
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
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
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
        self.model_version: str | None = None
        self._df_train: pd.DataFrame | None = None
        self._df_val: pd.DataFrame | None = None
        self._baselines: dict | None = None
        self._cohort_stats_payload: dict[str, Any] | None = None
        self._feature_norms: dict[str, float] | None = None
        self._feature_spec: dict[str, Any] | None = None

    def load_training_data(self) -> pd.DataFrame:
        """
        Load and validate training data from the XSight Database.

        Supports multiple loading modes:
        - Standard: 5 core XSight tables + MC DevInfo (default)
        - Extended: Includes location, events, system health data
        - Hourly: Includes hourly granularity tables (cs_DataUsageByHour, etc.)
        - Auto-discovery: Dynamically discovers high-value tables
        - Multi-source: Loads from multiple customer databases

        Returns:
            DataFrame with raw telemetry data
        """
        logger.info("Loading training data from DW...")
        logger.info(f"Date range: {self.config.start_date} to {self.config.end_date}")
        logger.info(
            f"Mode: extended={self.config.use_extended_features}, hourly={self.config.include_hourly_data}, "
            f"discovery={self.config.use_auto_discovery}, multi_source={self.config.use_multi_source}"
        )

        # Multi-source takes precedence (combines data from multiple databases)
        if self.config.use_multi_source:
            logger.info("Multi-source training enabled - loading from all configured sources")
            from device_anomaly.data_access.unified_loader import (
                get_multi_source_summary,
                load_multi_source_training_data,
            )

            df = load_multi_source_training_data(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                row_limit_per_source=self.config.row_limit_per_source,
            )

            if not df.empty:
                summary = get_multi_source_summary(df)
                logger.info(f"Multi-source summary: {summary.get('sources', {})}")

        # Extended features: location, events, system health, WiFi
        elif self.config.use_extended_features:
            logger.info("Extended features enabled - loading comprehensive dataset")
            from device_anomaly.data_access.unified_loader import load_extended_device_dataset

            df = load_extended_device_dataset(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                row_limit=self.config.row_limit,
                include_location=self.config.include_location_features,
                include_events=self.config.include_event_features,
                include_system_health=self.config.include_system_health_features,
                include_wifi=self.config.include_wifi_features,
            )

        # Auto-discovery: dynamically find high-value tables
        elif self.config.use_auto_discovery:
            logger.info("Auto-discovery enabled - discovering high-value tables")
            df = self._load_discovered_tables()

        # Standard: 5 core tables + MC DevInfo
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

        # Optionally merge hourly data
        if self.config.include_hourly_data:
            df = self._merge_hourly_data(df)

        self.tracker.advance(PipelineStage.INGESTION)
        return df

    def _load_discovered_tables(self) -> pd.DataFrame:
        """Load training data using auto-discovered high-value tables."""
        from device_anomaly.data_access.schema_discovery import (
            discover_training_tables,
        )

        # Discover tables suitable for ML training
        discovered = discover_training_tables(
            min_rows=self.config.discovery_min_rows,
            require_time_series=True,
        )

        xsight_tables = discovered.get("xsight", [])
        mc_tables = discovered.get("mobicontrol", [])

        logger.info(f"Discovered {len(xsight_tables)} XSight tables, {len(mc_tables)} MC tables")

        # For now, use extended loading with discovered tables
        # Future: Dynamic loader based on discovered schema
        from device_anomaly.data_access.unified_loader import load_extended_device_dataset

        df = load_extended_device_dataset(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            row_limit=self.config.row_limit,
            include_location=True,
            include_events=True,
            include_system_health=True,
            include_wifi=True,
        )

        return df

    def _merge_hourly_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge hourly granularity data with base dataset."""
        from device_anomaly.data_access.unified_loader import load_hourly_device_dataset

        logger.info(f"Loading hourly data from tables: {self.config.hourly_tables}")

        try:
            df_hourly = load_hourly_device_dataset(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                tables=self.config.hourly_tables,
                aggregation_level=self.config.hourly_aggregation,
                max_days=self.config.hourly_max_days,
            )

            if df_hourly.empty:
                logger.warning("No hourly data loaded")
                return df

            # Merge on DeviceId (and date if aggregated to device_day)
            if self.config.hourly_aggregation == "device_day":
                merge_cols = ["DeviceId"]
                if "CollectedDate" in df.columns and "CollectedDate" in df_hourly.columns:
                    merge_cols.append("CollectedDate")
                elif "Timestamp" in df.columns:
                    # Normalize timestamp to date for merge
                    df["_merge_date"] = pd.to_datetime(df["Timestamp"]).dt.date
                    if "CollectedDate" in df_hourly.columns:
                        df_hourly["_merge_date"] = pd.to_datetime(
                            df_hourly["CollectedDate"]
                        ).dt.date
                    merge_cols = ["DeviceId", "_merge_date"]

                df = df.merge(df_hourly, on=merge_cols, how="left", suffixes=("", "_hourly"))

                # Clean up temp column
                if "_merge_date" in df.columns:
                    df = df.drop(columns=["_merge_date"])

            else:
                # For hourly aggregation, just merge on DeviceId (aggregate hourly to device level)
                df_hourly_agg = df_hourly.groupby("DeviceId").agg("mean").reset_index()
                df = df.merge(df_hourly_agg, on="DeviceId", how="left", suffixes=("", "_hourly"))

            logger.info(f"Merged hourly data: {len(df.columns)} columns after merge")

        except Exception as e:
            logger.warning(f"Failed to load hourly data: {e}")

        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering: rolling windows, rate calculations, etc.

        Supports multiple feature engineering modes:
        - Standard: Core device features (rolling stats, derived metrics)
        - Extended: Includes location, events, system health, temporal features
        - Hourly: Adds hourly rolling windows for fine-grained detection

        Args:
            df: Raw telemetry DataFrame

        Returns:
            DataFrame with engineered features
        """
        logger.info("Applying feature engineering...")
        logger.info(
            f"Mode: extended={self.config.use_extended_features}, "
            f"hourly_windows={self.config.hourly_windows if self.config.include_hourly_data else 'disabled'}"
        )

        from device_anomaly.features.device_features import (
            DeviceFeatureBuilder,
            build_extended_features,
            compute_feature_norms,
        )

        if self.config.use_extended_features:
            # Extended features: all feature modules (location, events, health, temporal)
            logger.info("Using extended feature engineering (all modules)")
            df_features = build_extended_features(
                df,
                include_location=self.config.include_location_features,
                include_events=self.config.include_event_features,
                include_system_health=self.config.include_system_health_features,
                include_temporal=True,
                window=self.config.validation_days * 2,  # Use 2x validation days
            )
            # Create builder for feature spec (even though we used build_extended_features)
            builder = DeviceFeatureBuilder(
                compute_cohort=False,
                hourly_windows=self.config.hourly_windows
                if self.config.include_hourly_data
                else None,
            )
            self._feature_spec = builder.get_feature_spec()
            self._feature_spec["extended"] = True
            self._feature_spec["modules"] = {
                "location": self.config.include_location_features,
                "events": self.config.include_event_features,
                "system_health": self.config.include_system_health_features,
            }
        else:
            # Standard features: core device features only
            builder = DeviceFeatureBuilder(
                compute_cohort=False,
                hourly_windows=self.config.hourly_windows
                if self.config.include_hourly_data
                else None,
            )
            self._feature_spec = builder.get_feature_spec()
            df_features = builder.build_features(df)

        # Drop all-NaN columns
        df_features = drop_all_nan_columns(df_features, "feature_engineering")
        self._feature_norms = compute_feature_norms(df_features)

        logger.info(f"Feature engineering complete: {len(df_features.columns)} columns")
        self.tracker.advance(PipelineStage.FEATURES)
        return df_features

    def apply_feature_selection(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Apply feature selection to reduce dimensionality for expanded features.

        Only applies when enable_feature_selection=True in config. Removes:
        - Near-zero variance features
        - Highly correlated feature pairs
        - Limits to max_features by variance ranking

        Args:
            df: Feature-engineered DataFrame

        Returns:
            Tuple of (filtered DataFrame, list of selected feature names)
        """
        if not self.config.enable_feature_selection:
            # Return all numeric features as selected
            exclude = {"DeviceId", "Timestamp", "CollectedDate"}
            selected = [
                c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)
            ]
            return df, selected

        from device_anomaly.features.device_features import select_features_for_training

        logger.info(
            f"Applying feature selection (max={self.config.max_features}, "
            f"var_threshold={self.config.variance_threshold}, "
            f"corr_threshold={self.config.correlation_threshold})"
        )

        df_selected, selected_features = select_features_for_training(
            df,
            max_features=self.config.max_features,
            variance_threshold=self.config.variance_threshold,
            correlation_threshold=self.config.correlation_threshold,
        )

        logger.info(f"Feature selection: {len(df.columns)} -> {len(df_selected.columns)} columns")
        return df_selected, selected_features

    def compute_cohort_stats(self, df: pd.DataFrame) -> CohortStatsStore | None:
        """Compute and store cohort statistics from training-only data."""
        payload = compute_cohort_stats(df=df)
        if not payload.get("stats") and not payload.get("global"):
            logger.warning("No cohort stats computed; skipping cohort z-scores.")
            return None
        self._cohort_stats_payload = payload
        return CohortStatsStore(payload)

    def compute_baselines(self, df: pd.DataFrame) -> dict[str, Any]:
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
            "DeviceId",
            "ModelId",
            "ManufacturerId",
            "OsVersionId",
            "CollectedDate",
            "Timestamp",
            "is_injected_anomaly",
        }
        feature_cols = [
            col
            for col in df.columns
            if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
            and col not in exclude_cols
        ]

        timestamp_col = self._resolve_timestamp_column(df) or self.config.timestamp_col
        baselines = compute_data_driven_baselines(
            df=df,
            feature_cols=feature_cols,
            timestamp_col=timestamp_col,
            device_type_col=self.config.device_type_col,
            include_temporal=True,
            min_samples=25,
        )

        logger.info(f"Computed baselines for {len(baselines)} metrics")
        self._baselines = baselines
        self.tracker.advance(PipelineStage.BASELINES)
        return baselines

    def train_validation_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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

        ts_col = self._resolve_timestamp_column(df)
        if ts_col is None:
            # If no timestamp, use random split as fallback
            logger.warning("No timestamp column found. Using random 80/20 split.")
            rng = np.random.RandomState(self.config.random_state)
            mask = rng.rand(len(df)) < 0.8
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

    def _resolve_timestamp_column(self, df: pd.DataFrame) -> str | None:
        candidates = [
            self.config.timestamp_col,
            "Timestamp",
            "timestamp",
            "CollectedDate",
            "Date",
            "date",
        ]
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def get_model_version(self) -> str:
        if not self.model_version:
            self.model_version = datetime.now(UTC).strftime("v%Y%m%d_%H%M%S")
        return self.model_version

    def train_model(self, train_df: pd.DataFrame) -> Any:
        """
        Train the anomaly detector (IsolationForest or Ensemble).

        Args:
            train_df: Training DataFrame

        Returns:
            Trained detector instance (IsolationForest or EnsembleAnomalyDetector)
        """
        if self.config.use_ensemble:
            return self._train_ensemble_model(train_df)
        else:
            return self._train_isolation_forest_model(train_df)

    def _train_isolation_forest_model(
        self, train_df: pd.DataFrame
    ) -> AnomalyDetectorIsolationForest:
        """Train IsolationForest anomaly detector."""
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

    def _train_ensemble_model(self, train_df: pd.DataFrame) -> Any:
        """Train ensemble anomaly detector."""
        logger.info("Training Ensemble anomaly detector...")

        try:
            from device_anomaly.models.ensemble_detector import (
                EnsembleAnomalyDetector,
                EnsembleConfig,
            )

            # Configure ensemble
            config = EnsembleConfig(
                contamination=self.config.ensemble_contamination,
                enable_if=True,
                enable_lof=True,
                enable_ocsvm=True,
                if_n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
            )

            # Set weights if provided
            if self.config.ensemble_weights:
                config.weight_if = self.config.ensemble_weights.get("isolation_forest", 0.5)
                config.weight_lof = self.config.ensemble_weights.get("lof", 0.3)
                config.weight_ocsvm = self.config.ensemble_weights.get("ocsvm", 0.2)

            detector = EnsembleAnomalyDetector(config=config)
            detector.fit(train_df)

            logger.info(f"Ensemble model trained on {len(detector.feature_cols)} features")
            logger.info(
                f"Ensemble algorithms: IF={config.enable_if}, LOF={config.enable_lof}, OCSVM={config.enable_ocsvm}"
            )
            self.tracker.advance(PipelineStage.TRAINING)
            return detector

        except ImportError:
            logger.warning("Ensemble detector not available, falling back to IsolationForest")
            return self._train_isolation_forest_model(train_df)

    def evaluate_model(
        self,
        detector: Any,
        val_df: pd.DataFrame,
    ) -> TrainingMetrics:
        """
        Evaluate model on validation data and compute metrics.

        Args:
            detector: Trained detector (IsolationForest or EnsembleAnomalyDetector)
            val_df: Validation DataFrame

        Returns:
            TrainingMetrics with validation results
        """
        logger.info("Evaluating model on validation set...")

        # Score validation data
        val_scored = detector.score_dataframe(val_df)

        # Calculate anomaly rates
        train_scored = detector.score_dataframe(self._df_train)

        # Handle both single-model and ensemble score columns
        score_col = "ensemble_score" if "ensemble_score" in val_scored.columns else "anomaly_score"
        label_col = "ensemble_label" if "ensemble_label" in val_scored.columns else "anomaly_label"

        train_anomaly_rate = (train_scored[label_col] == -1).mean()
        val_anomaly_rate = (val_scored[label_col] == -1).mean()

        # Calculate feature importance
        feature_importance = self._estimate_feature_importance(detector, self._df_train)

        metrics = TrainingMetrics(
            train_rows=len(self._df_train),
            validation_rows=len(val_df),
            feature_count=len(detector.feature_cols),
            anomaly_rate_train=float(train_anomaly_rate),
            anomaly_rate_validation=float(val_anomaly_rate),
            feature_importance=feature_importance,
        )

        # If using ensemble, capture per-algorithm metrics
        if self.config.use_ensemble:
            metrics.ensemble_algorithm_scores = self._get_ensemble_algorithm_scores(
                detector, val_scored
            )

        # If we have ground truth labels, compute AUC
        if "is_injected_anomaly" in val_df.columns:
            y_true = val_df["is_injected_anomaly"].values
            y_scores = -val_scored[score_col].values  # Flip sign for AUC

            try:
                metrics.validation_auc = float(roc_auc_score(y_true, y_scores))
                logger.info(f"Validation AUC: {metrics.validation_auc:.4f}")

                # Precision at 80% recall
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                idx = np.argmin(np.abs(recall - 0.8))
                metrics.precision_at_recall_80 = float(precision[idx])
            except Exception as e:
                logger.warning(f"Could not compute AUC metrics: {e}")

        # Compute enhanced metrics
        if self.config.compute_score_stability:
            metrics.score_stability = self._compute_score_stability(
                detector, val_df, val_scored, score_col
            )

        if self.config.compute_cohort_fairness:
            metrics.cohort_fairness = self._compute_cohort_fairness(val_scored, label_col)

        # Compute temporal stability if we have timestamps
        ts_col = self._resolve_timestamp_column(val_df)
        if ts_col:
            metrics.temporal_stability = self._compute_temporal_stability(
                val_scored, ts_col, label_col
            )

        logger.info(f"Train anomaly rate: {train_anomaly_rate:.2%}")
        logger.info(f"Validation anomaly rate: {val_anomaly_rate:.2%}")

        self.tracker.advance(PipelineStage.SCORING)
        return metrics

    def _get_ensemble_algorithm_scores(
        self,
        detector: Any,
        scored_df: pd.DataFrame,
    ) -> dict[str, float]:
        """Get per-algorithm anomaly rates from ensemble."""
        algorithm_scores = {}

        if "if_score" in scored_df.columns:
            algorithm_scores["isolation_forest_mean_score"] = float(scored_df["if_score"].mean())
        if "lof_score" in scored_df.columns:
            algorithm_scores["lof_mean_score"] = float(scored_df["lof_score"].mean())
        if "ocsvm_score" in scored_df.columns:
            algorithm_scores["ocsvm_mean_score"] = float(scored_df["ocsvm_score"].mean())

        return algorithm_scores

    def _compute_score_stability(
        self,
        detector: Any,
        val_df: pd.DataFrame,
        val_scored: pd.DataFrame,
        score_col: str,
    ) -> dict[str, float]:
        """
        Compute score stability using bootstrap variance.

        Measures how stable the anomaly scores are across different subsamples.
        """
        try:
            n_bootstrap = 10
            bootstrap_rates = []

            for _i in range(n_bootstrap):
                sample_idx = np.random.choice(len(val_df), size=len(val_df), replace=True)
                sample_df = val_df.iloc[sample_idx]
                sample_scored = detector.score_dataframe(sample_df)

                label_col = (
                    "ensemble_label"
                    if "ensemble_label" in sample_scored.columns
                    else "anomaly_label"
                )
                rate = (sample_scored[label_col] == -1).mean()
                bootstrap_rates.append(rate)

            return {
                "bootstrap_mean": float(np.mean(bootstrap_rates)),
                "bootstrap_std": float(np.std(bootstrap_rates)),
                "bootstrap_cv": float(np.std(bootstrap_rates) / (np.mean(bootstrap_rates) + 1e-6)),
            }
        except Exception as e:
            logger.warning(f"Could not compute score stability: {e}")
            return {}

    def _compute_cohort_fairness(
        self,
        scored_df: pd.DataFrame,
        label_col: str,
    ) -> dict[str, Any]:
        """
        Compute cohort fairness metrics.

        Measures if anomaly detection rate is consistent across device segments.
        """
        try:
            from device_anomaly.features.cohort_stats import compute_cohort_fairness

            return compute_cohort_fairness(scored_df, anomaly_col=label_col)
        except Exception as e:
            logger.warning(f"Could not compute cohort fairness: {e}")
            return {}

    def _compute_temporal_stability(
        self,
        scored_df: pd.DataFrame,
        ts_col: str,
        label_col: str,
    ) -> dict[str, float]:
        """
        Compute temporal stability of anomaly detection.

        Measures if anomaly rate is consistent across time periods.
        """
        try:
            scored_df = scored_df.copy()
            scored_df[ts_col] = pd.to_datetime(scored_df[ts_col])

            # Group by date and compute daily anomaly rate
            scored_df["_date"] = scored_df[ts_col].dt.date
            daily_rates = scored_df.groupby("_date").apply(lambda x: (x[label_col] == -1).mean())

            if len(daily_rates) < 2:
                return {}

            return {
                "daily_rate_mean": float(daily_rates.mean()),
                "daily_rate_std": float(daily_rates.std()),
                "daily_rate_cv": float(daily_rates.std() / (daily_rates.mean() + 1e-6)),
                "daily_rate_min": float(daily_rates.min()),
                "daily_rate_max": float(daily_rates.max()),
            }
        except Exception as e:
            logger.warning(f"Could not compute temporal stability: {e}")
            return {}

    def _estimate_feature_importance(
        self,
        detector: AnomalyDetectorIsolationForest,
        df: pd.DataFrame,
    ) -> dict[str, float]:
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
        model_version: str | None = None,
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

        version = model_version or self.get_model_version()

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

        # 2b. Save cohort statistics
        if self._cohort_stats_payload:
            cohort_stats_path = output_dir / "cohort_stats.json"
            save_cohort_stats(self._cohort_stats_payload, cohort_stats_path)
            artifacts.cohort_stats_path = cohort_stats_path

        # 2c. Train and save cross-device pattern detector
        try:
            from device_anomaly.models.cohort_detector import CrossDevicePatternDetector

            # Use training data for cohort detector
            df_train = getattr(self, "_df_train", None)
            if df_train is not None and not df_train.empty:
                logger.info("Training cross-device pattern detector...")
                cohort_detector = CrossDevicePatternDetector(
                    min_sample_size=20,
                    z_threshold=2.0,
                )
                cohort_detector.fit(df_train)

                cohort_detector_path = output_dir / "cohort_detector.pkl"
                cohort_detector.save(cohort_detector_path)
                artifacts.cohort_detector_path = cohort_detector_path
                logger.info(f"Saved cohort detector to {cohort_detector_path}")
        except Exception as e:
            logger.warning(f"Failed to train cohort detector: {e}")

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
            "detector_config": asdict(detector.config),
            "metrics": metrics.to_dict(),
            "feature_cols": detector.feature_cols,
            "feature_spec": self._feature_spec,
            "feature_norms": self._feature_norms,
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

        def _parse_dt(value: str | None) -> datetime | None:
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

        Full pipeline supports:
        - Standard training with 5 core tables
        - Extended features (location, events, health, temporal)
        - Hourly data integration
        - Auto-discovery of high-value tables
        - Feature selection for high dimensionality

        Args:
            output_dir: Directory for model artifacts

        Returns:
            TrainingResult with complete run information
        """
        started_at = datetime.now(UTC).isoformat()
        logger.info(f"=== Starting Training Pipeline (run_id: {self.run_id}) ===")
        logger.info(
            f"Config: extended={self.config.use_extended_features}, "
            f"hourly={self.config.include_hourly_data}, "
            f"discovery={self.config.use_auto_discovery}, "
            f"feature_selection={self.config.enable_feature_selection}"
        )

        try:
            model_version = self.get_model_version()
            # 1. Load data (supports extended, hourly, discovery modes)
            df = self.load_training_data()

            # 2. Feature engineering (supports extended modules)
            df_features = self.prepare_features(df)

            # 2.5 Feature selection (for expanded feature sets)
            selected_features: list[str] = []
            if (
                self.config.enable_feature_selection
                and len(df_features.columns) > self.config.max_features
            ):
                logger.info(
                    f"Applying feature selection: {len(df_features.columns)} cols -> max {self.config.max_features}"
                )
                df_features, selected_features = self.apply_feature_selection(df_features)
                logger.info(f"Selected {len(selected_features)} features after selection")
            else:
                # Track all numeric features as selected
                exclude = {"DeviceId", "Timestamp", "CollectedDate"}
                selected_features = [
                    c
                    for c in df_features.columns
                    if c not in exclude and np.issubdtype(df_features[c].dtype, np.number)
                ]

            # Store selected features in spec
            self._feature_spec["selected_features"] = selected_features
            self._feature_spec["original_feature_count"] = len(df_features.columns)

            # 3. Train/validation split
            df_train, df_val = self.train_validation_split(df_features)

            # 4. Compute cohort stats on train-only split, then apply to train/val
            cohort_stats = self.compute_cohort_stats(df_train)
            if cohort_stats is not None:
                df_train = apply_cohort_stats(df_train, cohort_stats)
                df_val = apply_cohort_stats(df_val, cohort_stats)
                self._df_train = df_train
                self._df_val = df_val

            # 5. Compute baselines (train-only to avoid leakage)
            self.compute_baselines(df_train)

            # 6. Train model
            detector = self.train_model(df_train)

            # 7. Evaluate
            metrics = self.evaluate_model(detector, df_val)

            # 8. Export artifacts
            artifacts = self.export_artifacts(
                detector, metrics, output_dir, model_version=model_version
            )

            completed_at = datetime.now(UTC).isoformat()
            result = TrainingResult(
                run_id=self.run_id,
                model_version=model_version,
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
            completed_at = datetime.now(UTC).isoformat()

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


def train_ultimate_model(
    start_date: str,
    end_date: str,
    validation_days: int = 7,
    output_dir: str = "models/ultimate",
    export_onnx: bool = True,
    contamination: float = 0.03,
    n_estimators: int = 500,
    use_multi_source: bool = False,
    row_limit_per_source: int = 1_000_000,
) -> TrainingResult:
    """
    Train the ultimate anomaly detection model with ALL available data sources.

    This function enables the full power of the anomaly detection system:
    - 200M+ rows of telemetry across 27+ tables
    - 300+ features including location, temporal, network, security
    - Hourly granularity for fine-grained detection
    - Multi-model ensemble (Isolation Forest + LOF + OCSVM)
    - SHAP explanations for interpretability

    Data Sources Activated:
    - MobiControl: DeviceStatInt (764K), DeviceStatLocation (619K),
                   DeviceStatNetTraffic (244K), MainLog (1M+), Alert (1.3K)
    - XSight: cs_DataUsageByHour (104M), cs_BatteryLevelDrop (14.8M),
              cs_WiFiLocation (790K), cs_WifiHour (755K), cs_LastKnown (674K)

    Feature Categories:
    - Core device metrics (battery, storage, app usage, connectivity)
    - Location intelligence (mobility, dead zones, WiFi patterns)
    - Temporal patterns (hourly entropy, peak hours, seasonality)
    - Network traffic (per-app usage, exfiltration risk, interface diversity)
    - Security posture (encryption, root detection, patch age)
    - System health (CPU, RAM, thermal events)
    - Cross-device patterns (cohort baselines)

    Args:
        start_date: Start date for training data (YYYY-MM-DD)
        end_date: End date for training data (YYYY-MM-DD)
        validation_days: Days to hold out for validation
        output_dir: Directory for model artifacts
        export_onnx: Whether to export ONNX model
        contamination: Expected anomaly rate (0-1)
        n_estimators: Number of trees in IsolationForest (increased for robustness)
        use_multi_source: If True, train on data from all configured sources
        row_limit_per_source: Max rows per source

    Returns:
        TrainingResult with complete run information

    Example:
        # Train the ultimate model with all features
        result = train_ultimate_model("2024-01-01", "2024-12-31")

        # Check metrics
        print(f"Features: {result.metrics.feature_count}")
        print(f"Train rows: {result.metrics.train_rows:,}")
        print(f"Anomaly rate: {result.metrics.anomaly_rate_validation:.2%}")
    """
    logger.info("=" * 60)
    logger.info("ULTIMATE ANOMALY DETECTION MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info("Enabled features:")
    logger.info("  - Extended data sources (MC timeseries + XSight extended)")
    logger.info("  - Hourly granularity (cs_DataUsageByHour, cs_BatteryLevelDrop)")
    logger.info("  - Location intelligence (GPS, WiFi patterns)")
    logger.info("  - Network traffic analysis (per-app, exfiltration detection)")
    logger.info("  - Security posture scoring")
    logger.info("  - Temporal pattern decomposition (STL)")
    logger.info("  - Ensemble detection (IF + LOF + OCSVM)")
    logger.info("=" * 60)

    config = TrainingConfig(
        start_date=start_date,
        end_date=end_date,
        validation_days=validation_days,
        contamination=contamination,
        n_estimators=n_estimators,
        export_onnx=export_onnx,
        use_multi_source=use_multi_source,
        row_limit_per_source=row_limit_per_source,
        # Enable extended features (ALL data sources)
        use_extended_features=True,
        include_location_features=True,
        include_event_features=True,
        include_system_health_features=True,
        include_wifi_features=True,
        # Enable hourly data (cs_DataUsageByHour, etc.)
        include_hourly_data=True,
        hourly_tables=["cs_DataUsageByHour", "cs_BatteryLevelDrop", "cs_WifiHour"],
        hourly_aggregation="device_day",
        hourly_max_days=7,
        # Enable ensemble detector
        use_ensemble=True,
        ensemble_contamination=contamination,
        ensemble_weights={"isolation_forest": 0.5, "lof": 0.3, "ocsvm": 0.2},
        # Feature selection for high dimensionality
        enable_feature_selection=True,
        max_features=300,
        variance_threshold=0.005,  # Lower threshold to keep more features
        correlation_threshold=0.98,  # Higher threshold to reduce removal
        # Enhanced evaluation
        compute_cohort_fairness=True,
        compute_score_stability=True,
    )

    pipeline = RealDataTrainingPipeline(config)
    result = pipeline.run(output_dir=output_dir)

    logger.info("=" * 60)
    logger.info("ULTIMATE MODEL TRAINING COMPLETE")
    logger.info(f"Status: {result.status}")
    logger.info(f"Model version: {result.model_version}")
    logger.info(f"Features used: {result.metrics.feature_count}")
    logger.info(f"Training rows: {result.metrics.train_rows:,}")
    logger.info(f"Validation rows: {result.metrics.validation_rows:,}")
    logger.info(f"Train anomaly rate: {result.metrics.anomaly_rate_train:.2%}")
    logger.info(f"Validation anomaly rate: {result.metrics.anomaly_rate_validation:.2%}")
    if result.metrics.validation_auc:
        logger.info(f"Validation AUC: {result.metrics.validation_auc:.4f}")
    logger.info("=" * 60)

    return result
