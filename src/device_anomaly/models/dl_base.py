"""Deep Learning base classes for anomaly detection.

This module provides abstract base classes and configuration dataclasses
for PyTorch-based anomaly detectors. All DL detectors inherit from
BaseDLDetector which provides a consistent interface matching the
existing sklearn-based detectors.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Type hints for PyTorch (imported at runtime to avoid hard dependency)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


@dataclass
class DLDetectorConfig:
    """Configuration for Deep Learning anomaly detectors.

    Attributes:
        name: Unique identifier for this detector instance
        latent_dim: Dimension of the latent space (for autoencoders)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate for regularization
        learning_rate: Initial learning rate for optimizer
        weight_decay: L2 regularization weight
        epochs: Maximum number of training epochs
        batch_size: Training batch size
        early_stopping_patience: Epochs to wait before early stopping
        min_delta: Minimum improvement to reset patience
        contamination: Expected proportion of anomalies (for threshold)
        use_gpu: Whether to use GPU if available
        random_state: Random seed for reproducibility
        reconstruction_loss: Loss function type ('mse', 'mae', 'huber')
        kl_weight: Weight for KL divergence in VAE (beta parameter)
        gradient_clip_val: Max gradient norm for clipping (None to disable)
        validation_split: Fraction of training data for validation
        scale_features: Whether to apply StandardScaler to features
    """
    name: str = "dl_detector"
    latent_dim: int = 32
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    batch_size: int = 256
    early_stopping_patience: int = 10
    min_delta: float = 1e-4
    contamination: float = 0.05
    use_gpu: bool = False  # CPU by default
    random_state: int = 42
    reconstruction_loss: str = "mse"
    kl_weight: float = 1e-3
    gradient_clip_val: float | None = 1.0
    validation_split: float = 0.1
    scale_features: bool = True

    # Severity thresholds (reconstruction error based)
    severity_thresholds: dict[str, float] = field(default_factory=lambda: {
        "critical": 3.0,  # >3 std above mean error
        "high": 2.0,
        "medium": 1.5,
        "low": 1.0,
    })
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DLTrainingMetrics:
    """Metrics collected during DL model training.

    Attributes:
        train_loss_history: Loss values per epoch
        val_loss_history: Validation loss per epoch
        best_epoch: Epoch with best validation loss
        best_val_loss: Best validation loss achieved
        final_train_loss: Final training loss
        training_time_seconds: Total training time
        early_stopped: Whether training stopped early
        reconstruction_error_mean: Mean reconstruction error on training data
        reconstruction_error_std: Std of reconstruction error
        reconstruction_error_percentiles: Percentile values for thresholding
    """
    train_loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    final_train_loss: float = 0.0
    training_time_seconds: float = 0.0
    early_stopped: bool = False
    reconstruction_error_mean: float = 0.0
    reconstruction_error_std: float = 0.0
    reconstruction_error_percentiles: dict[str, float] = field(default_factory=dict)


class BaseDLDetector(ABC):
    """Abstract base class for PyTorch-based anomaly detectors.

    All DL anomaly detectors must implement these core methods to ensure
    a consistent interface compatible with the existing sklearn detectors.

    The interface mirrors BaseAnomalyDetector and AnomalyDetectorIsolationForest
    to allow seamless substitution in the training and inference pipelines.

    Example:
        class VAEDetector(BaseDLDetector):
            def _build_model(self, input_dim: int) -> nn.Module:
                return VAE(input_dim, self.config.hidden_dims, self.config.latent_dim)

            def _compute_anomaly_score(self, x: np.ndarray) -> np.ndarray:
                recon = self.model(torch.tensor(x))
                return ((x - recon.numpy()) ** 2).mean(axis=1)
    """

    def __init__(self, config: DLDetectorConfig | None = None):
        """Initialize the DL detector with configuration.

        Args:
            config: DLDetectorConfig with detector settings
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DL detectors. "
                "Install with: pip install torch"
            )

        self.config = config or DLDetectorConfig()
        self._is_fitted = False
        self._feature_cols: list[str] = []
        self._fit_timestamp: datetime | None = None
        self._training_metrics: DLTrainingMetrics | None = None

        # Model and preprocessing state
        self.model: nn.Module | None = None
        self.device: torch.device = self._select_device()
        self.scaler: Any | None = None  # StandardScaler
        self.impute_values: pd.Series | None = None

        # Anomaly detection thresholds (learned from training data)
        self._threshold: float = 0.0
        self._error_mean: float = 0.0
        self._error_std: float = 1.0

        # Set random seed for reproducibility
        self._set_random_seed()

    def _select_device(self) -> torch.device:
        """Select compute device (CPU or GPU)."""
        if self.config.use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _set_random_seed(self) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_state)

    @property
    def is_fitted(self) -> bool:
        """Check if the detector has been trained."""
        return self._is_fitted

    @property
    def feature_cols(self) -> list[str]:
        """Get the feature columns used for detection."""
        return self._feature_cols

    @property
    def name(self) -> str:
        """Get the detector name."""
        return self.config.name

    @property
    def training_metrics(self) -> DLTrainingMetrics | None:
        """Get training metrics if available."""
        return self._training_metrics

    @abstractmethod
    def _build_model(self, input_dim: int) -> nn.Module:
        """Build the PyTorch model architecture.

        Args:
            input_dim: Number of input features

        Returns:
            PyTorch nn.Module implementing the model
        """
        pass

    @abstractmethod
    def _compute_reconstruction_error(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction error for input batch.

        Args:
            x: Input tensor of shape (batch_size, n_features)

        Returns:
            Tensor of reconstruction errors, shape (batch_size,)
        """
        pass

    @abstractmethod
    def _compute_loss(
        self,
        x: torch.Tensor,
        model_output: Any,
    ) -> torch.Tensor:
        """Compute training loss for a batch.

        Args:
            x: Input tensor
            model_output: Output from model forward pass

        Returns:
            Scalar loss tensor
        """
        pass

    def _select_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Select numeric feature columns for training.

        Strategy mirrors AnomalyDetectorIsolationForest:
        - Prefer cohort-normalized features (*_cohort_z)
        - Fall back to all numeric columns
        - Exclude IDs, labels, and targets
        """
        import pandas.api.types as ptypes

        numeric_cols = [
            c for c in df.columns
            if ptypes.is_numeric_dtype(df[c])
        ]

        exclude = {
            "DeviceId", "ModelId", "ManufacturerId", "OsVersionId",
            "is_injected_anomaly", "anomaly_score", "anomaly_label",
        }

        candidates = [c for c in numeric_cols if c not in exclude]

        # Prefer cohort-normalized columns
        cohort_cols = [c for c in candidates if c.endswith("_cohort_z")]
        baseline_cols = [c for c in candidates if "_z_" in c]

        feature_cols = cohort_cols or baseline_cols or candidates

        if not feature_cols:
            raise ValueError("No feature columns found to train on.")

        return feature_cols

    def _prepare_training_data(
        self,
        df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training data with train/validation split.

        Args:
            df: Input DataFrame with features

        Returns:
            Tuple of (train_array, val_array)
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        self._feature_cols = self._select_feature_columns(df)

        feature_df = df[self._feature_cols].copy()
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Compute imputation values
        medians = feature_df.median().fillna(0.0)
        self.impute_values = medians
        feature_df = feature_df.fillna(self.impute_values)

        # Remove near-constant features
        variances = feature_df.var(ddof=0).fillna(0.0)
        keep_mask = variances > 1e-6
        kept_cols = list(variances[keep_mask].index)

        if not kept_cols:
            raise ValueError("All feature columns have near-zero variance.")

        self._feature_cols = kept_cols
        self.impute_values = self.impute_values[kept_cols]
        feature_df = feature_df[kept_cols]

        # Scale features
        if self.config.scale_features:
            self.scaler = StandardScaler()
            data = self.scaler.fit_transform(feature_df.values)
        else:
            data = feature_df.values.astype(np.float32)

        # Train/validation split
        if self.config.validation_split > 0:
            train_data, val_data = train_test_split(
                data,
                test_size=self.config.validation_split,
                random_state=self.config.random_state,
            )
        else:
            train_data = data
            val_data = data[:min(100, len(data))]  # Small validation set

        return train_data.astype(np.float32), val_data.astype(np.float32)

    def _prepare_inference_data(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare data for inference.

        Args:
            df: Input DataFrame

        Returns:
            Numpy array ready for model inference
        """
        if not self._feature_cols:
            raise RuntimeError("Model has not been fit yet.")

        missing = [c for c in self._feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        feature_df = df[self._feature_cols].copy()
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        if self.impute_values is None:
            raise RuntimeError("Imputation values missing; has the model been fit?")

        feature_df = feature_df.fillna(self.impute_values)

        if self.config.scale_features and self.scaler is not None:
            data = self.scaler.transform(feature_df.values)
        else:
            data = feature_df.values

        return data.astype(np.float32)

    def fit(self, df: pd.DataFrame) -> DLTrainingMetrics:
        """Train the detector on normal data.

        Args:
            df: DataFrame with training data (assumed mostly normal)

        Returns:
            DLTrainingMetrics with training statistics
        """
        import logging
        import time

        logger = logging.getLogger(__name__)
        start_time = time.time()

        # Prepare data
        train_data, val_data = self._prepare_training_data(df)
        input_dim = train_data.shape[1]

        logger.info(
            "Training %s on %d samples with %d features",
            self.__class__.__name__,
            len(train_data),
            input_dim,
        )

        # Build model
        self.model = self._build_model(input_dim)
        self.model = self.model.to(self.device)

        # Create data loaders
        train_loader = self._create_dataloader(train_data, shuffle=True)
        val_loader = self._create_dataloader(val_data, shuffle=False)

        # Training loop
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        metrics = DLTrainingMetrics()
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(self.config.epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer)
            metrics.train_loss_history.append(train_loss)

            # Validation phase
            val_loss = self._validate_epoch(val_loader)
            metrics.val_loss_history.append(val_loss)

            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                metrics.best_epoch = epoch
                metrics.best_val_loss = val_loss
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                logger.debug(
                    "Epoch %d: train_loss=%.4f, val_loss=%.4f, lr=%.6f",
                    epoch, train_loss, val_loss, optimizer.param_groups[0]['lr']
                )

            if patience_counter >= self.config.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch)
                metrics.early_stopped = True
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        metrics.final_train_loss = metrics.train_loss_history[-1]
        metrics.training_time_seconds = time.time() - start_time

        # Compute reconstruction error statistics for thresholding
        self._compute_threshold_statistics(train_data, metrics)

        self._is_fitted = True
        self._fit_timestamp = datetime.now(UTC)
        self._training_metrics = metrics

        logger.info(
            "Training complete: best_epoch=%d, best_val_loss=%.4f, time=%.1fs",
            metrics.best_epoch,
            metrics.best_val_loss,
            metrics.training_time_seconds,
        )

        return metrics

    def _create_dataloader(
        self,
        data: np.ndarray,
        shuffle: bool = False
    ) -> torch.utils.data.DataLoader:
        """Create a PyTorch DataLoader from numpy array."""
        tensor = torch.tensor(data, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(tensor)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Keep simple for CPU training
            pin_memory=False,
        )

    def _train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for (batch,) in train_loader:
            batch = batch.to(self.device)

            optimizer.zero_grad()
            output = self.model(batch)
            loss = self._compute_loss(batch, output)
            loss.backward()

            if self.config.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val
                )

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _validate_epoch(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(self.device)
                output = self.model(batch)
                loss = self._compute_loss(batch, output)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _compute_threshold_statistics(
        self,
        train_data: np.ndarray,
        metrics: DLTrainingMetrics,
    ) -> None:
        """Compute reconstruction error statistics for anomaly thresholding."""
        self.model.eval()

        loader = self._create_dataloader(train_data, shuffle=False)
        all_errors = []

        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                errors = self._compute_reconstruction_error(batch)
                all_errors.append(errors.cpu().numpy())

        errors = np.concatenate(all_errors)

        self._error_mean = float(np.mean(errors))
        self._error_std = float(np.std(errors))

        # Set threshold based on contamination (percentile-based)
        threshold_percentile = 100 * (1 - self.config.contamination)
        self._threshold = float(np.percentile(errors, threshold_percentile))

        # Store percentiles in metrics
        metrics.reconstruction_error_mean = self._error_mean
        metrics.reconstruction_error_std = self._error_std
        metrics.reconstruction_error_percentiles = {
            "p50": float(np.percentile(errors, 50)),
            "p90": float(np.percentile(errors, 90)),
            "p95": float(np.percentile(errors, 95)),
            "p99": float(np.percentile(errors, 99)),
            "threshold": self._threshold,
        }

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Compute anomaly scores for data points.

        Higher scores indicate more anomalous behavior (reconstruction error).
        This is inverted from sklearn's convention where lower = more anomalous.

        Args:
            df: DataFrame with data to score

        Returns:
            Array of anomaly scores (one per row)
        """
        if not self._is_fitted:
            raise RuntimeError(f"Detector '{self.name}' has not been fitted")

        data = self._prepare_inference_data(df)
        loader = self._create_dataloader(data, shuffle=False)

        self.model.eval()
        all_errors = []

        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                errors = self._compute_reconstruction_error(batch)
                all_errors.append(errors.cpu().numpy())

        return np.concatenate(all_errors)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict anomaly labels.

        Args:
            df: DataFrame with data to classify

        Returns:
            Array of labels: 1 for normal, -1 for anomaly
        """
        scores = self.score(df)
        # Higher reconstruction error = anomaly
        labels = np.where(scores > self._threshold, -1, 1)
        return labels

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score data and add results as columns.

        Args:
            df: DataFrame to score

        Returns:
            DataFrame with 'anomaly_score' and 'anomaly_label' columns
        """
        df_scored = df.copy()

        scores = self.score(df)
        labels = np.where(scores > self._threshold, -1, 1)

        # Convert to sklearn convention (lower = more anomalous)
        # by negating the scores
        df_scored["anomaly_score"] = -scores
        df_scored["anomaly_label"] = labels

        return df_scored

    def save_model(
        self,
        output_path: str | Path,
        export_onnx: bool = False
    ) -> dict[str, Path]:
        """Save trained model to disk.

        Args:
            output_path: Base path for model files
            export_onnx: If True, also export to ONNX format

        Returns:
            Dictionary with paths to saved models
        """
        import logging

        logger = logging.getLogger(__name__)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        # Save PyTorch model and detector state
        pytorch_path = output_path.with_suffix(".pt")
        state = {
            "model_state_dict": self.model.state_dict(),
            "model_class": self.model.__class__.__name__,
            "input_dim": len(self._feature_cols),
            "feature_cols": self._feature_cols,
            "impute_values": self.impute_values.to_dict() if self.impute_values is not None else None,
            "scaler": self.scaler,
            "config": self.config,
            "threshold": self._threshold,
            "error_mean": self._error_mean,
            "error_std": self._error_std,
            "training_metrics": self._training_metrics,
            "fit_timestamp": self._fit_timestamp.isoformat() if self._fit_timestamp else None,
        }
        torch.save(state, pytorch_path)
        saved_paths["pytorch"] = pytorch_path
        logger.info("Saved PyTorch model to %s", pytorch_path)

        # Export to ONNX if requested
        if export_onnx:
            try:
                from device_anomaly.models.pytorch_onnx_exporter import PyTorchONNXExporter

                onnx_path = output_path.with_suffix(".onnx")
                exporter = PyTorchONNXExporter()
                exporter.export(
                    model=self.model,
                    input_dim=len(self._feature_cols),
                    output_path=onnx_path,
                )
                saved_paths["onnx"] = onnx_path
                logger.info("Exported ONNX model to %s", onnx_path)
            except Exception as e:
                logger.warning("Failed to export ONNX model: %s", e)

        return saved_paths

    @classmethod
    def load_model(cls, model_path: str | Path) -> BaseDLDetector:
        """Load a trained model from disk.

        Args:
            model_path: Path to the saved .pt model file

        Returns:
            Loaded detector instance
        """
        import logging

        logger = logging.getLogger(__name__)
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        state = torch.load(model_path, map_location="cpu", weights_only=False)

        # Recreate instance with saved config
        instance = cls(config=state.get("config"))

        # Rebuild model architecture
        input_dim = state["input_dim"]
        instance.model = instance._build_model(input_dim)
        instance.model.load_state_dict(state["model_state_dict"])
        instance.model = instance.model.to(instance.device)
        instance.model.eval()

        # Restore state
        instance._feature_cols = state["feature_cols"]
        if state.get("impute_values"):
            instance.impute_values = pd.Series(state["impute_values"])
        instance.scaler = state.get("scaler")
        instance._threshold = state.get("threshold", 0.0)
        instance._error_mean = state.get("error_mean", 0.0)
        instance._error_std = state.get("error_std", 1.0)
        instance._training_metrics = state.get("training_metrics")
        instance._is_fitted = True

        if state.get("fit_timestamp"):
            instance._fit_timestamp = datetime.fromisoformat(state["fit_timestamp"])

        logger.info(
            "Loaded %s from %s with %d features",
            instance.__class__.__name__,
            model_path,
            len(instance._feature_cols),
        )

        return instance

    def get_metadata(self) -> dict[str, Any]:
        """Get detector metadata for observability."""
        metadata = {
            "name": self.config.name,
            "type": self.__class__.__name__,
            "is_fitted": self._is_fitted,
            "fit_timestamp": self._fit_timestamp.isoformat() if self._fit_timestamp else None,
            "feature_count": len(self._feature_cols),
            "device": str(self.device),
            "config": {
                "latent_dim": self.config.latent_dim,
                "hidden_dims": self.config.hidden_dims,
                "epochs": self.config.epochs,
                "contamination": self.config.contamination,
            },
        }

        if self._training_metrics:
            metadata["training"] = {
                "best_epoch": self._training_metrics.best_epoch,
                "best_val_loss": self._training_metrics.best_val_loss,
                "training_time_seconds": self._training_metrics.training_time_seconds,
                "early_stopped": self._training_metrics.early_stopped,
            }

        return metadata

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.config.name}', fitted={self._is_fitted})>"
