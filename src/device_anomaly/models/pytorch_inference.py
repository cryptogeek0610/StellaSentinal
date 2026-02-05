"""PyTorch inference engine for Deep Learning models.

This module provides an inference engine that implements the same interface
as the existing sklearn and ONNX engines, allowing seamless integration
with the anomaly detection pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


logger = logging.getLogger(__name__)


@dataclass
class PyTorchEngineConfig:
    """Configuration for PyTorch inference engine.

    Attributes:
        device: Device to run inference on ('cpu', 'cuda', 'cuda:0', etc.)
        use_half_precision: Use FP16 for faster inference (GPU only)
        batch_size: Batch size for inference (None for all at once)
        num_workers: Number of data loading workers
        collect_metrics: Whether to collect inference metrics
        warmup_iterations: Number of warmup iterations
    """

    device: str = "cpu"
    use_half_precision: bool = False
    batch_size: int | None = None
    num_workers: int = 0
    collect_metrics: bool = True
    warmup_iterations: int = 3


@dataclass
class PyTorchInferenceMetrics:
    """Metrics collected during PyTorch inference.

    Attributes:
        total_samples: Total samples processed
        total_time_ms: Total inference time in milliseconds
        avg_time_per_sample_ms: Average time per sample
        avg_time_per_batch_ms: Average time per batch
        throughput_samples_per_sec: Samples processed per second
        device: Device used for inference
        model_type: Type of model
    """

    total_samples: int = 0
    total_time_ms: float = 0.0
    avg_time_per_sample_ms: float = 0.0
    avg_time_per_batch_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    device: str = "cpu"
    model_type: str = ""


class PyTorchInferenceEngine:
    """Inference engine for PyTorch models.

    Provides the same interface as sklearn and ONNX engines:
    - predict(X) -> labels
    - score_samples(X) -> anomaly scores

    Example:
        # Load from saved model
        engine = PyTorchInferenceEngine.from_checkpoint("model.pt")
        scores = engine.score_samples(X)
        labels = engine.predict(X)

        # From existing model
        engine = PyTorchInferenceEngine(model, threshold=0.5)
        scores = engine.score_samples(X)
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.0,
        config: PyTorchEngineConfig | None = None,
        feature_cols: list[str] | None = None,
        scaler: Any | None = None,
        impute_values: dict[str, float] | None = None,
    ):
        """Initialize the inference engine.

        Args:
            model: PyTorch model for inference
            threshold: Anomaly threshold (samples above this are anomalies)
            config: Engine configuration
            feature_cols: Expected feature column names
            scaler: Fitted scaler for preprocessing
            impute_values: Imputation values for missing features
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self.config = config or PyTorchEngineConfig()
        self.model = model
        self.threshold = threshold
        self.feature_cols = feature_cols or []
        self.scaler = scaler
        self.impute_values = impute_values

        # Setup device
        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Half precision
        if self.config.use_half_precision and self.device.type == "cuda":
            self.model = self.model.half()

        # Metrics
        self._metrics = PyTorchInferenceMetrics(
            device=str(self.device),
            model_type=self.model.__class__.__name__,
        )
        self._batch_times: list[float] = []

        # Warmup
        if self.config.warmup_iterations > 0:
            self._warmup()

    def _warmup(self) -> None:
        """Run warmup iterations to optimize model."""
        if not self.feature_cols:
            return

        n_features = len(self.feature_cols)
        dummy = np.random.randn(10, n_features).astype(np.float32)

        for _ in range(self.config.warmup_iterations):
            self._forward(dummy)

        logger.debug("Completed %d warmup iterations", self.config.warmup_iterations)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Run forward pass and return reconstruction error.

        Args:
            X: Input array of shape (n_samples, n_features)

        Returns:
            Reconstruction error per sample
        """
        # Convert to tensor
        if self.config.use_half_precision and self.device.type == "cuda":
            x_tensor = torch.tensor(X, dtype=torch.float16, device=self.device)
        else:
            x_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # Handle VAE (returns tuple) vs Autoencoder (returns tensor)
            output = self.model(x_tensor)
            if isinstance(output, tuple):
                recon = output[0]  # reconstruction
            else:
                recon = output

            # Compute reconstruction error
            error = torch.mean((x_tensor - recon) ** 2, dim=1)

        return error.cpu().numpy().astype(np.float32)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples.

        Higher scores indicate more anomalous samples.

        Args:
            X: Input array of shape (n_samples, n_features)

        Returns:
            Array of anomaly scores
        """
        X = self._ensure_float32(X)
        n_samples = len(X)

        start_time = time.perf_counter()

        if self.config.batch_size and n_samples > self.config.batch_size:
            # Batch processing
            scores = []
            for i in range(0, n_samples, self.config.batch_size):
                batch = X[i : i + self.config.batch_size]
                batch_start = time.perf_counter()
                batch_scores = self._forward(batch)
                if self.config.collect_metrics:
                    self._batch_times.append(time.perf_counter() - batch_start)
                scores.append(batch_scores)
            result = np.concatenate(scores)
        else:
            result = self._forward(X)

        total_time = time.perf_counter() - start_time

        # Update metrics
        if self.config.collect_metrics:
            self._update_metrics(n_samples, total_time)

        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels.

        Args:
            X: Input array of shape (n_samples, n_features)

        Returns:
            Labels: 1 for normal, -1 for anomaly
        """
        scores = self.score_samples(X)
        return np.where(scores > self.threshold, -1, 1)

    def _ensure_float32(self, X: np.ndarray) -> np.ndarray:
        """Ensure input is float32."""
        if X.dtype != np.float32:
            return X.astype(np.float32)
        return X

    def _update_metrics(self, n_samples: int, total_time: float) -> None:
        """Update inference metrics."""
        self._metrics.total_samples += n_samples
        self._metrics.total_time_ms += total_time * 1000

        if self._metrics.total_samples > 0:
            self._metrics.avg_time_per_sample_ms = (
                self._metrics.total_time_ms / self._metrics.total_samples
            )
            self._metrics.throughput_samples_per_sec = self._metrics.total_samples / (
                self._metrics.total_time_ms / 1000
            )

        if self._batch_times:
            self._metrics.avg_time_per_batch_ms = np.mean(self._batch_times) * 1000

    def get_metrics(self) -> PyTorchInferenceMetrics:
        """Get inference metrics.

        Returns:
            PyTorchInferenceMetrics with collected statistics
        """
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset collected metrics."""
        self._metrics = PyTorchInferenceMetrics(
            device=str(self.device),
            model_type=self.model.__class__.__name__,
        )
        self._batch_times = []

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: PyTorchEngineConfig | None = None,
    ) -> PyTorchInferenceEngine:
        """Load engine from saved checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            config: Engine configuration

        Returns:
            Loaded PyTorchInferenceEngine
        """
        from device_anomaly.models.vae_detector import VAEDetector

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load detector
        detector = VAEDetector.load_model(checkpoint_path)

        return cls(
            model=detector.model,
            threshold=detector._threshold,
            config=config,
            feature_cols=detector._feature_cols,
            scaler=detector.scaler,
            impute_values=detector.impute_values.to_dict()
            if detector.impute_values is not None
            else None,
        )


class FallbackPyTorchEngine:
    """Fallback engine that tries PyTorch first, then sklearn.

    Useful for gradual rollout of DL models.

    Example:
        engine = FallbackPyTorchEngine(
            pytorch_path="vae.pt",
            sklearn_path="isolation_forest.pkl"
        )
        scores = engine.score_samples(X)
    """

    def __init__(
        self,
        pytorch_path: str | Path | None = None,
        sklearn_path: str | Path | None = None,
        prefer_pytorch: bool = True,
    ):
        """Initialize fallback engine.

        Args:
            pytorch_path: Path to PyTorch model
            sklearn_path: Path to sklearn model
            prefer_pytorch: Try PyTorch first if True
        """
        self.pytorch_engine: PyTorchInferenceEngine | None = None
        self.sklearn_engine = None
        self.prefer_pytorch = prefer_pytorch
        self._active_engine: str = "none"

        # Try to load engines
        if pytorch_path:
            try:
                self.pytorch_engine = PyTorchInferenceEngine.from_checkpoint(pytorch_path)
                logger.info("Loaded PyTorch engine from %s", pytorch_path)
            except Exception as e:
                logger.warning("Failed to load PyTorch engine: %s", e)

        if sklearn_path:
            try:
                from device_anomaly.models.anomaly_detector import AnomalyDetectorIsolationForest

                self.sklearn_engine = AnomalyDetectorIsolationForest.load_model(sklearn_path)
                logger.info("Loaded sklearn engine from %s", sklearn_path)
            except Exception as e:
                logger.warning("Failed to load sklearn engine: %s", e)

        # Determine active engine
        if prefer_pytorch and self.pytorch_engine:
            self._active_engine = "pytorch"
        elif self.sklearn_engine:
            self._active_engine = "sklearn"
        elif self.pytorch_engine:
            self._active_engine = "pytorch"
        else:
            raise RuntimeError("No inference engine could be loaded")

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Score samples using active engine.

        Args:
            X: Input array

        Returns:
            Anomaly scores
        """
        if self._active_engine == "pytorch":
            try:
                return self.pytorch_engine.score_samples(X)
            except Exception as e:
                if self.sklearn_engine:
                    logger.warning("PyTorch failed, falling back to sklearn: %s", e)
                    self._active_engine = "sklearn"
                    return self.sklearn_engine.score(X)
                raise

        return self.sklearn_engine.score(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels using active engine.

        Args:
            X: Input array

        Returns:
            Labels: 1 for normal, -1 for anomaly
        """
        if self._active_engine == "pytorch":
            try:
                return self.pytorch_engine.predict(X)
            except Exception as e:
                if self.sklearn_engine:
                    logger.warning("PyTorch failed, falling back to sklearn: %s", e)
                    self._active_engine = "sklearn"
                    return self.sklearn_engine.predict(X)
                raise

        return self.sklearn_engine.predict(X)

    @property
    def active_engine(self) -> str:
        """Get the currently active engine type."""
        return self._active_engine

    def get_metrics(self) -> PyTorchInferenceMetrics | None:
        """Get metrics if PyTorch engine is active."""
        if self.pytorch_engine and self._active_engine == "pytorch":
            return self.pytorch_engine.get_metrics()
        return None
