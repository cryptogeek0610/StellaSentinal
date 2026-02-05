"""
Inference Engine Abstraction Layer

Provides a unified interface for model inference that supports both
scikit-learn (joblib) and ONNX Runtime backends. This allows seamless
switching between engines for performance optimization and deployment.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class EngineType(StrEnum):
    """Available inference engine types"""

    SKLEARN = "sklearn"
    ONNX = "onnx"
    PYTORCH = "pytorch"  # Deep Learning models (VAE, Autoencoder)


class ExecutionProvider(StrEnum):
    """ONNX Runtime execution providers"""

    CPU = "CPUExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    TENSORRT = "TensorrtExecutionProvider"
    OPENVINO = "OpenVINOExecutionProvider"
    DIRECTML = "DmlExecutionProvider"  # Windows GPU


@dataclass
class InferenceMetrics:
    """Metrics collected during inference"""

    inference_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    num_samples: int = 0
    engine_type: str = ""
    provider: str | None = None

    @property
    def throughput_samples_per_sec(self) -> float:
        """Calculate throughput in samples/second"""
        total_time_sec = self.total_time_ms / 1000.0
        if total_time_sec > 0:
            return self.num_samples / total_time_sec
        return 0.0

    @property
    def total_time_ms(self) -> float:
        """Total time including all stages"""
        return self.inference_time_ms + self.preprocessing_time_ms + self.postprocessing_time_ms

    def __str__(self) -> str:
        return (
            f"InferenceMetrics(engine={self.engine_type}, "
            f"samples={self.num_samples}, "
            f"total={self.total_time_ms:.2f}ms, "
            f"throughput={self.throughput_samples_per_sec:.1f} samples/sec)"
        )


@dataclass
class EngineConfig:
    """Configuration for inference engines"""

    engine_type: EngineType = EngineType.SKLEARN
    onnx_provider: ExecutionProvider = ExecutionProvider.CPU
    intra_op_num_threads: int = 4
    inter_op_num_threads: int = 4
    enable_profiling: bool = False
    enable_graph_optimization: bool = True
    collect_metrics: bool = True


class InferenceEngine(ABC):
    """
    Abstract base class for inference engines.

    All engines must implement predict() and score_samples() methods
    to ensure compatibility with anomaly detection pipeline.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.metrics: InferenceMetrics | None = None

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions: 1 for normal, -1 for anomaly
        """
        pass

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Anomaly scores (higher = more normal)
        """
        pass

    @abstractmethod
    def get_engine_info(self) -> dict[str, Any]:
        """Return information about the engine"""
        pass

    def get_metrics(self) -> InferenceMetrics | None:
        """Get the latest inference metrics"""
        return self.metrics


class ScikitLearnEngine(InferenceEngine):
    """
    Inference engine using scikit-learn models loaded via joblib.

    This is the traditional approach using Python-based sklearn inference.
    """

    def __init__(self, model_path: str | Path, config: EngineConfig | None = None):
        super().__init__(config or EngineConfig(engine_type=EngineType.SKLEARN))
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load sklearn model from disk"""
        start = time.time()
        logger.info("Loading sklearn model from %s", self.model_path)

        try:
            self.model = joblib.load(self.model_path)
            load_time = (time.time() - start) * 1000
            logger.info("Sklearn model loaded in %.2f ms", load_time)

        except Exception as e:
            logger.error("Failed to load sklearn model: %s", e, exc_info=True)
            raise RuntimeError(f"Could not load sklearn model: {e}") from e

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using sklearn model"""
        start = time.time()

        predictions = self.model.predict(X)

        if self.config.collect_metrics:
            self.metrics = InferenceMetrics(
                inference_time_ms=(time.time() - start) * 1000,
                num_samples=len(X),
                engine_type="sklearn",
            )

        return predictions

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Score samples using sklearn model"""
        start = time.time()

        # IsolationForest uses decision_function for anomaly scores
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
        elif hasattr(self.model, "score_samples"):
            scores = self.model.score_samples(X)
        else:
            raise AttributeError(f"Model {type(self.model)} does not support scoring")

        if self.config.collect_metrics:
            self.metrics = InferenceMetrics(
                inference_time_ms=(time.time() - start) * 1000,
                num_samples=len(X),
                engine_type="sklearn",
            )

        return scores

    def get_engine_info(self) -> dict[str, Any]:
        """Return sklearn engine information"""
        return {
            "engine_type": "sklearn",
            "model_class": self.model.__class__.__name__,
            "model_path": str(self.model_path),
            "model_size_mb": self.model_path.stat().st_size / 1e6
            if self.model_path.exists()
            else 0,
        }


class ONNXInferenceEngine(InferenceEngine):
    """
    Inference engine using ONNX Runtime.

    Provides optimized inference with support for CPU, GPU, and other
    hardware accelerators. Typically 3-10x faster than sklearn.
    """

    def __init__(self, model_path: str | Path, config: EngineConfig | None = None):
        super().__init__(config or EngineConfig(engine_type=EngineType.ONNX))
        self.model_path = Path(model_path)
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.output_names: list[str] = []
        self._load_model()

    def _load_model(self) -> None:
        """Load ONNX model and create InferenceSession"""
        start = time.time()
        logger.info("Loading ONNX model from %s", self.model_path)

        try:
            # Configure session options
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.config.intra_op_num_threads
            sess_options.inter_op_num_threads = self.config.inter_op_num_threads

            if self.config.enable_graph_optimization:
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            else:
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

            if self.config.enable_profiling:
                sess_options.enable_profiling = True

            # Select execution provider
            providers = self._get_available_providers()
            logger.info("Available ONNX providers: %s", providers)
            logger.info("Using provider: %s", self.config.onnx_provider.value)

            # Create inference session
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=[self.config.onnx_provider.value],
            )

            # Cache input/output metadata
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [out.name for out in self.session.get_outputs()]

            load_time = (time.time() - start) * 1000
            logger.info("ONNX model loaded in %.2f ms", load_time)
            logger.info("Active provider: %s", self.session.get_providers())

        except Exception as e:
            logger.error("Failed to load ONNX model: %s", e, exc_info=True)
            raise RuntimeError(f"Could not load ONNX model: {e}") from e

    def _get_available_providers(self) -> list[str]:
        """Get list of available execution providers"""
        return ort.get_available_providers()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ONNX model"""
        start = time.time()

        # ONNX expects float32
        X = X.astype(np.float32)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: X})

        # First output is typically the prediction
        predictions = outputs[0].flatten()

        if self.config.collect_metrics:
            self.metrics = InferenceMetrics(
                inference_time_ms=(time.time() - start) * 1000,
                num_samples=len(X),
                engine_type="onnx",
                provider=self.config.onnx_provider.value,
            )

        return predictions

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Score samples using ONNX model"""
        start = time.time()

        # ONNX expects float32
        X = X.astype(np.float32)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: X})

        # For IsolationForest, second output is typically the score
        # If only one output, use that
        scores = outputs[1].flatten() if len(outputs) > 1 else outputs[0].flatten()

        if self.config.collect_metrics:
            self.metrics = InferenceMetrics(
                inference_time_ms=(time.time() - start) * 1000,
                num_samples=len(X),
                engine_type="onnx",
                provider=self.config.onnx_provider.value,
            )

        return scores

    def get_engine_info(self) -> dict[str, Any]:
        """Return ONNX engine information"""
        return {
            "engine_type": "onnx",
            "model_path": str(self.model_path),
            "model_size_mb": self.model_path.stat().st_size / 1e6
            if self.model_path.exists()
            else 0,
            "provider": self.config.onnx_provider.value,
            "active_providers": self.session.get_providers() if self.session else [],
            "available_providers": self._get_available_providers(),
            "intra_op_threads": self.config.intra_op_num_threads,
            "inter_op_threads": self.config.inter_op_num_threads,
        }

    def get_profiling_data(self) -> str | None:
        """Get profiling data if enabled"""
        if self.config.enable_profiling and self.session:
            return self.session.end_profiling()
        return None


def create_inference_engine(
    model_path: str | Path,
    engine_type: EngineType = EngineType.SKLEARN,
    config: EngineConfig | None = None,
) -> InferenceEngine:
    """
    Factory function to create the appropriate inference engine.

    Args:
        model_path: Path to model file (.pkl for sklearn, .onnx for ONNX, .pt for PyTorch)
        engine_type: Type of engine to create
        config: Engine configuration (optional)

    Returns:
        Configured inference engine

    Example:
        # Create ONNX engine with GPU
        config = EngineConfig(
            engine_type=EngineType.ONNX,
            onnx_provider=ExecutionProvider.CUDA
        )
        engine = create_inference_engine("model.onnx", config=config)

        # Create PyTorch engine for VAE model
        engine = create_inference_engine("vae.pt", engine_type=EngineType.PYTORCH)

        # Use for inference
        scores = engine.score_samples(features)
    """
    if config is None:
        config = EngineConfig(engine_type=engine_type)

    if engine_type == EngineType.ONNX:
        return ONNXInferenceEngine(model_path, config)
    elif engine_type == EngineType.SKLEARN:
        return ScikitLearnEngine(model_path, config)
    elif engine_type == EngineType.PYTORCH:
        return create_pytorch_engine(model_path, config)
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")


def create_pytorch_engine(
    model_path: str | Path,
    config: EngineConfig | None = None,
) -> InferenceEngine:
    """
    Create a PyTorch inference engine for Deep Learning models.

    Args:
        model_path: Path to PyTorch model file (.pt)
        config: Engine configuration

    Returns:
        PyTorch inference engine

    Example:
        engine = create_pytorch_engine("vae.pt")
        scores = engine.score_samples(features)
    """
    try:
        from device_anomaly.models.pytorch_inference import (
            PyTorchEngineConfig,
            PyTorchInferenceEngine,
        )
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for PyTorch engine. Install with: pip install torch"
        ) from e

    # Convert EngineConfig to PyTorchEngineConfig
    pytorch_config = PyTorchEngineConfig(
        device="cpu",  # Default to CPU
        collect_metrics=config.collect_metrics if config else True,
    )

    return PyTorchInferenceEngine.from_checkpoint(model_path, pytorch_config)


class FallbackInferenceEngine(InferenceEngine):
    """
    Inference engine with automatic fallback.

    Tries to use ONNX first, falls back to sklearn if ONNX fails.
    Useful for gradual rollout and high availability.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        sklearn_path: str | Path,
        config: EngineConfig | None = None,
    ):
        super().__init__(config or EngineConfig())
        self.onnx_path = Path(onnx_path)
        self.sklearn_path = Path(sklearn_path)
        self.active_engine: InferenceEngine | None = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize with ONNX, fallback to sklearn if needed"""
        try:
            logger.info("Attempting to load ONNX engine...")
            self.active_engine = ONNXInferenceEngine(self.onnx_path, self.config)
            logger.info("Successfully loaded ONNX engine")

        except Exception as e:
            logger.warning("ONNX engine failed to load: %s. Falling back to sklearn.", e)

            try:
                self.active_engine = ScikitLearnEngine(self.sklearn_path, self.config)
                logger.info("Successfully loaded sklearn fallback engine")

            except Exception as e2:
                logger.error("Both ONNX and sklearn engines failed to load")
                raise RuntimeError(
                    f"No valid inference engine available: ONNX={e}, sklearn={e2}"
                ) from e2

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forward to active engine"""
        return self.active_engine.predict(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Forward to active engine"""
        return self.active_engine.score_samples(X)

    def get_engine_info(self) -> dict[str, Any]:
        """Return active engine info with fallback status"""
        info = self.active_engine.get_engine_info()
        info["is_fallback"] = isinstance(self.active_engine, ScikitLearnEngine)
        return info
