"""
ONNX Runtime Configuration

Configuration for ONNX model export, inference, and optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from device_anomaly.models.inference_engine import EngineType, ExecutionProvider


@dataclass
class ONNXExportSettings:
    """Settings for ONNX model export"""

    enabled: bool = True
    target_opset: int = 15
    disable_zipmap: bool = True
    disable_class_labels: bool = True
    optimize: bool = True
    validate_export: bool = True
    export_quantized: bool = False


@dataclass
class ONNXInferenceSettings:
    """Settings for ONNX Runtime inference"""

    engine_type: EngineType = EngineType.SKLEARN
    onnx_provider: ExecutionProvider = ExecutionProvider.CPU
    intra_op_num_threads: int = 4
    inter_op_num_threads: int = 4
    enable_profiling: bool = False
    enable_graph_optimization: bool = True
    collect_metrics: bool = True
    enable_fallback: bool = True  # Fallback to sklearn if ONNX fails


@dataclass
class ONNXModelPaths:
    """Paths for ONNX model storage"""

    base_dir: Path = field(default_factory=lambda: Path("models/onnx"))
    isolation_forest_fp32: str = "isolation_forest.onnx"
    isolation_forest_int8: str = "isolation_forest_int8.onnx"
    calibration_model_fp32: str = "calibration_model.onnx"
    calibration_model_int8: str = "calibration_model_int8.onnx"

    def get_full_path(self, model_name: str) -> Path:
        """Get full path for a model file"""
        return self.base_dir / model_name

    def ensure_dirs(self) -> None:
        """Create model directories if they don't exist"""
        self.base_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ONNXConfig:
    """
    Complete ONNX configuration.

    Usage:
        config = ONNXConfig()

        # Enable ONNX with GPU
        config.inference.engine_type = EngineType.ONNX
        config.inference.onnx_provider = ExecutionProvider.CUDA

        # Export models
        config.export.export_quantized = True
    """

    export: ONNXExportSettings = field(default_factory=ONNXExportSettings)
    inference: ONNXInferenceSettings = field(default_factory=ONNXInferenceSettings)
    paths: ONNXModelPaths = field(default_factory=ONNXModelPaths)

    @classmethod
    def from_env(cls) -> ONNXConfig:
        """
        Load configuration from environment variables.

        Environment variables:
            ONNX_ENABLED: Enable ONNX export (default: true)
            ONNX_ENGINE: Engine type (sklearn|onnx, default: sklearn)
            ONNX_PROVIDER: Execution provider (cpu|cuda|tensorrt, default: cpu)
            ONNX_THREADS: Number of threads (default: 4)
            ONNX_QUANTIZE: Enable quantization (default: false)
            ONNX_PROFILING: Enable profiling (default: false)
        """
        import os

        config = cls()

        # Export settings
        config.export.enabled = os.getenv("ONNX_ENABLED", "true").lower() == "true"
        config.export.export_quantized = os.getenv("ONNX_QUANTIZE", "false").lower() == "true"

        # Inference settings
        engine_str = os.getenv("ONNX_ENGINE", "sklearn").lower()
        config.inference.engine_type = (
            EngineType.ONNX if engine_str == "onnx" else EngineType.SKLEARN
        )

        provider_str = os.getenv("ONNX_PROVIDER", "cpu").lower()
        provider_map = {
            "cpu": ExecutionProvider.CPU,
            "cuda": ExecutionProvider.CUDA,
            "tensorrt": ExecutionProvider.TENSORRT,
            "openvino": ExecutionProvider.OPENVINO,
            "directml": ExecutionProvider.DIRECTML,
        }
        config.inference.onnx_provider = provider_map.get(provider_str, ExecutionProvider.CPU)

        threads = int(os.getenv("ONNX_THREADS", "4"))
        config.inference.intra_op_num_threads = threads
        config.inference.inter_op_num_threads = threads

        config.inference.enable_profiling = os.getenv("ONNX_PROFILING", "false").lower() == "true"

        # Model paths
        model_dir = os.getenv("ONNX_MODEL_DIR", "models/onnx")
        config.paths.base_dir = Path(model_dir)

        return config

    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            "export": {
                "enabled": self.export.enabled,
                "target_opset": self.export.target_opset,
                "optimize": self.export.optimize,
                "export_quantized": self.export.export_quantized,
            },
            "inference": {
                "engine_type": self.inference.engine_type.value,
                "onnx_provider": self.inference.onnx_provider.value,
                "intra_op_threads": self.inference.intra_op_num_threads,
                "inter_op_threads": self.inference.inter_op_num_threads,
                "enable_profiling": self.inference.enable_profiling,
                "collect_metrics": self.inference.collect_metrics,
            },
            "paths": {
                "base_dir": str(self.paths.base_dir),
            },
        }


# Global configuration instance
_global_config: ONNXConfig | None = None


def get_onnx_config() -> ONNXConfig:
    """Get global ONNX configuration (loads from env on first call)"""
    global _global_config
    if _global_config is None:
        _global_config = ONNXConfig.from_env()
    return _global_config


def set_onnx_config(config: ONNXConfig) -> None:
    """Set global ONNX configuration"""
    global _global_config
    _global_config = config
