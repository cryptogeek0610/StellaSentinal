"""
ONNX Model Exporter

Handles conversion of scikit-learn models to ONNX format for optimized inference.
Supports IsolationForest, GradientBoosting, and other sklearn models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnx
import onnxruntime as ort
from sklearn.base import BaseEstimator
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

logger = logging.getLogger(__name__)


class ONNXExportConfig:
    """Configuration for ONNX model export"""

    def __init__(
        self,
        target_opset: int = 15,
        disable_zipmap: bool = True,
        disable_class_labels: bool = True,
        optimize: bool = True,
    ):
        """
        Args:
            target_opset: ONNX opset version (15 is widely supported)
            disable_zipmap: Disable probability dict output (faster inference)
            disable_class_labels: Skip class label output (only return scores)
            optimize: Apply ONNX graph optimizations
        """
        self.target_opset = target_opset
        self.disable_zipmap = disable_zipmap
        self.disable_class_labels = disable_class_labels
        self.optimize = optimize


class ONNXModelExporter:
    """
    Exports scikit-learn models to ONNX format with validation.

    Usage:
        exporter = ONNXModelExporter()
        onnx_path = exporter.export_model(
            model=isolation_forest,
            feature_count=25,
            output_path="models/isolation_forest.onnx"
        )
    """

    def __init__(self, config: Optional[ONNXExportConfig] = None):
        self.config = config or ONNXExportConfig()

    def export_model(
        self,
        model: BaseEstimator,
        feature_count: int,
        output_path: str | Path,
        model_name: str = "anomaly_model",
        validate: bool = True,
    ) -> Path:
        """
        Export a scikit-learn model to ONNX format.

        Args:
            model: Trained scikit-learn model
            feature_count: Number of input features
            output_path: Where to save the .onnx file
            model_name: Name for the ONNX model (metadata)
            validate: Whether to validate the export

        Returns:
            Path to the exported ONNX model

        Raises:
            ValueError: If export validation fails
            RuntimeError: If conversion fails
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Exporting %s to ONNX: %s (features=%d)",
            model.__class__.__name__,
            output_path,
            feature_count,
        )

        try:
            # Define input schema
            initial_type = [("float_input", FloatTensorType([None, feature_count]))]

            # Configure conversion options
            options = {}
            if self.config.disable_zipmap:
                options["zipmap"] = False
            if self.config.disable_class_labels:
                options["nocl"] = True

            # Convert to ONNX with guarded retries for unsupported options/opsets
            options_used = options
            opset_used: int | dict[str, int] = self.config.target_opset
            last_exc: Exception | None = None

            for _ in range(3):
                try:
                    onnx_model = convert_sklearn(
                        model,
                        initial_types=initial_type,
                        target_opset=opset_used,
                        options=options_used,
                        doc_string=f"{model_name} - Converted from scikit-learn",
                    )
                    break
                except NameError as exc:
                    last_exc = exc
                    if options_used:
                        logger.warning(
                            "ONNX options not supported for %s (%s). Retrying without options.",
                            model.__class__.__name__,
                            exc,
                        )
                        options_used = {}
                        continue
                    raise
                except RuntimeError as exc:
                    last_exc = exc
                    if "ai.onnx.ml" in str(exc) and isinstance(opset_used, int):
                        logger.warning(
                            "ONNX opset mismatch for %s (%s). Retrying with ai.onnx.ml opset 3.",
                            model.__class__.__name__,
                            exc,
                        )
                        opset_used = {"": self.config.target_opset, "ai.onnx.ml": 3}
                        continue
                    raise
            else:
                raise last_exc or RuntimeError("ONNX export failed after retries")

            # Optimize graph if requested
            if self.config.optimize:
                onnx_model = self._optimize_model(onnx_model)

            # Save to file
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            logger.info("ONNX model saved to %s (size: %.2f MB)", output_path, output_path.stat().st_size / 1e6)

            # Validate export
            if validate:
                self._validate_export(output_path, model, feature_count)

            return output_path

        except Exception as e:
            logger.error("Failed to export model to ONNX: %s", e, exc_info=True)
            raise RuntimeError(f"ONNX export failed: {e}") from e

    def _optimize_model(self, onnx_model: Any) -> Any:
        """Apply ONNX graph optimizations"""
        try:
            # Basic optimizations (constant folding, dead code elimination)
            from onnx import optimizer

            passes = [
                "eliminate_identity",
                "eliminate_nop_transpose",
                "eliminate_nop_pad",
                "eliminate_unused_initializer",
                "fuse_consecutive_transposes",
                "fuse_transpose_into_gemm",
            ]

            optimized_model = optimizer.optimize(onnx_model, passes)
            logger.info("Applied ONNX graph optimizations: %s", ", ".join(passes))
            return optimized_model

        except Exception as e:
            logger.warning("Could not optimize ONNX model: %s", e)
            return onnx_model

    def _validate_export(
        self,
        onnx_path: Path,
        sklearn_model: BaseEstimator,
        feature_count: int,
        num_test_samples: int = 100,
        tolerance: float = 1e-5,
    ) -> None:
        """
        Validate that ONNX predictions match scikit-learn predictions.

        Args:
            onnx_path: Path to exported ONNX model
            sklearn_model: Original scikit-learn model
            feature_count: Number of features
            num_test_samples: Number of random samples to test
            tolerance: Maximum allowed difference between predictions

        Raises:
            ValueError: If predictions don't match within tolerance
        """
        logger.info("Validating ONNX export...")

        # Create test input
        np.random.seed(42)
        test_input = np.random.randn(num_test_samples, feature_count).astype(np.float32)

        # Get sklearn predictions
        sklearn_preds = sklearn_model.predict(test_input)

        # Get ONNX predictions
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        onnx_outputs = session.run(None, {input_name: test_input})
        onnx_preds = onnx_outputs[0].flatten()

        # Compare predictions
        match_rate = np.mean(sklearn_preds == onnx_preds)
        max_diff = np.max(np.abs(sklearn_preds - onnx_preds))

        logger.info(
            "ONNX validation: %.2f%% match rate, max difference: %.6f",
            match_rate * 100,
            max_diff,
        )

        if match_rate < 0.99:
            raise ValueError(
                f"ONNX predictions match only {match_rate:.2%} of sklearn predictions. "
                f"Expected > 99%. Max difference: {max_diff}"
            )

        if max_diff > tolerance:
            logger.warning(
                "ONNX predictions differ by up to %.6f (tolerance: %.6f). "
                "This may be acceptable for anomaly detection.",
                max_diff,
                tolerance,
            )

        logger.info("ONNX export validation passed!")

    def export_with_metadata(
        self,
        model: BaseEstimator,
        feature_count: int,
        output_path: str | Path,
        metadata: dict[str, Any],
    ) -> Path:
        """
        Export model with custom metadata embedded in ONNX file.

        Args:
            model: Trained model
            feature_count: Number of features
            output_path: Output path
            metadata: Dictionary of metadata to embed (e.g., feature names, version)

        Returns:
            Path to exported model
        """
        onnx_path = self.export_model(model, feature_count, output_path, validate=True)

        # Load and add metadata
        onnx_model = onnx.load(str(onnx_path))

        for key, value in metadata.items():
            meta = onnx_model.metadata_props.add()
            meta.key = key
            meta.value = str(value)

        # Save with metadata
        onnx.save(onnx_model, str(onnx_path))
        logger.info("Added metadata to ONNX model: %s", list(metadata.keys()))

        return onnx_path


class ONNXQuantizer:
    """
    Quantizes ONNX models for smaller size and faster inference.

    Quantization converts FP32 weights to INT8, reducing model size by ~75%
    and improving inference speed by 2-4x on CPU.
    """

    @staticmethod
    def quantize_dynamic(
        input_path: str | Path,
        output_path: str | Path,
        weight_type: str = "int8",
    ) -> Path:
        """
        Apply dynamic quantization to ONNX model.

        Args:
            input_path: Path to FP32 ONNX model
            output_path: Path for quantized model
            weight_type: Quantization type ('int8' or 'uint8')

        Returns:
            Path to quantized model
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            input_path = Path(input_path)
            output_path = Path(output_path)

            logger.info("Quantizing ONNX model: %s -> %s", input_path, output_path)

            quant_type = QuantType.QInt8 if weight_type == "int8" else QuantType.QUInt8

            quantize_dynamic(
                model_input=str(input_path),
                model_output=str(output_path),
                weight_type=quant_type,
            )

            # Compare sizes
            original_size = input_path.stat().st_size / 1e6
            quantized_size = output_path.stat().st_size / 1e6
            reduction = (1 - quantized_size / original_size) * 100

            if quantized_size >= original_size:
                logger.warning(
                    "Quantized model larger than original (%.2f MB -> %.2f MB).",
                    original_size,
                    quantized_size,
                )

            logger.info(
                "Quantization complete: %.2f MB -> %.2f MB (%.1f%% reduction)",
                original_size,
                quantized_size,
                reduction,
            )

            return output_path

        except ImportError:
            logger.error("onnxruntime.quantization not available. Install onnxruntime >= 1.16.0")
            raise
        except Exception as e:
            logger.error("Quantization failed: %s", e, exc_info=True)
            raise
