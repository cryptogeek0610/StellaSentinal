"""PyTorch to ONNX model exporter for Deep Learning models.

This module provides utilities for exporting PyTorch models (VAE, Autoencoder)
to ONNX format for optimized inference. It includes:
- Export with dynamic batch size
- Validation against PyTorch output
- Metadata embedding
- Optional quantization
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    onnx = None
    ort = None


logger = logging.getLogger(__name__)


@dataclass
class PyTorchONNXExportConfig:
    """Configuration for PyTorch ONNX export.

    Attributes:
        opset_version: ONNX opset version (15 recommended for compatibility)
        do_constant_folding: Optimize constants in graph
        export_params: Include trained parameters in model
        verbose: Print export details
        validate: Validate exported model against PyTorch
        validation_samples: Number of samples for validation
        validation_tolerance: Maximum allowed difference
        validate_reconstruction: For VAE, validate reconstruction output
    """
    opset_version: int = 15
    do_constant_folding: bool = True
    export_params: bool = True
    verbose: bool = False
    validate: bool = True
    validation_samples: int = 100
    validation_tolerance: float = 1e-5
    validate_reconstruction: bool = True


class PyTorchONNXExporter:
    """Export PyTorch models to ONNX format.

    Supports exporting:
    - VAE models (exports reconstruction path)
    - Autoencoder models
    - Custom models with specified input/output names

    Example:
        exporter = PyTorchONNXExporter()

        # Export VAE
        exporter.export_vae(vae_model, input_dim=100, output_path="vae.onnx")

        # Validate
        is_valid = exporter.validate_vae_export(vae_model, "vae.onnx", input_dim=100)
    """

    def __init__(self, config: Optional[PyTorchONNXExportConfig] = None):
        """Initialize the exporter.

        Args:
            config: Export configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX is required. Install with: pip install onnx onnxruntime")

        self.config = config or PyTorchONNXExportConfig()

    def export(
        self,
        model: nn.Module,
        input_dim: int,
        output_path: str | Path,
        input_names: List[str] = None,
        output_names: List[str] = None,
        dynamic_axes: Dict[str, Dict[int, str]] = None,
    ) -> Path:
        """Export a generic PyTorch model to ONNX.

        Args:
            model: PyTorch model to export
            input_dim: Number of input features
            output_path: Path for output ONNX file
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes configuration

        Returns:
            Path to exported ONNX file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        device = next(model.parameters()).device

        # Create dummy input
        dummy_input = torch.randn(1, input_dim, device=device)

        # Default names
        input_names = input_names or ["input"]
        output_names = output_names or ["output"]

        # Default dynamic axes (batch size)
        if dynamic_axes is None:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }

        # Export
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=self.config.export_params,
            opset_version=self.config.opset_version,
            do_constant_folding=self.config.do_constant_folding,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=self.config.verbose,
        )

        logger.info("Exported PyTorch model to %s", output_path)

        # Validate if configured
        if self.config.validate:
            self._validate_export(model, output_path, input_dim, output_names[0])

        return output_path

    def export_vae(
        self,
        model: nn.Module,
        input_dim: int,
        output_path: str | Path,
        export_full: bool = False,
    ) -> Path:
        """Export VAE model to ONNX.

        For inference, we typically only need the reconstruction path.
        The model is exported in eval mode where reparameterization
        uses the mean directly (deterministic).

        Args:
            model: VAE model to export
            input_dim: Number of input features
            output_path: Path for output ONNX file
            export_full: If True, export all outputs (recon, mu, log_var)

        Returns:
            Path to exported ONNX file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        device = next(model.parameters()).device

        # Create wrapper for clean ONNX export
        if export_full:
            wrapper = VAEFullWrapper(model)
            output_names = ["reconstruction", "mu", "log_var"]
            dynamic_axes = {
                "input": {0: "batch_size"},
                "reconstruction": {0: "batch_size"},
                "mu": {0: "batch_size"},
                "log_var": {0: "batch_size"},
            }
        else:
            wrapper = VAEReconstructionWrapper(model)
            output_names = ["reconstruction"]
            dynamic_axes = {
                "input": {0: "batch_size"},
                "reconstruction": {0: "batch_size"},
            }

        wrapper.eval()

        # Create dummy input
        dummy_input = torch.randn(1, input_dim, device=device)

        # Export
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            export_params=self.config.export_params,
            opset_version=self.config.opset_version,
            do_constant_folding=self.config.do_constant_folding,
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=self.config.verbose,
        )

        logger.info("Exported VAE model to %s", output_path)

        # Validate
        if self.config.validate and self.config.validate_reconstruction:
            self._validate_vae_export(model, output_path, input_dim)

        return output_path

    def export_autoencoder(
        self,
        model: nn.Module,
        input_dim: int,
        output_path: str | Path,
    ) -> Path:
        """Export Autoencoder model to ONNX.

        Args:
            model: Autoencoder model
            input_dim: Number of input features
            output_path: Path for output ONNX file

        Returns:
            Path to exported ONNX file
        """
        return self.export(
            model=model,
            input_dim=input_dim,
            output_path=output_path,
            input_names=["input"],
            output_names=["reconstruction"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "reconstruction": {0: "batch_size"},
            },
        )

    def _validate_export(
        self,
        pytorch_model: nn.Module,
        onnx_path: Path,
        input_dim: int,
        output_name: str,
    ) -> bool:
        """Validate ONNX export against PyTorch model.

        Args:
            pytorch_model: Original PyTorch model
            onnx_path: Path to exported ONNX file
            input_dim: Input dimension
            output_name: Name of output to validate

        Returns:
            True if validation passes
        """
        pytorch_model.eval()
        device = next(pytorch_model.parameters()).device

        # Create test input
        test_input = np.random.randn(
            self.config.validation_samples, input_dim
        ).astype(np.float32)

        # PyTorch inference
        with torch.no_grad():
            pt_input = torch.tensor(test_input, device=device)
            pt_output = pytorch_model(pt_input)
            if isinstance(pt_output, tuple):
                pt_output = pt_output[0]  # Get reconstruction
            pt_output = pt_output.cpu().numpy()

        # ONNX inference
        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"]
        )
        onnx_output = session.run([output_name], {"input": test_input})[0]

        # Compare
        max_diff = np.max(np.abs(pt_output - onnx_output))
        mean_diff = np.mean(np.abs(pt_output - onnx_output))
        match_rate = np.mean(
            np.abs(pt_output - onnx_output) < self.config.validation_tolerance
        )

        if match_rate < 0.99:
            logger.warning(
                "ONNX validation: only %.2f%% values match within tolerance. "
                "max_diff=%.6f, mean_diff=%.6f",
                match_rate * 100, max_diff, mean_diff
            )
            return False

        logger.info(
            "ONNX validation passed: %.2f%% match, max_diff=%.6f",
            match_rate * 100, max_diff
        )
        return True

    def _validate_vae_export(
        self,
        pytorch_model: nn.Module,
        onnx_path: Path,
        input_dim: int,
    ) -> bool:
        """Validate VAE ONNX export.

        Args:
            pytorch_model: Original VAE model
            onnx_path: Path to ONNX file
            input_dim: Input dimension

        Returns:
            True if validation passes
        """
        pytorch_model.eval()
        device = next(pytorch_model.parameters()).device

        # Create test input
        test_input = np.random.randn(
            self.config.validation_samples, input_dim
        ).astype(np.float32)

        # PyTorch inference
        with torch.no_grad():
            pt_input = torch.tensor(test_input, device=device)
            pt_recon, pt_mu, _ = pytorch_model(pt_input)
            pt_recon = pt_recon.cpu().numpy()

        # ONNX inference
        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"]
        )
        onnx_recon = session.run(["reconstruction"], {"input": test_input})[0]

        # Compare reconstructions
        max_diff = np.max(np.abs(pt_recon - onnx_recon))
        mean_diff = np.mean(np.abs(pt_recon - onnx_recon))
        match_rate = np.mean(
            np.abs(pt_recon - onnx_recon) < self.config.validation_tolerance
        )

        if match_rate < 0.99:
            logger.warning(
                "VAE ONNX validation: %.2f%% match. max_diff=%.6f, mean_diff=%.6f",
                match_rate * 100, max_diff, mean_diff
            )
            return False

        logger.info(
            "VAE ONNX validation passed: %.2f%% match, max_diff=%.6f",
            match_rate * 100, max_diff
        )
        return True

    def add_metadata(
        self,
        onnx_path: str | Path,
        metadata: Dict[str, Any],
    ) -> None:
        """Add metadata to ONNX model.

        Args:
            onnx_path: Path to ONNX file
            metadata: Dictionary of metadata to add
        """
        model = onnx.load(str(onnx_path))

        for key, value in metadata.items():
            meta = model.metadata_props.add()
            meta.key = str(key)
            meta.value = str(value)

        onnx.save(model, str(onnx_path))
        logger.info("Added metadata to %s", onnx_path)

    def get_model_info(self, onnx_path: str | Path) -> Dict[str, Any]:
        """Get information about an ONNX model.

        Args:
            onnx_path: Path to ONNX file

        Returns:
            Dictionary with model info
        """
        model = onnx.load(str(onnx_path))

        info = {
            "opset_version": model.opset_import[0].version if model.opset_import else None,
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "ir_version": model.ir_version,
        }

        # Get inputs
        info["inputs"] = []
        for inp in model.graph.input:
            input_info = {"name": inp.name}
            if inp.type.tensor_type.shape.dim:
                input_info["shape"] = [
                    d.dim_value if d.dim_value else d.dim_param
                    for d in inp.type.tensor_type.shape.dim
                ]
            info["inputs"].append(input_info)

        # Get outputs
        info["outputs"] = []
        for out in model.graph.output:
            output_info = {"name": out.name}
            if out.type.tensor_type.shape.dim:
                output_info["shape"] = [
                    d.dim_value if d.dim_value else d.dim_param
                    for d in out.type.tensor_type.shape.dim
                ]
            info["outputs"].append(output_info)

        # Get metadata
        info["metadata"] = {
            meta.key: meta.value for meta in model.metadata_props
        }

        return info


class VAEReconstructionWrapper(nn.Module):
    """Wrapper for VAE that only outputs reconstruction.

    This simplifies the ONNX export by having a single output.
    During inference, the model uses mean (deterministic) for latent sampling.
    """

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only reconstruction."""
        recon, _, _ = self.vae(x)
        return recon


class VAEFullWrapper(nn.Module):
    """Wrapper for VAE that outputs all components.

    Outputs: (reconstruction, mu, log_var)
    """

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction, mu, and log_var."""
        return self.vae(x)


class ONNXVAEInference:
    """ONNX-based inference for VAE models.

    Provides a simple interface for scoring with exported VAE models.

    Example:
        inference = ONNXVAEInference("vae.onnx")
        reconstruction = inference.reconstruct(input_data)
        errors = inference.reconstruction_error(input_data)
    """

    def __init__(
        self,
        onnx_path: str | Path,
        providers: List[str] = None,
    ):
        """Initialize ONNX inference.

        Args:
            onnx_path: Path to ONNX model
            providers: ONNX Runtime execution providers
        """
        self.onnx_path = Path(onnx_path)

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Get reconstruction for input.

        Args:
            x: Input array of shape (batch_size, n_features)

        Returns:
            Reconstruction array
        """
        if x.dtype != np.float32:
            x = x.astype(np.float32)

        outputs = self.session.run(["reconstruction"], {self.input_name: x})
        return outputs[0]

    def reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        """Compute reconstruction error (anomaly score).

        Args:
            x: Input array

        Returns:
            Array of reconstruction errors (one per sample)
        """
        recon = self.reconstruct(x)
        # MSE per sample
        return np.mean((x - recon) ** 2, axis=1)

    def predict(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """Predict anomaly labels.

        Args:
            x: Input array
            threshold: Reconstruction error threshold

        Returns:
            Labels: 1 for normal, -1 for anomaly
        """
        errors = self.reconstruction_error(x)
        return np.where(errors > threshold, -1, 1)
