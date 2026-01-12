"""Deep Learning configuration module.

This module provides configuration classes and utilities for Deep Learning
model training and inference. It integrates with the existing configuration
system while providing DL-specific settings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from device_anomaly.models.dl_base import DLDetectorConfig


@dataclass
class DLModelConfig:
    """Configuration for DL model architecture.

    Attributes:
        model_type: Type of model ('vae', 'autoencoder')
        latent_dim: Dimension of latent space
        hidden_dims: Hidden layer dimensions
        dropout: Dropout rate
        use_batch_norm: Whether to use batch normalization
    """
    model_type: str = "vae"
    latent_dim: int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.2
    use_batch_norm: bool = True


@dataclass
class DLTrainingConfig:
    """Configuration for DL model training.

    Attributes:
        epochs: Maximum training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        early_stopping_patience: Epochs before early stopping
        min_delta: Minimum improvement for early stopping
        gradient_clip_val: Gradient clipping value
        validation_split: Fraction for validation
        use_gpu: Whether to use GPU if available
        random_state: Random seed
    """
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    min_delta: float = 1e-4
    gradient_clip_val: float = 1.0
    validation_split: float = 0.1
    use_gpu: bool = False
    random_state: int = 42


@dataclass
class DLInferenceConfig:
    """Configuration for DL model inference.

    Attributes:
        device: Inference device ('cpu', 'cuda')
        batch_size: Inference batch size
        use_onnx: Use ONNX for inference if available
        warmup_iterations: Warmup iterations for benchmarking
    """
    device: str = "cpu"
    batch_size: Optional[int] = None
    use_onnx: bool = True
    warmup_iterations: int = 3


@dataclass
class DLConfig:
    """Complete Deep Learning configuration.

    Combines model, training, and inference configurations.

    Example:
        config = DLConfig()
        config.model.latent_dim = 64
        config.training.epochs = 200

        detector = VAEDetector.from_config(config)
    """
    model: DLModelConfig = field(default_factory=DLModelConfig)
    training: DLTrainingConfig = field(default_factory=DLTrainingConfig)
    inference: DLInferenceConfig = field(default_factory=DLInferenceConfig)
    contamination: float = 0.05  # Expected anomaly rate

    def to_detector_config(self) -> DLDetectorConfig:
        """Convert to DLDetectorConfig for detector initialization.

        Returns:
            DLDetectorConfig instance
        """
        return DLDetectorConfig(
            name=f"{self.model.model_type}_detector",
            latent_dim=self.model.latent_dim,
            hidden_dims=self.model.hidden_dims,
            dropout=self.model.dropout,
            learning_rate=self.training.learning_rate,
            weight_decay=self.training.weight_decay,
            epochs=self.training.epochs,
            batch_size=self.training.batch_size,
            early_stopping_patience=self.training.early_stopping_patience,
            min_delta=self.training.min_delta,
            contamination=self.contamination,
            use_gpu=self.training.use_gpu,
            random_state=self.training.random_state,
            gradient_clip_val=self.training.gradient_clip_val,
            validation_split=self.training.validation_split,
        )


def get_dl_config() -> DLConfig:
    """Get DL configuration from environment variables.

    Environment variables:
        DL_MODEL_TYPE: Model type ('vae', 'autoencoder')
        DL_LATENT_DIM: Latent dimension
        DL_HIDDEN_DIMS: Hidden dimensions (comma-separated)
        DL_EPOCHS: Training epochs
        DL_BATCH_SIZE: Batch size
        DL_LEARNING_RATE: Learning rate
        DL_USE_GPU: Use GPU (true/false)
        DL_CONTAMINATION: Expected anomaly rate

    Returns:
        DLConfig populated from environment
    """
    config = DLConfig()

    # Model config
    if os.getenv("DL_MODEL_TYPE"):
        config.model.model_type = os.getenv("DL_MODEL_TYPE")

    if os.getenv("DL_LATENT_DIM"):
        config.model.latent_dim = int(os.getenv("DL_LATENT_DIM"))

    if os.getenv("DL_HIDDEN_DIMS"):
        dims_str = os.getenv("DL_HIDDEN_DIMS")
        config.model.hidden_dims = [int(d.strip()) for d in dims_str.split(",")]

    if os.getenv("DL_DROPOUT"):
        config.model.dropout = float(os.getenv("DL_DROPOUT"))

    # Training config
    if os.getenv("DL_EPOCHS"):
        config.training.epochs = int(os.getenv("DL_EPOCHS"))

    if os.getenv("DL_BATCH_SIZE"):
        config.training.batch_size = int(os.getenv("DL_BATCH_SIZE"))

    if os.getenv("DL_LEARNING_RATE"):
        config.training.learning_rate = float(os.getenv("DL_LEARNING_RATE"))

    if os.getenv("DL_WEIGHT_DECAY"):
        config.training.weight_decay = float(os.getenv("DL_WEIGHT_DECAY"))

    if os.getenv("DL_EARLY_STOPPING_PATIENCE"):
        config.training.early_stopping_patience = int(os.getenv("DL_EARLY_STOPPING_PATIENCE"))

    if os.getenv("DL_USE_GPU"):
        config.training.use_gpu = os.getenv("DL_USE_GPU", "false").lower() == "true"

    if os.getenv("DL_VALIDATION_SPLIT"):
        config.training.validation_split = float(os.getenv("DL_VALIDATION_SPLIT"))

    # Inference config
    if os.getenv("DL_INFERENCE_DEVICE"):
        config.inference.device = os.getenv("DL_INFERENCE_DEVICE")

    if os.getenv("DL_USE_ONNX"):
        config.inference.use_onnx = os.getenv("DL_USE_ONNX", "true").lower() == "true"

    # General
    if os.getenv("DL_CONTAMINATION"):
        config.contamination = float(os.getenv("DL_CONTAMINATION"))

    return config


def create_default_vae_config() -> DLConfig:
    """Create default configuration for VAE anomaly detector.

    Returns:
        DLConfig with VAE defaults
    """
    return DLConfig(
        model=DLModelConfig(
            model_type="vae",
            latent_dim=32,
            hidden_dims=[256, 128, 64],
            dropout=0.2,
        ),
        training=DLTrainingConfig(
            epochs=100,
            batch_size=256,
            learning_rate=1e-3,
            early_stopping_patience=10,
        ),
        contamination=0.05,
    )


def create_default_autoencoder_config() -> DLConfig:
    """Create default configuration for Autoencoder anomaly detector.

    Returns:
        DLConfig with Autoencoder defaults
    """
    return DLConfig(
        model=DLModelConfig(
            model_type="autoencoder",
            latent_dim=32,
            hidden_dims=[256, 128, 64],
            dropout=0.2,
        ),
        training=DLTrainingConfig(
            epochs=100,
            batch_size=256,
            learning_rate=1e-3,
            early_stopping_patience=10,
        ),
        contamination=0.05,
    )


# Preset configurations for common use cases
DL_PRESETS: Dict[str, DLConfig] = {
    "vae_small": DLConfig(
        model=DLModelConfig(
            model_type="vae",
            latent_dim=16,
            hidden_dims=[128, 64],
            dropout=0.1,
        ),
        training=DLTrainingConfig(
            epochs=50,
            batch_size=128,
        ),
    ),
    "vae_medium": create_default_vae_config(),
    "vae_large": DLConfig(
        model=DLModelConfig(
            model_type="vae",
            latent_dim=64,
            hidden_dims=[512, 256, 128],
            dropout=0.3,
        ),
        training=DLTrainingConfig(
            epochs=200,
            batch_size=512,
            learning_rate=5e-4,
        ),
    ),
    "autoencoder_small": DLConfig(
        model=DLModelConfig(
            model_type="autoencoder",
            latent_dim=16,
            hidden_dims=[128, 64],
        ),
        training=DLTrainingConfig(
            epochs=50,
            batch_size=128,
        ),
    ),
    "autoencoder_medium": create_default_autoencoder_config(),
}


def get_preset(name: str) -> DLConfig:
    """Get a preset DL configuration.

    Args:
        name: Preset name ('vae_small', 'vae_medium', 'vae_large',
              'autoencoder_small', 'autoencoder_medium')

    Returns:
        DLConfig for the specified preset

    Raises:
        KeyError: If preset name is not found
    """
    if name not in DL_PRESETS:
        available = ", ".join(DL_PRESETS.keys())
        raise KeyError(f"Unknown preset: {name}. Available: {available}")
    return DL_PRESETS[name]
