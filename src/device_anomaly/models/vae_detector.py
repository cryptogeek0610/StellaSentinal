"""Variational Autoencoder (VAE) based anomaly detector.

This module provides a complete VAE-based anomaly detector that implements
the same interface as AnomalyDetectorIsolationForest, allowing seamless
integration with the existing training and inference pipelines.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from device_anomaly.models.dl_base import (
    BaseDLDetector,
    DLDetectorConfig,
)
from device_anomaly.models.vae_architecture import VAE, Autoencoder


class VAEDetectorConfig(DLDetectorConfig):
    """Configuration specific to VAE detector.

    Extends DLDetectorConfig with VAE-specific parameters.
    """

    def __init__(
        self,
        name: str = "vae_detector",
        model_type: str = "vae",  # 'vae' or 'autoencoder'
        **kwargs
    ):
        """Initialize VAE detector config.

        Args:
            name: Detector name
            model_type: 'vae' for Variational Autoencoder, 'autoencoder' for standard AE
            **kwargs: Additional arguments passed to DLDetectorConfig
        """
        super().__init__(name=name, **kwargs)
        self.model_type = model_type


class VAEDetector(BaseDLDetector):
    """Variational Autoencoder based anomaly detector.

    This detector uses reconstruction error as the anomaly score.
    Higher reconstruction error indicates more anomalous behavior.

    The detector implements the same interface as AnomalyDetectorIsolationForest:
    - fit(df) - Train on normal data
    - score(df) - Return anomaly scores
    - predict(df) - Return labels (1=normal, -1=anomaly)
    - score_dataframe(df) - Return DataFrame with scores and labels

    Example:
        config = VAEDetectorConfig(
            latent_dim=32,
            hidden_dims=[256, 128, 64],
            epochs=100,
            contamination=0.05,
        )
        detector = VAEDetector(config)
        detector.fit(train_df)

        # Score new data
        df_scored = detector.score_dataframe(test_df)
        anomalies = df_scored[df_scored['anomaly_label'] == -1]
    """

    def __init__(self, config: VAEDetectorConfig | None = None):
        """Initialize the VAE detector.

        Args:
            config: VAEDetectorConfig with detector settings
        """
        if config is None:
            config = VAEDetectorConfig()
        super().__init__(config)
        self.config: VAEDetectorConfig = config

    def _build_model(self, input_dim: int) -> nn.Module:
        """Build the VAE or Autoencoder model.

        Args:
            input_dim: Number of input features

        Returns:
            PyTorch model (VAE or Autoencoder)
        """
        if self.config.model_type == "autoencoder":
            return Autoencoder(
                input_dim=input_dim,
                hidden_dims=self.config.hidden_dims,
                latent_dim=self.config.latent_dim,
                dropout=self.config.dropout,
                use_batch_norm=True,
            )
        else:
            return VAE(
                input_dim=input_dim,
                hidden_dims=self.config.hidden_dims,
                latent_dim=self.config.latent_dim,
                dropout=self.config.dropout,
                use_batch_norm=True,
            )

    def _compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error for input batch.

        Args:
            x: Input tensor of shape (batch_size, n_features)

        Returns:
            Tensor of reconstruction errors, shape (batch_size,)
        """
        if isinstance(self.model, VAE):
            recon, _, _ = self.model(x)
        else:
            recon = self.model(x)

        # MSE per sample (mean over features)
        error = torch.mean((x - recon) ** 2, dim=1)
        return error

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
        if isinstance(self.model, VAE):
            recon, mu, log_var = model_output
            total_loss, _, _ = self.model.loss_function(
                x, recon, mu, log_var,
                kl_weight=self.config.kl_weight,
                reduction="mean",
            )
            return total_loss
        else:
            # Standard autoencoder - just MSE
            recon = model_output
            return torch.nn.functional.mse_loss(recon, x, reduction="mean")

    def get_latent_representation(self, df: pd.DataFrame) -> np.ndarray:
        """Get latent space representation for data.

        Useful for visualization and analysis.

        Args:
            df: Input DataFrame

        Returns:
            Latent vectors of shape (n_samples, latent_dim)
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted")

        data = self._prepare_inference_data(df)
        loader = self._create_dataloader(data, shuffle=False)

        self.model.eval()
        latents = []

        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                if isinstance(self.model, VAE):
                    mu, _ = self.model.encode(batch)
                    latents.append(mu.cpu().numpy())
                else:
                    z = self.model.encode(batch)
                    latents.append(z.cpu().numpy())

        return np.concatenate(latents, axis=0)

    def get_reconstruction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get reconstructed features for data.

        Useful for understanding what the model learned.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with reconstructed feature values
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted")

        data = self._prepare_inference_data(df)
        loader = self._create_dataloader(data, shuffle=False)

        self.model.eval()
        reconstructions = []

        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                if isinstance(self.model, VAE):
                    recon, _, _ = self.model(batch)
                else:
                    recon = self.model(batch)
                reconstructions.append(recon.cpu().numpy())

        recon_data = np.concatenate(reconstructions, axis=0)

        # Inverse transform if scaled
        if self.config.scale_features and self.scaler is not None:
            recon_data = self.scaler.inverse_transform(recon_data)

        return pd.DataFrame(recon_data, columns=self._feature_cols, index=df.index)

    def get_feature_reconstruction_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get per-feature reconstruction errors.

        Useful for identifying which features contribute most to anomalies.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with per-feature reconstruction errors
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted")

        data = self._prepare_inference_data(df)
        loader = self._create_dataloader(data, shuffle=False)

        self.model.eval()
        all_errors = []

        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                if isinstance(self.model, VAE):
                    recon, _, _ = self.model(batch)
                else:
                    recon = self.model(batch)

                # Per-feature squared error
                errors = (batch - recon) ** 2
                all_errors.append(errors.cpu().numpy())

        error_data = np.concatenate(all_errors, axis=0)
        return pd.DataFrame(error_data, columns=self._feature_cols, index=df.index)

    def explain_anomaly(
        self,
        df: pd.DataFrame,
        top_k: int = 5
    ) -> dict[int, dict[str, float]]:
        """Explain which features contribute most to each anomaly.

        Args:
            df: Input DataFrame
            top_k: Number of top contributing features to return

        Returns:
            Dict mapping row index to feature contributions
        """
        feature_errors = self.get_feature_reconstruction_errors(df)
        predictions = self.predict(df)

        contributions = {}
        for i, pred in enumerate(predictions):
            if pred == -1:  # Anomaly
                row_errors = feature_errors.iloc[i]
                top_features = row_errors.nlargest(top_k)
                contributions[i] = top_features.to_dict()

        return contributions

    def save_model(
        self,
        output_path: str | Path,
        export_onnx: bool = False
    ) -> dict[str, Path]:
        """Save trained model to disk.

        Extends parent method to save VAE-specific state.

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
            "model_config": self.model.get_config(),
            "model_type": self.config.model_type,
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
        logger.info("Saved VAE detector to %s", pytorch_path)

        # Export to ONNX if requested
        if export_onnx:
            try:
                from device_anomaly.models.pytorch_onnx_exporter import PyTorchONNXExporter

                onnx_path = output_path.with_suffix(".onnx")
                exporter = PyTorchONNXExporter()
                exporter.export_vae(
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
    def load_model(cls, model_path: str | Path) -> VAEDetector:
        """Load a trained VAE detector from disk.

        Args:
            model_path: Path to the saved .pt model file

        Returns:
            Loaded VAEDetector instance
        """
        import logging

        logger = logging.getLogger(__name__)
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        state = torch.load(model_path, map_location="cpu", weights_only=False)

        # Recreate config
        config = state.get("config", VAEDetectorConfig())
        if hasattr(config, 'model_type'):
            pass  # Already has model_type
        else:
            config.model_type = state.get("model_type", "vae")

        instance = cls(config=config)

        # Rebuild model from saved config
        model_config = state.get("model_config", {})
        input_dim = state["input_dim"]

        if state.get("model_type") == "autoencoder":
            instance.model = Autoencoder(
                input_dim=input_dim,
                hidden_dims=model_config.get("hidden_dims", config.hidden_dims),
                latent_dim=model_config.get("latent_dim", config.latent_dim),
                dropout=model_config.get("dropout", config.dropout),
            )
        else:
            instance.model = VAE(
                input_dim=input_dim,
                hidden_dims=model_config.get("hidden_dims", config.hidden_dims),
                latent_dim=model_config.get("latent_dim", config.latent_dim),
                dropout=model_config.get("dropout", config.dropout),
            )

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
            from datetime import datetime
            instance._fit_timestamp = datetime.fromisoformat(state["fit_timestamp"])

        logger.info(
            "Loaded VAEDetector from %s with %d features",
            model_path,
            len(instance._feature_cols),
        )

        return instance

    def get_metadata(self) -> dict[str, Any]:
        """Get detector metadata for observability."""
        metadata = super().get_metadata()
        metadata["model_type"] = self.config.model_type
        metadata["config"]["kl_weight"] = self.config.kl_weight

        if self._training_metrics:
            metadata["threshold_stats"] = {
                "threshold": self._threshold,
                "error_mean": self._error_mean,
                "error_std": self._error_std,
            }
            if self._training_metrics.reconstruction_error_percentiles:
                metadata["percentiles"] = self._training_metrics.reconstruction_error_percentiles

        return metadata


# Convenience function for creating detector
def create_vae_detector(
    latent_dim: int = 32,
    hidden_dims: list = None,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    contamination: float = 0.05,
    use_gpu: bool = False,
    model_type: str = "vae",
    **kwargs
) -> VAEDetector:
    """Create a VAE detector with common configuration.

    Args:
        latent_dim: Dimension of latent space
        hidden_dims: Hidden layer dimensions
        epochs: Training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        contamination: Expected anomaly rate
        use_gpu: Whether to use GPU
        model_type: 'vae' or 'autoencoder'
        **kwargs: Additional config parameters

    Returns:
        Configured VAEDetector instance
    """
    config = VAEDetectorConfig(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims or [256, 128, 64],
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        contamination=contamination,
        use_gpu=use_gpu,
        model_type=model_type,
        **kwargs
    )
    return VAEDetector(config)
