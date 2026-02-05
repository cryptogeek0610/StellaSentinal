"""Variational Autoencoder (VAE) architecture for anomaly detection.

This module implements a VAE architecture optimized for tabular anomaly detection
on device telemetry data. The architecture uses:
- Symmetric encoder/decoder with batch normalization
- Reparameterization trick for latent space sampling
- Configurable hidden dimensions and latent size
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """Variational Autoencoder for tabular anomaly detection.

    Architecture:
        Input (n_features)
          → Encoder: Linear → BN → ReLU → ... → mu, log_var
          → Reparameterize: z = mu + std * epsilon
          → Decoder: Linear → BN → ReLU → ... → reconstruction
          → Output (n_features)

    The anomaly score is computed as the reconstruction error (MSE).
    Higher reconstruction error indicates more anomalous data.

    Example:
        model = VAE(input_dim=100, hidden_dims=[256, 128, 64], latent_dim=32)
        x = torch.randn(32, 100)  # batch of 32 samples
        recon, mu, log_var = model(x)
        loss = model.loss_function(x, recon, mu, log_var)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = None,
        latent_dim: int = 32,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        """Initialize the VAE.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions for encoder
                         (decoder uses reverse). Default: [256, 128, 64]
            latent_dim: Dimension of the latent space
            dropout: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # Build encoder
        self.encoder = self._build_encoder()

        # Latent space projections
        encoder_output_dim = self.hidden_dims[-1]
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_log_var = nn.Linear(encoder_output_dim, latent_dim)

        # Build decoder
        self.decoder = self._build_decoder()

        # Final output layer (no activation - reconstruction)
        self.output_layer = nn.Linear(self.hidden_dims[0], input_dim)

        # Initialize weights
        self._init_weights()

    def _build_encoder(self) -> nn.Sequential:
        """Build the encoder network."""
        layers = []
        in_dim = self.input_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = hidden_dim

        return nn.Sequential(*layers)

    def _build_decoder(self) -> nn.Sequential:
        """Build the decoder network (reverse of encoder)."""
        layers = []
        reversed_dims = list(reversed(self.hidden_dims))

        # Start from latent dim
        layers.append(nn.Linear(self.latent_dim, reversed_dims[0]))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(reversed_dims[0]))
        layers.append(nn.ReLU(inplace=True))
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))

        # Hidden layers (skip first since we just added it)
        for i in range(len(reversed_dims) - 1):
            layers.append(nn.Linear(reversed_dims[i], reversed_dims[i + 1]))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(reversed_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space parameters.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tuple of (mu, log_var) tensors, each shape (batch_size, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample from latent space using reparameterization trick.

        z = mu + std * epsilon, where epsilon ~ N(0, 1)

        This allows gradients to flow through the sampling operation.

        Args:
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution

        Returns:
            Sampled latent vector z
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            epsilon = torch.randn_like(std)
            return mu + std * epsilon
        else:
            # During inference, use mean directly (deterministic)
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction.

        Args:
            z: Latent vector of shape (batch_size, latent_dim)

        Returns:
            Reconstruction of shape (batch_size, input_dim)
        """
        h = self.decoder(z)
        return self.output_layer(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tuple of (reconstruction, mu, log_var)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    def loss_function(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        kl_weight: float = 1e-3,
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss (ELBO).

        Loss = Reconstruction Loss + KL Weight * KL Divergence

        Args:
            x: Original input
            recon: Reconstruction from decoder
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            kl_weight: Weight for KL divergence (beta in beta-VAE)
            reduction: 'mean' or 'sum' for loss reduction

        Returns:
            Tuple of (total_loss, recon_loss, kl_loss)
        """
        # Reconstruction loss (MSE)
        if reduction == "mean":
            recon_loss = F.mse_loss(recon, x, reduction="mean")
        else:
            recon_loss = F.mse_loss(recon, x, reduction="sum")

        # KL divergence: D_KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # Formula: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

        kl_loss = kl_loss.mean() if reduction == "mean" else kl_loss.sum()

        total_loss = recon_loss + kl_weight * kl_loss

        return total_loss, recon_loss, kl_loss

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reconstruction error.

        This is used as the anomaly score - higher error indicates
        more anomalous samples.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tensor of reconstruction errors, shape (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            recon, _, _ = self.forward(x)
            # MSE per sample (mean over features)
            error = F.mse_loss(recon, x, reduction="none").mean(dim=1)
        return error

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation for input.

        Args:
            x: Input tensor

        Returns:
            Latent vector (using mean, deterministic)
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu

    def sample(self, num_samples: int, device: torch.device = None) -> torch.Tensor:
        """Generate samples from the learned distribution.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on

        Returns:
            Generated samples of shape (num_samples, input_dim)
        """
        device = device or next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        with torch.no_grad():
            samples = self.decode(z)
        return samples

    def get_config(self) -> dict:
        """Get model configuration for serialization."""
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "latent_dim": self.latent_dim,
            "dropout": self.dropout,
            "use_batch_norm": self.use_batch_norm,
        }

    @classmethod
    def from_config(cls, config: dict) -> VAE:
        """Create model from configuration dict."""
        return cls(
            input_dim=config["input_dim"],
            hidden_dims=config.get("hidden_dims", [256, 128, 64]),
            latent_dim=config.get("latent_dim", 32),
            dropout=config.get("dropout", 0.2),
            use_batch_norm=config.get("use_batch_norm", True),
        )


class Autoencoder(nn.Module):
    """Standard Autoencoder for comparison with VAE.

    Simpler than VAE - no probabilistic latent space.
    Can be useful as a baseline or when VAE's KL term causes issues.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = None,
        latent_dim: int = 32,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        """Initialize the Autoencoder.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of the latent/bottleneck layer
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in self.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        reversed_dims = list(reversed(self.hidden_dims))
        in_dim = latent_dim

        for h_dim in reversed_dims:
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reconstruction error."""
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            error = F.mse_loss(recon, x, reduction="none").mean(dim=1)
        return error

    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "latent_dim": self.latent_dim,
            "dropout": self.dropout,
            "use_batch_norm": self.use_batch_norm,
        }
