# Copyright (C) 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Loss functions for VAE model.

This module provides the loss function for training the Variational Autoencoder (VAE)
model. The loss combines reconstruction error with KL divergence to regularize the
latent space.

The VAE loss consists of two components:
1. Reconstruction loss: Measures how well the model reconstructs the input
2. KL divergence: Regularizes the latent space to follow a standard normal distribution

Example:
    >>> from anomalib.models.image.vae.loss import VaeLoss
    >>> criterion = VaeLoss(kl_weight=0.00025)
    >>> reconstruction, mu, log_var = model(images)
    >>> loss = criterion(images, reconstruction, mu, log_var)

See Also:
    - :class:`anomalib.models.image.vae.torch_model.VaeModel`:
        VAE model that this loss is designed for
    - :class:`anomalib.models.image.vae.lightning_model.Vae`:
        Lightning module that uses this loss
"""

import torch
from torch import nn


class VaeLoss(nn.Module):
    """Loss function for VAE combining reconstruction and KL divergence.

    The total loss is computed as:
        ``loss = reconstruction_loss + kl_weight * kl_divergence``

    Where:
        - reconstruction_loss: MSE between input and reconstructed image
        - kl_divergence: KL divergence between learned distribution and standard normal
        - kl_weight: Weight factor for KL divergence term

    Args:
        kl_weight (float, optional): Weight for KL divergence term.
            Defaults to ``0.00025``.

    Example:
        >>> criterion = VaeLoss(kl_weight=0.0005)
        >>> images = torch.randn(32, 3, 256, 256)
        >>> reconstruction, mu, log_var = model(images)
        >>> loss = criterion(images, reconstruction, mu, log_var)

    See Also:
        - :class:`VaeModel`: The model this loss is designed for
        - :meth:`Vae.training_step`: Where this loss is used during training
    """

    def __init__(self, kl_weight: float = 0.00025) -> None:
        super().__init__()
        self.kl_weight = kl_weight

    def forward(
        self,
        input_images: torch.Tensor,
        reconstructions: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """Compute VAE loss.

        Args:
            input_images (torch.Tensor): Original input images of shape
                ``(batch_size, channels, height, width)``
            reconstructions (torch.Tensor): Reconstructed images of shape
                ``(batch_size, channels, height, width)``
            mu (torch.Tensor): Mean of latent distribution of shape
                ``(batch_size, latent_vec_size, 1, 1)``
            log_var (torch.Tensor): Log variance of latent distribution of shape
                ``(batch_size, latent_vec_size, 1, 1)``

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = torch.mean((input_images - reconstructions) ** 2)

        # KL divergence loss
        # KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_divergence = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var), dim=1))

        # Total loss
        total_loss = reconstruction_loss + self.kl_weight * kl_divergence

        return total_loss

    def compute_anomaly_score(
        self,
        input_images: torch.Tensor,
        reconstructions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute anomaly score based on reconstruction error.

        Args:
            input_images (torch.Tensor): Original input images of shape
                ``(batch_size, channels, height, width)``
            reconstructions (torch.Tensor): Reconstructed images of shape
                ``(batch_size, channels, height, width)``

        Returns:
            torch.Tensor: Anomaly scores of shape ``(batch_size,)``
        """
        # Compute per-sample reconstruction error
        reconstruction_error = torch.mean((input_images - reconstructions) ** 2, dim=(1, 2, 3))
        return reconstruction_error
