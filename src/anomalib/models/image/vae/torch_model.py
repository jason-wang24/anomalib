# Copyright (C) 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch models defining encoder, decoder, and VAE networks.

The VAE (Variational Autoencoder) model consists of several key components:

1. Encoder: Compresses input images into latent distribution parameters (mu, log_var)
2. Decoder: Reconstructs images from sampled latent vectors
3. VAE: Combines encoder-decoder with reparameterization trick

The architecture follows an encoder-decoder pattern where:
- Encoder compresses input image to latent distribution parameters (mu, logvar)
- Reparameterization trick samples from the distribution
- Decoder reconstructs the image from sampled latent vector
- Anomaly score is based on reconstruction error and KL divergence

Example:
    >>> from anomalib.models.image.vae.torch_model import VaeModel
    >>> model = VaeModel(
    ...     input_size=(256, 256),
    ...     num_input_channels=3,
    ...     n_features=64,
    ...     latent_vec_size=128,
    ...     extra_layers=0
    ... )
    >>> input_tensor = torch.randn(32, 3, 256, 256)
    >>> output = model(input_tensor)

References:
    Based on implementations from:
    - MedIAnomaly: https://github.com/caiyu6666/MedIAnomaly
    - Original VAE paper: https://arxiv.org/abs/1312.6114

See Also:
    - :class:`anomalib.models.image.vae.lightning_model.Vae`:
        Lightning implementation of the VAE model
    - :class:`anomalib.models.image.vae.loss.VaeLoss`:
        Loss function combining reconstruction and KL divergence
"""

import math

import torch
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.data.utils.image import pad_nextpow2


class Encoder(nn.Module):
    """Encoder Network for VAE.

    Compresses input images into latent distribution parameters (mean and log-variance)
    through a series of convolution layers.

    Args:
        input_size (tuple[int, int]): Size of input image (height, width)
        latent_vec_size (int): Size of output latent vector
        num_input_channels (int): Number of input image channels
        n_features (int): Number of feature maps in convolution layers
        extra_layers (int, optional): Number of extra intermediate layers.
            Defaults to ``0``.

    Example:
        >>> encoder = Encoder(
        ...     input_size=(256, 256),
        ...     latent_vec_size=128,
        ...     num_input_channels=3,
        ...     n_features=64
        ... )
        >>> input_tensor = torch.randn(32, 3, 256, 256)
        >>> mu, log_var = encoder(input_tensor)
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
    ) -> None:
        super().__init__()

        self.input_layers = nn.Sequential()
        self.input_layers.add_module(
            f"initial-conv-{num_input_channels}-{n_features}",
            nn.Conv2d(num_input_channels, n_features, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.input_layers.add_module(f"initial-batchnorm-{n_features}", nn.BatchNorm2d(n_features))
        self.input_layers.add_module(f"initial-relu-{n_features}", nn.LeakyReLU(0.2, inplace=True))

        # Extra Layers
        self.extra_layers = nn.Sequential()
        for layer in range(extra_layers):
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_features}-conv",
                nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.extra_layers.add_module(f"extra-layers-{layer}-{n_features}-batchnorm", nn.BatchNorm2d(n_features))
            self.extra_layers.add_module(f"extra-layers-{layer}-{n_features}-relu", nn.LeakyReLU(0.2, inplace=True))

        # Create pyramid features to reach latent vector
        self.pyramid_features = nn.Sequential()
        pyramid_dim = min(*input_size) // 2  # Use the smaller dimension to create pyramid.
        while pyramid_dim > 4:
            in_features = n_features
            out_features = n_features * 2
            self.pyramid_features.add_module(
                f"pyramid-{in_features}-{out_features}-conv",
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False),
            )
            self.pyramid_features.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            self.pyramid_features.add_module(f"pyramid-{out_features}-relu", nn.LeakyReLU(0.2, inplace=True))
            n_features = out_features
            pyramid_dim = pyramid_dim // 2

        # Final layers to output mu and log_var
        # Output size is 2 * latent_vec_size to account for both mu and log_var
        self.fc_mu_logvar = nn.Conv2d(
            n_features,
            2 * latent_vec_size,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder network.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape
                ``(batch_size, channels, height, width)``

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of (mu, log_var) tensors
                representing the latent distribution parameters
        """
        output = self.input_layers(input_tensor)
        output = self.extra_layers(output)
        output = self.pyramid_features(output)
        output = self.fc_mu_logvar(output)

        # Split into mu and log_var
        mu, log_var = output.chunk(2, dim=1)

        return mu, log_var


class Decoder(nn.Module):
    """Decoder Network for VAE.

    Reconstructs images from latent vectors through transposed convolutions.

    Args:
        input_size (tuple[int, int]): Size of output image (height, width)
        latent_vec_size (int): Size of input latent vector
        num_input_channels (int): Number of output image channels
        n_features (int): Number of feature maps in convolution layers
        extra_layers (int, optional): Number of extra intermediate layers.
            Defaults to ``0``.

    Example:
        >>> decoder = Decoder(
        ...     input_size=(256, 256),
        ...     latent_vec_size=128,
        ...     num_input_channels=3,
        ...     n_features=64
        ... )
        >>> latent = torch.randn(32, 128, 1, 1)
        >>> reconstruction = decoder(latent)
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
    ) -> None:
        super().__init__()

        self.latent_input = nn.Sequential()

        # Calculate input channel size to recreate inverse pyramid
        exp_factor = math.ceil(math.log(min(input_size) // 2, 2)) - 2
        n_input_features = n_features * (2**exp_factor)

        # CNN layer for latent vector input
        self.latent_input.add_module(
            f"initial-{latent_vec_size}-{n_input_features}-convt",
            nn.ConvTranspose2d(
                latent_vec_size,
                n_input_features,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
        )
        self.latent_input.add_module(f"initial-{n_input_features}-batchnorm", nn.BatchNorm2d(n_input_features))
        self.latent_input.add_module(f"initial-{n_input_features}-relu", nn.ReLU(inplace=True))

        # Create inverse pyramid
        self.inverse_pyramid = nn.Sequential()
        pyramid_dim = min(*input_size) // 2  # Use the smaller dimension to create pyramid.
        while pyramid_dim > 4:
            in_features = n_input_features
            out_features = n_input_features // 2
            self.inverse_pyramid.add_module(
                f"pyramid-{in_features}-{out_features}-convt",
                nn.ConvTranspose2d(
                    in_features,
                    out_features,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
            )
            self.inverse_pyramid.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            self.inverse_pyramid.add_module(f"pyramid-{out_features}-relu", nn.ReLU(inplace=True))
            n_input_features = out_features
            pyramid_dim = pyramid_dim // 2

        # Extra Layers
        self.extra_layers = nn.Sequential()
        for layer in range(extra_layers):
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-conv",
                nn.Conv2d(n_input_features, n_input_features, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-batchnorm",
                nn.BatchNorm2d(n_input_features),
            )
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-relu",
                nn.ReLU(inplace=True),
            )

        # Final layers
        self.final_layers = nn.Sequential()
        self.final_layers.add_module(
            f"final-{n_input_features}-{num_input_channels}-convt",
            nn.ConvTranspose2d(
                n_input_features,
                num_input_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
        )
        self.final_layers.add_module(f"final-{num_input_channels}-tanh", nn.Tanh())

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder network.

        Args:
            input_tensor (torch.Tensor): Input latent tensor

        Returns:
            torch.Tensor: Reconstructed image tensor
        """
        output = self.latent_input(input_tensor)
        output = self.inverse_pyramid(output)
        output = self.extra_layers(output)
        return self.final_layers(output)


class VaeModel(nn.Module):
    """VAE (Variational Autoencoder) Model.

    Combines encoder-decoder architecture with reparameterization trick for
    probabilistic latent space modeling. The model learns to reconstruct normal
    images while maintaining a structured latent space.

    Args:
        input_size (tuple[int, int]): Input/output image size (height, width)
        latent_vec_size (int): Size of latent vector
        num_input_channels (int): Number of input/output image channels
        n_features (int): Number of feature maps in convolution layers
        extra_layers (int, optional): Number of extra intermediate layers.
            Defaults to ``0``.

    Example:
        >>> vae = VaeModel(
        ...     input_size=(256, 256),
        ...     latent_vec_size=128,
        ...     num_input_channels=3,
        ...     n_features=64
        ... )
        >>> input_tensor = torch.randn(32, 3, 256, 256)
        >>> reconstruction, mu, log_var = vae(input_tensor)
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            input_size,
            latent_vec_size,
            num_input_channels,
            n_features,
            extra_layers,
        )
        self.decoder = Decoder(input_size, latent_vec_size, num_input_channels, n_features, extra_layers)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var) from N(0,1).

        Args:
            mu (torch.Tensor): Mean of the latent distribution
            log_var (torch.Tensor): Log variance of the latent distribution

        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        batch: torch.Tensor | InferenceBatch,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | InferenceBatch:
        """Forward pass through VAE.

        Args:
            batch (torch.Tensor | InferenceBatch): Input batch

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | InferenceBatch:
                During training, returns tuple of (reconstruction, mu, log_var).
                During inference, returns InferenceBatch with predictions.
        """
        # Handle both tensor and InferenceBatch inputs
        if isinstance(batch, InferenceBatch):
            input_tensor = batch.image
        else:
            input_tensor = batch

        # Encode
        mu, log_var = self.encoder(input_tensor)

        # Reparameterize
        z = self.reparameterize(mu, log_var)

        # Decode
        reconstruction = self.decoder(z)

        # Return different outputs based on training mode
        if self.training:
            return reconstruction, mu, log_var

        # Compute anomaly map (pixel-wise reconstruction error)
        anomaly_map = torch.mean((input_tensor - reconstruction) ** 2, dim=1, keepdim=True)

        # Compute anomaly score (mean over all pixels)
        pred_score = torch.mean(anomaly_map, dim=(1, 2, 3))

        return InferenceBatch(
            pred_score=pred_score,
            anomaly_map=anomaly_map.squeeze(1),
            pred_label=None,
        )
