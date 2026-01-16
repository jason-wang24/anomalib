# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""VAE Algorithm Implementation.

VAE (Variational Autoencoder) is an anomaly detection model that uses a variational
autoencoder architecture to learn the normal data distribution. The model consists
of an encoder network that maps images to a latent distribution, and a decoder that
reconstructs images from sampled latent vectors.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Vae
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = Vae()
    >>> engine = Engine()

    >>> engine.fit(model, datamodule=datamodule)  # doctest: +SKIP
    >>> predictions = engine.predict(model, datamodule=datamodule)  # doctest: +SKIP

Paper:
    Title: Auto-Encoding Variational Bayes
    URL: https://arxiv.org/abs/1312.6114

See Also:
    :class:`anomalib.models.image.vae.lightning_model.Vae`:
        PyTorch Lightning implementation of the VAE model.
"""

from .lightning_model import Vae

__all__ = ["Vae"]
