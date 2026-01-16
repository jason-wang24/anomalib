# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""VAE: Variational Autoencoder for Anomaly Detection.

VAE is an anomaly detection model that uses a variational autoencoder architecture
to learn the normal data distribution. The model consists of an encoder network that
maps images to a latent distribution, and a decoder that reconstructs images from
sampled latent vectors.

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
    :class:`anomalib.models.image.vae.torch_model.VaeModel`:
        PyTorch implementation of the VAE model architecture.
    :class:`anomalib.models.image.vae.loss.VaeLoss`:
        Loss function for the VAE model.
"""

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import AUROC, Evaluator, F1Score
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .loss import VaeLoss
from .torch_model import VaeModel

logger = logging.getLogger(__name__)


class Vae(AnomalibModule):
    """PL Lightning Module for the VAE Algorithm.

    The VAE model consists of an encoder and decoder network. The encoder maps
    images to a latent distribution (parameterized by mean and log-variance),
    and the decoder reconstructs images from sampled latent vectors. Anomalies
    are detected by measuring the reconstruction error.

    Args:
        batch_size (int): Number of samples in each batch.
            Defaults to ``32``.
        n_features (int): Number of feature channels in CNN layers.
            Defaults to ``64``.
        latent_vec_size (int): Dimension of the latent space vectors.
            Defaults to ``128``.
        extra_layers (int, optional): Number of extra layers in encoder/decoder.
            Defaults to ``0``.
        kl_weight (float, optional): Weight for KL divergence loss component.
            Defaults to ``0.00025``.
        lr (float, optional): Learning rate for optimizer.
            Defaults to ``0.0002``.
        beta1 (float, optional): Beta1 parameter for Adam optimizer.
            Defaults to ``0.9``.
        beta2 (float, optional): Beta2 parameter for Adam optimizer.
            Defaults to ``0.999``.
        pre_processor (PreProcessor | bool, optional): Pre-processor to transform
            inputs before passing to model.
            Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor to generate
            predictions from model outputs.
            Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator to compute metrics.
            Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer to display results.
            Defaults to ``True``.

    Example:
        >>> from anomalib.models import Vae
        >>> model = Vae(
        ...     batch_size=32,
        ...     n_features=64,
        ...     latent_vec_size=128,
        ...     kl_weight=0.0005,
        ... )

    See Also:
        :class:`anomalib.models.image.vae.torch_model.VaeModel`:
            PyTorch implementation of the VAE model architecture.
        :class:`anomalib.models.image.vae.loss.VaeLoss`:
            Loss function for the VAE model.
    """

    def __init__(
        self,
        batch_size: int = 32,
        n_features: int = 64,
        latent_vec_size: int = 128,
        extra_layers: int = 0,
        kl_weight: float = 0.00025,
        lr: float = 0.0002,
        beta1: float = 0.9,
        beta2: float = 0.999,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        if self.input_size is None:
            msg = "VAE needs input size to build torch model."
            raise ValueError(msg)

        self.n_features = n_features
        self.latent_vec_size = latent_vec_size
        self.extra_layers = extra_layers

        self.min_scores: torch.Tensor = torch.tensor(float("inf"), dtype=torch.float32)  # pylint: disable=not-callable
        self.max_scores: torch.Tensor = torch.tensor(float("-inf"), dtype=torch.float32)  # pylint: disable=not-callable

        self.model = VaeModel(
            input_size=self.input_size,
            num_input_channels=3,
            n_features=self.n_features,
            latent_vec_size=self.latent_vec_size,
            extra_layers=self.extra_layers,
        )

        self.loss = VaeLoss(kl_weight=kl_weight)

        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.model: VaeModel

    def _reset_min_max(self) -> None:
        """Reset min_max scores."""
        self.min_scores = torch.tensor(float("inf"), dtype=torch.float32)  # pylint: disable=not-callable
        self.max_scores = torch.tensor(float("-inf"), dtype=torch.float32)  # pylint: disable=not-callable

    def configure_optimizers(self) -> optim.Optimizer:
        """Configure optimizer.

        Returns:
            Optimizer: Adam optimizer for the VAE model
        """
        return optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )

    def training_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> STEP_OUTPUT:
        """Perform the training step.

        Args:
            batch (Batch): Input batch containing images.
            batch_idx (int): Batch index.

        Returns:
            STEP_OUTPUT: Loss value
        """
        del batch_idx  # `batch_idx` variable is not used.

        # Forward pass
        reconstruction, mu, log_var = self.model(batch.image)

        # Compute loss
        loss = self.loss(batch.image, reconstruction, mu, log_var)

        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_start(self) -> None:
        """Reset min and max values for current validation epoch."""
        self._reset_min_max()
        return super().on_validation_start()

    def validation_step(self, batch: Batch, *args, **kwargs) -> Batch:
        """Update min and max scores from the current step.

        Args:
            batch (Batch): Input batch containing images.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            Batch: Updated batch with predictions.
        """
        del args, kwargs  # Unused arguments.

        # Forward pass - model returns InferenceBatch with predictions
        predictions = self.model(batch.image)

        # Update min/max
        self.max_scores = max(self.max_scores, torch.max(predictions.pred_score))
        self.min_scores = min(self.min_scores, torch.min(predictions.pred_score))

        # Update batch with predictions
        return batch.update(**predictions._asdict())

    def on_validation_batch_end(
        self,
        outputs: Batch,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Normalize outputs based on min/max values."""
        outputs.pred_score = self._normalize(outputs.pred_score)
        super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx=dataloader_idx)

    def on_test_start(self) -> None:
        """Reset min max values before test batch starts."""
        self._reset_min_max()
        return super().on_test_start()

    def test_step(self, batch: Batch, batch_idx: int, *args, **kwargs) -> Batch:
        """Update min and max scores from the current step.

        Args:
            batch (Batch): Input batch containing images.
            batch_idx (int): Batch index.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            Batch: Updated batch with predictions.
        """
        del args, kwargs  # Unused arguments.

        super().test_step(batch, batch_idx)
        self.max_scores = max(self.max_scores, torch.max(batch.pred_score))
        self.min_scores = min(self.min_scores, torch.min(batch.pred_score))
        return batch

    def on_test_batch_end(
        self,
        outputs: Batch,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Normalize outputs based on min/max values."""
        outputs.pred_score = self._normalize(outputs.pred_score)
        super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx=dataloader_idx)

    def _normalize(self, scores: torch.Tensor) -> torch.Tensor:
        """Normalize the scores based on min/max of entire dataset.

        Args:
            scores (torch.Tensor): Un-normalized scores.

        Returns:
            Tensor: Normalized scores.
        """
        return (scores - self.min_scores.to(scores.device)) / (
            self.max_scores.to(scores.device) - self.min_scores.to(scores.device)
        )

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return VAE trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_evaluator() -> Evaluator:
        """Default evaluator for VAE."""
        image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
        test_metrics = [image_auroc, image_f1score]
        return Evaluator(test_metrics=test_metrics)
