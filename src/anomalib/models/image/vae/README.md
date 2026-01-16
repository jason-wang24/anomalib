# VAE - Variational Autoencoder

This is the implementation of the [VAE](https://arxiv.org/abs/1312.6114) model for anomaly detection.

Model Type: Segmentation

## Description

Variational Autoencoder (VAE) is a generative model that learns to encode input images into a probabilistic latent space and reconstruct them. The model is trained only on normal images and learns to represent the normal data distribution. During inference, anomalies are detected based on reconstruction error - images that deviate from the learned normal distribution will have higher reconstruction errors.

The VAE consists of:
- **Encoder**: Maps input images to latent distribution parameters (mean and log-variance)
- **Reparameterization**: Samples from the learned distribution using the reparameterization trick
- **Decoder**: Reconstructs images from the sampled latent vectors

The loss function combines:
1. **Reconstruction Loss**: MSE between input and reconstructed images
2. **KL Divergence**: Regularizes the latent space to follow a standard normal distribution

## Architecture

```
Input Image → Encoder → (μ, log σ²) → Reparameterization → z → Decoder → Reconstructed Image
                                           ↓
                                    KL Divergence Loss
                                           ↓
                                    Reconstruction Loss
```

The encoder progressively downsamples the input through convolutional layers to extract features and outputs distribution parameters. The decoder upsamples from the latent space through transposed convolutions to reconstruct the original image dimensions.

## Usage

```python
from anomalib.data import MVTecAD
from anomalib.models import Vae
from anomalib.engine import Engine

# Initialize datamodule, model and engine
datamodule = MVTecAD()
model = Vae(
    n_features=64,
    latent_vec_size=128,
    kl_weight=0.00025
)
engine = Engine()

# Train the model
engine.fit(model, datamodule=datamodule)

# Test the model
predictions = engine.test(model, datamodule=datamodule)
```

## Citation

```bibtex
@article{kingma2013auto,
  title={Auto-encoding variational bayes},
  author={Kingma, Diederik P and Welling, Max},
  journal={arXiv preprint arXiv:1312.6114},
  year={2013}
}
```
