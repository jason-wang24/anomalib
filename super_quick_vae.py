"""Super quick VAE test - minimal training, just to verify it works."""

from pathlib import Path
import torch
from lightning.pytorch import Trainer

from anomalib.data import Folder
from anomalib.models import Vae

print("="*60)
print("VAE Super Quick Test")
print("="*60)

# Setup paths
lag_root = Path("datasets/LAG_organized")

print("\n[1/3] Loading dataset...")
datamodule = Folder(
    name="LAG",
    root=lag_root,
    normal_dir="good",
    abnormal_dir="defect",
    train_batch_size=2,
    eval_batch_size=2,
    num_workers=0,
)
datamodule.setup()
print(f"✓ Dataset ready")

print("\n[2/3] Creating VAE model...")
model = Vae(
    batch_size=2,
    n_features=16,  # Tiny for speed
    latent_vec_size=32,
    kl_weight=0.0005,
)
print("✓ Model ready")

print("\n[3/3] Quick training (1 epoch, 3 batches only)...")
trainer = Trainer(
    max_epochs=1,
    accelerator="cpu",
    devices=1,
    default_root_dir="results/VAE",
    enable_progress_bar=True,
    enable_checkpointing=False,
    logger=False,
    limit_train_batches=3,
    limit_val_batches=1,
)

print("Training...")
trainer.fit(model, datamodule=datamodule)

print("\n✓ Training complete!")
print("\nNow running inference...")

# Save a quick checkpoint for external inference scripts
ckpt_dir = Path("results/VAE")
ckpt_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = ckpt_dir / "vae_quick.pth"
torch.save(model.state_dict(), ckpt_path)
print(f"Saved checkpoint: {ckpt_path}")

# Quick inference on a few test samples
trainer.test(model, datamodule=datamodule, verbose=False)

print("\n" + "="*60)
print("✓ VAE works! Check results/ folder for visualizations")
print("="*60)
