"""Run inference with GANomaly and extract reconstructions directly from the generator."""

from pathlib import Path
import os
import torch
from PIL import Image
import numpy as np
from scipy.ndimage import zoom

from anomalib.data import Folder
from anomalib.models import Ganomaly


def find_checkpoint():
    """Find the latest checkpoint file in the results directory."""
    results_dir = Path("./results/lag_ganomaly")
    
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return None
    
    checkpoints = list(results_dir.rglob("*.ckpt"))
    
    if not checkpoints:
        print(f"ERROR: No checkpoints found")
        return None
    
    latest = max(checkpoints, key=os.path.getctime)
    print(f"Using checkpoint: {latest}")
    return latest


def run_inference_with_reconstructions():
    """Run inference and extract reconstructions from GANomaly generator."""
    
    checkpoint_path = find_checkpoint()
    if checkpoint_path is None:
        return
    
    # Initialize datamodule
    datamodule = Folder(
        name="LAG",
        root=Path("./datasets/LAG_organized"),
        normal_dir="good",
        abnormal_dir="defect",
        eval_batch_size=1,
        num_workers=0,
    )
    
    # Load model
    model = Ganomaly.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    device = next(model.parameters()).device
    print(f"Device: {device}\n")
    
    # Setup datamodule
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    
    # Create output directory
    output_dir = Path("./results/lag_inference_ganomaly/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating predictions...")
    
    # Process each test image
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            image = batch["image"].to(device)
            
            # Get the reconstruction from the model's generator
            # We only need the generated image, not the latent vectors
            try:
                # Encode to get latent
                latent_i = model.model.generator.encoder1(image)
                # Decode to get reconstruction  
                reconstruction = model.model.generator.decoder(latent_i)
            except Exception as e:
                print(f"ERROR extracting reconstruction: {e}")
                print(f"  Image shape: {image.shape}")
                continue
            
            # Get anomaly score
            output = model(image)
            anomaly_score = output.pred_score.cpu().item()
            
            # Convert to numpy
            original = image.cpu().numpy()[0]
            recon = reconstruction.cpu().numpy()[0]
            
            # Denormalize
            if original.max() <= 1.0:
                original = (original * 0.5 + 0.5).clip(0, 1)
                recon = (recon * 0.5 + 0.5).clip(0, 1)
            
            # Resize reconstruction to match original
            # Reconstruction is 448x448, original is 500x500
            from scipy.ndimage import zoom
            h_orig, w_orig = original.shape[1], original.shape[2]
            h_recon, w_recon = recon.shape[1], recon.shape[2]
            zoom_factors = (1, h_orig / h_recon, w_orig / w_recon)
            recon = zoom(recon, zoom_factors, order=1)  # Bilinear interpolation
            
            # Compute reconstruction error
            anomaly_map = np.abs(original - recon).mean(axis=0)
            if anomaly_map.max() > 0:
                anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
            
            # Create images
            original_img = Image.fromarray((original.transpose(1, 2, 0) * 255).astype(np.uint8))
            recon_img = Image.fromarray((recon.transpose(1, 2, 0) * 255).astype(np.uint8))
            anomaly_img = Image.fromarray((anomaly_map * 255).astype(np.uint8))
            
            # Composite
            width = original_img.width
            height = original_img.height
            composite = Image.new('RGB', (width * 3, height))
            composite.paste(original_img, (0, 0))
            composite.paste(recon_img, (width, 0))
            composite.paste(anomaly_img.convert('RGB'), (width * 2, 0))
            
            save_path = output_dir / f"{i:04d}_score_{anomaly_score:.4f}.png"
            composite.save(save_path)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1} images...")
    
    print(f"\n✓ Complete!")
    print(f"Results: ./results/lag_inference_ganomaly/images/")
    print(f"Format: [Original] [Reconstructed] [Anomaly Map]")
    print(f"Filename shows anomaly score (0=normal, 1=anomalous)")


if __name__ == "__main__":
    run_inference_with_reconstructions()
