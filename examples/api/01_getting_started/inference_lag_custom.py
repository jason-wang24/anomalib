"""Run inference on LAG dataset and visualize anomaly detection results.

This script loads a trained model and generates visualizations showing:
- Original images
- Reconstructed/pseudo-normal images (intermediate output from generator)
- Predicted anomaly heatmaps
- Anomaly scores

The visualizations are saved to disk for review.
"""

from pathlib import Path
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from anomalib.data import Folder
from anomalib.models import Ganomaly
from anomalib.engine import Engine


def find_checkpoint():
    """Find the latest checkpoint file in the results directory."""
    results_dir = Path("./results/lag_ganomaly")
    
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        print(f"\nPlease run training first:")
        print(f"  python examples/api/01_getting_started/train_lag_custom.py")
        return None
    
    # Search recursively for .ckpt files
    checkpoints = list(results_dir.rglob("*.ckpt"))
    
    print(f"Searching for checkpoints in: {results_dir}")
    print(f"Found {len(checkpoints)} checkpoint(s)")
    
    if not checkpoints:
        print(f"\nERROR: No checkpoints found")
        print(f"Directory structure:")
        for item in results_dir.rglob("*"):
            if item.is_file():
                print(f"  {item.relative_to(results_dir)}")
        print(f"\nPlease run training first:")
        print(f"  python examples/api/01_getting_started/train_lag_custom.py")
        return None
    
    # Return the most recently modified checkpoint
    latest = max(checkpoints, key=os.path.getctime)
    print(f"Using checkpoint: {latest}")
    return latest


def run_inference():
    """Run inference with trained GANomaly model and save visualizations with reconstructions."""
    
    # Find the latest checkpoint
    checkpoint_path = find_checkpoint()
    if checkpoint_path is None:
        return
    
    # Initialize the same datamodule
    datamodule = Folder(
        name="LAG",
        root=Path("./datasets/LAG_organized"),
        normal_dir="good",
        abnormal_dir="defect",
        eval_batch_size=1,
        num_workers=0,
    )
    
    # Load the trained model from checkpoint
    model = Ganomaly.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Initialize engine for inference
    engine = Engine(
        default_root_dir="./results/lag_inference",
    )
    
    print(f"\nGenerating predictions with reconstructions...")
    
    # Setup datamodule to get test data
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    
    # Create output directory for visualizations
    output_dir = Path("./results/lag_inference_with_recon/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = next(model.parameters()).device
    
    # Process each test image
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            image = batch["image"].to(device)
            
            # Forward pass through model
            output = model(image)
            
            # Handle different output types
            try:
                # Try to get reconstruction from InferenceBatch object
                reconstruction = output.recon_images.cpu()
            except (AttributeError, TypeError):
                try:
                    # Try dict access
                    reconstruction = output.get("recon_images", image).cpu()
                except (AttributeError, TypeError):
                    # Fallback to original
                    reconstruction = image.cpu()
            
            try:
                # Try to get anomaly map
                anomaly_maps = output.pred_scores.cpu()
            except (AttributeError, TypeError):
                try:
                    anomaly_maps = output.get("pred_scores", None)
                    if anomaly_maps is not None:
                        anomaly_maps = anomaly_maps.cpu()
                except (AttributeError, TypeError):
                    anomaly_maps = None
            
            # Convert to numpy and denormalize
            original = image.cpu().numpy()[0]
            recon = reconstruction.numpy()[0]
            
            # Denormalize if needed (assuming ImageNet normalization)
            if original.max() <= 1.0:
                original = (original * 0.5 + 0.5).clip(0, 1)
                recon = (recon * 0.5 + 0.5).clip(0, 1)
            
            # Compute reconstruction error as anomaly score
            if anomaly_maps is None:
                # If model doesn't provide anomaly map, compute it
                anomaly_map = np.abs(original - recon).mean(axis=0)
            else:
                anomaly_map = anomaly_maps.numpy()[0]
            
            # Normalize anomaly map to 0-1
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-6)
            
            # Create visualization with original, reconstruction, and anomaly map
            original_img = Image.fromarray((original.transpose(1, 2, 0) * 255).astype(np.uint8))
            recon_img = Image.fromarray((recon.transpose(1, 2, 0) * 255).astype(np.uint8))
            anomaly_img = Image.fromarray((anomaly_map * 255).astype(np.uint8))
            
            # Create composite image: original | reconstruction | anomaly
            width = original_img.width
            height = original_img.height
            composite = Image.new('RGB', (width * 3, height))
            composite.paste(original_img, (0, 0))
            composite.paste(recon_img, (width, 0))
            composite.paste(anomaly_img.convert('RGB'), (width * 2, 0))
            
            # Save composite
            save_path = output_dir / f"image_{i:04d}_original_recon_anomaly.png"
            composite.save(save_path)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1} images...")
    
    print(f"\n✓ Inference with reconstructions complete!")
    print(f"\n📊 Results saved to: ./results/lag_inference_with_recon/")
    print(f"\nVisualization format (left to right):")
    print(f"  1. Original Image")
    print(f"  2. Reconstructed/Pseudo-Normal Image (what the model thinks is normal)")
    print(f"  3. Anomaly Map (white = normal, black = anomalous)")
    print(f"\nLocation: ./results/lag_inference_with_recon/images/")


if __name__ == "__main__":
    run_inference()
