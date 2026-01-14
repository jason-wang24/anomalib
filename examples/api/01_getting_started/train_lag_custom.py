"""Train an anomaly detection model on the LAG custom dataset.

This example demonstrates how to train an anomaly detection model on your
custom LAG dataset that has been reorganized using convert_lag_dataset.py.

The LAG dataset structure should be:
    datasets/LAG_organized/
    ├── train/
    │   └── good/           (normal training images)
    ├── test/
    │   ├── good/           (normal test images)
    │   └── defect/         (anomalous test images)
"""

from pathlib import Path
import os

from anomalib.data import Folder
from anomalib.models import Ganomaly
from anomalib.engine import Engine

# Workaround for Windows symlink permission issues
# Patch the create_versioned_dir function to not create symlinks on Windows
if os.name == "nt":  # Windows
    import anomalib.utils.path as path_utils
    original_create_versioned_dir = path_utils.create_versioned_dir
    
    def patched_create_versioned_dir(root_dir):
        """Create versioned directory without symlinks on Windows."""
        root_dir = Path(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)
        
        # Just return the root dir without trying to create symlinks
        return root_dir
    
    path_utils.create_versioned_dir = patched_create_versioned_dir


def train_lag_dataset():
    """Train Patchcore model on LAG dataset."""
    
    # Initialize the data module with your LAG dataset
    # All normal images in good/, all anomalous in defect/
    # Anomalib will automatically split them for train/val/test
    datamodule = Folder(
        name="LAG",
        root=Path("./datasets/LAG_organized"),
        normal_dir="good",
        abnormal_dir="defect",
        train_batch_size=8,        # Reduced from 32 for faster training
        eval_batch_size=8,         # Reduced from 32
        num_workers=0,             # Disabled for Windows/faster iteration
        # Split configuration
        normal_split_ratio=0.8,    # 80% of normal images for training
        test_split_mode="from_dir", # Use remaining for test
        test_split_ratio=0.5,      # 50% of remaining normal for test
    )
    
    # Initialize the model
    model = Ganomaly()
    
    # Initialize the engine
    engine = Engine(
        max_epochs=50,  # Increased - GANomaly needs proper training
        accelerator="auto",
        devices=1,
        default_root_dir="./results/lag_ganomaly",
    )
    
    # Train the model
    # Note: versioned_dir=False to avoid Windows symlink permission issues
    engine.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=None,
    )


if __name__ == "__main__":
    train_lag_dataset()
