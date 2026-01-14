#!/usr/bin/env python
"""Convert LAG dataset to anomalib folder structure.

The LAG dataset has all images in one folder with a data.json file that maps
image names to labels (0=normal, 1=anomaly). This script reorganizes the
images into the folder structure expected by anomalib's Folder datamodule:

    LAG_organized/
    ├── train/
    │   └── good/           (normal training images)
    ├── test/
    │   ├── good/           (normal test images)
    │   └── defect/         (anomalous test images)
"""

import json
import shutil
from pathlib import Path


def convert_lag_dataset(
    input_dir: str | Path,
    output_dir: str | Path = "./datasets/LAG_organized",
) -> None:
    """Convert LAG dataset to anomalib Folder datamodule format.
    
    Creates structure expected by Folder datamodule:
        output_dir/
        ├── good/           (all normal images from train + test/0)
        └── defect/         (all anomalous images from test/1)
    
    Args:
        input_dir: Path to LAG dataset folder containing images and data.json
        output_dir: Path where the reorganized dataset will be created
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Read data.json
    data_json_path = input_dir / "data.json"
    with open(data_json_path, "r") as f:
        data = json.load(f)
    
    # Create output directories in the format Folder datamodule expects
    good_dir = output_dir / "good"
    defect_dir = output_dir / "defect"
    
    good_dir.mkdir(parents=True, exist_ok=True)
    defect_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting LAG dataset from {input_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Process training data
    print("Processing training data...")
    train_data = data.get("train", {})
    
    # Copy normal training images to good/
    normal_train = train_data.get("0", [])
    for i, filename in enumerate(normal_train, 1):
        # Try both with and without images/ subfolder
        src = input_dir / "images" / filename
        if not src.exists():
            src = input_dir / filename
        
        dst = good_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"  Warning: {filename} not found")
        if i % 100 == 0:
            print(f"  Copied {i}/{len(normal_train)} normal training images")
    
    print(f"  ✓ Copied {len(normal_train)} normal training images\n")
    
    # Process test data
    print("Processing test data...")
    test_data = data.get("test", {})
    
    # Copy normal test images to good/
    normal_test = test_data.get("0", [])
    for i, filename in enumerate(normal_test, 1):
        # Try both with and without images/ subfolder
        src = input_dir / "images" / filename
        if not src.exists():
            src = input_dir / filename
        
        dst = good_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"  Warning: {filename} not found")
        if i % 100 == 0:
            print(f"  Copied {i}/{len(normal_test)} normal test images")
    
    print(f"  ✓ Copied {len(normal_test)} normal test images")
    
    # Copy anomalous test images to defect/
    anomaly_test = test_data.get("1", [])
    for i, filename in enumerate(anomaly_test, 1):
        # Try both with and without images/ subfolder
        src = input_dir / "images" / filename
        if not src.exists():
            src = input_dir / filename
        
        dst = defect_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"  Warning: {filename} not found")
        if i % 100 == 0:
            print(f"  Copied {i}/{len(anomaly_test)} anomalous test images")
    
    print(f"  ✓ Copied {len(anomaly_test)} anomalous test images\n")
    
    # Print summary
    print("=" * 50)
    print("CONVERSION COMPLETE")
    print("=" * 50)
    total_normal = len(normal_train) + len(normal_test)
    total_anomaly = len(anomaly_test)
    print(f"Total normal images: {total_normal}")
    print(f"Total anomalous images: {total_anomaly}")
    print(f"\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"  ├── good/           ({total_normal} images)")
    print(f"  └── defect/         ({total_anomaly} images)")
    print(f"\nReady to use with anomalib!")
    print(f"\nExample usage:")
    print(f"  from anomalib.data import Folder")
    print(f"  datamodule = Folder(")
    print(f"      name='lag',")
    print(f"      root=Path('{output_dir}'),")
    print(f"      normal_dir='good',")
    print(f"      abnormal_dir='defect',")
    print(f"      train_batch_size=32,")
    print(f"      eval_batch_size=32,")
    print(f"  )")


if __name__ == "__main__":
    import sys
    
    # Get input directory from command line or use default
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = "./datasets/LAG"
    
    # Get output directory if provided
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./datasets/LAG_organized"
    
    convert_lag_dataset(input_dir, output_dir)
