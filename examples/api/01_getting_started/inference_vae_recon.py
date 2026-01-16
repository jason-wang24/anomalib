"""VAE inference demo: original | reconstruction | anomaly heatmap.

Runs a quick inference on the LAG dataset using the VAE components directly
to produce triptych images similar to GANomaly's visualization.

Outputs are written under results/VAE/recon/."""

from pathlib import Path
import numpy as np
import cv2
import torch

from anomalib.data import Folder
from anomalib.models import Vae
from anomalib.data.utils.image import read_image
from anomalib.utils.post_processing import superimpose_anomaly_map


def to_uint8(img: torch.Tensor) -> np.ndarray:
    """Convert a tensor image in [-1,1] or [0,1] to uint8 HxWxC."""
    if img.dim() == 3:
        # C,H,W -> H,W,C
        img = img.detach().cpu().clamp(min=-1.0, max=1.0)
        # Assume tanh output in [-1,1]; map to [0,1]
        img = (img + 1.0) / 2.0
        img = (img * 255.0).round().byte().permute(1, 2, 0).numpy()
    elif img.dim() == 2:
        img = img.detach().cpu().clamp(min=0.0, max=1.0)
        img = (img * 255.0).round().byte().numpy()
    else:
        raise ValueError("Unsupported image tensor shape")
    return img


def main() -> None:
    lag_root = Path("datasets/LAG_organized").resolve()
    out_dir = Path("results/VAE/recon")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Minimal datamodule for iterating test images
    dm = Folder(
        name="LAG",
        root=lag_root,
        normal_dir="good",
        abnormal_dir="defect",
        train_batch_size=2,
        eval_batch_size=2,
        num_workers=0,
    )
    dm.setup()

    # Use a tiny VAE; load quick-trained weights if available
    model = Vae(batch_size=2, n_features=16, latent_vec_size=32)
    ckpt_path = Path("results/VAE/vae_quick.pth")
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found; using randomly initialized VAE (recon may be gray).")
    model.eval()
    model.freeze()

    # Iterate a few samples
    count = 0
    max_images = 12
    with torch.no_grad():
        for batch in dm.test_dataloader():
            images = batch.image  # BxCxHxW

            # Forward through components to get reconstruction and anomaly map
            mu, log_var = model.model.encoder(images)
            z = model.model.reparameterize(mu, log_var)
            recon = model.model.decoder(z)
            # Resize images to reconstruction size if needed
            if images.shape[-2:] != recon.shape[-2:]:
                images_resized = torch.nn.functional.interpolate(
                    images, size=recon.shape[-2:], mode="bilinear", align_corners=False
                )
            else:
                images_resized = images
            anomaly_map = ((images_resized - recon) ** 2).mean(dim=1, keepdim=True)  # Bx1xHxW

            for i in range(images.size(0)):
                img_path = Path(batch.image_path[i])
                # Read original at native resolution for display
                orig = read_image(img_path)
                if orig.ndim == 2:
                    orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
                if orig.dtype != np.uint8:
                    # Assume [0,1] float, convert to uint8
                    orig = (orig * 255.0).clip(0, 255).astype(np.uint8)

                # Convert recon to uint8 and resize to original
                recon_img = to_uint8(recon[i])
                recon_img = cv2.resize(recon_img, (orig.shape[1], orig.shape[0]))

                # Heatmap overlay from anomaly_map (normalize inside util)
                amap = anomaly_map[i].detach().cpu().squeeze(0).numpy()
                heat = superimpose_anomaly_map(amap.astype(np.float32), orig, normalize=True)

                # Resize originals to heat size to align
                h, w = heat.shape[:2]
                orig_resized = cv2.resize(orig, (w, h))
                recon_resized = cv2.resize(recon_img, (w, h))
                # Concatenate: original | recon | heat
                triptych = np.concatenate([orig_resized, recon_resized, heat], axis=1)

                # Save under mirrored structure
                try:
                    rel = img_path.resolve().relative_to(lag_root)
                except Exception:
                    rel = Path(img_path.name)
                out_path = out_dir / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_file = out_path.with_suffix(".png")
                cv2.imwrite(str(out_file), cv2.cvtColor(triptych, cv2.COLOR_RGB2BGR))

                count += 1
                if count >= max_images:
                    print(f"Saved {count} samples to {out_dir}")
                    return

    print(f"Saved {count} samples to {out_dir}")


if __name__ == "__main__":
    main()
