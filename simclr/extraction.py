"""Feature extraction script for SimCLR v2 on the ginseng retrieval gallery.

Usage example:
  python extraction.py \
    --image-dir E:/path/to/gallery_root \
    --model-path model_epoch_200(1).pth \
    --output-feats cache/gallery_feats_simclr.pt

This follows the same conventions as hybrid_v2/hybrid_v3:
- ImageNet preprocessing
- L2-normalized projection features via SimCLR_V2.get_features
- Saves dict: {'features': tensor [N,D], 'paths': list[str]}
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from simclr_v2_model import SimCLR_V2


IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class GalleryDataset(Dataset):
    def __init__(self, root_dir: str, image_size: int = 224):
        self.root_dir = root_dir
        self.image_size = image_size
        self.paths: List[str] = []
        for dirpath, _, filenames in os.walk(root_dir):
            for name in filenames:
                if name.lower().endswith(IMG_SUFFIXES):
                    self.paths.append(os.path.join(dirpath, name))
        self.paths.sort()

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, path


def build_model(model_path: str, device: torch.device) -> SimCLR_V2:
    model = SimCLR_V2(
        base_model="resnet50",
        feature_dim=256,
        temperature=0.1,
        device=device,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)

    # SimCLR checkpoints may be either raw state_dict or wrapped.
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state)}")

    # Strip DDP prefix.
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.removeprefix("module."): v for k, v in state.items()}

    def remap_projector_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Map old `projector.*` names to `projection_head.*` for compatibility."""
        remapped: Dict[str, torch.Tensor] = {}
        for key, value in sd.items():
            new_key = key
            if key.startswith("projector.0."):
                new_key = "projection_head.0." + key[len("projector.0."):]
            elif key.startswith("projector.2."):
                # Old 2-layer head last linear -> new 3-layer head last linear
                new_key = "projection_head.6." + key[len("projector.2."):]
            elif key.startswith("projector."):
                new_key = "projection_head." + key[len("projector."):]
            remapped[new_key] = value
        return remapped

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        remapped_state = remap_projector_keys(state)
        load_info = model.load_state_dict(remapped_state, strict=False)
        if load_info.missing_keys:
            print(f"[load_state_dict] missing keys: {load_info.missing_keys}")
        if load_info.unexpected_keys:
            print(f"[load_state_dict] unexpected keys: {load_info.unexpected_keys}")

    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract gallery features with SimCLR v2")
    p.add_argument("--image-dir", required=True, help="Root folder of gallery images")
    p.add_argument("--model-path", required=True, help="Path to trained SimCLR v2 checkpoint (.pth)")
    p.add_argument("--output-feats", required=True, help="Output .pt file for features")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Gallery root: {args.image_dir}")
    print(f"Model path: {args.model_path}")

    dataset = GalleryDataset(args.image_dir, image_size=args.image_size)
    if len(dataset) == 0:
        print("No images found, aborting.")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args.model_path, device)

    all_feats: List[torch.Tensor] = []
    all_paths: List[str] = []

    with torch.no_grad():
        for i, (images, paths) in enumerate(loader):
            images = images.to(device)
            feats = model.get_features(images)
            all_feats.append(feats.cpu())
            all_paths.extend(paths)

            if (i + 1) % 10 == 0 or (i + 1) == len(loader):
                print(f"[Batch {i + 1}/{len(loader)}] processed {len(all_paths)} images")

    feats_tensor = torch.cat(all_feats, dim=0)
    assert feats_tensor.shape[0] == len(all_paths)

    out_path = Path(args.output_feats)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "features": feats_tensor,
        "paths": all_paths,
    }, out_path)

    print(f"Saved features: {out_path}")
    print(f"  Num images: {len(all_paths)}")
    print(f"  Feature shape: {feats_tensor.shape}")


if __name__ == "__main__":
    main()
