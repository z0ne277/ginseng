#!/usr/bin/env python
"""Extract normalized gallery features from a trained SSL checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Optional, Sequence

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from ginseng_benchmark.ssl_runtime import build_ssl_model, configure_model_cache
from ginseng_benchmark.ssl_training import load_ssl_config
from ginseng_benchmark.vision_features import discover_flat_image_paths


class GalleryDataset(Dataset):
    def __init__(self, paths):
        from torchvision import transforms

        self.paths = tuple(paths)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    256,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-count", type=int, default=12787)
    parser.add_argument("--device", default="auto")
    return parser


def _device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        config = load_ssl_config(args.config)
        matches = [item for item in config.models if item.model_id == args.model_id]
        if len(matches) != 1:
            raise ValueError(f"unknown self-supervised model id: {args.model_id}")
        spec = matches[0]
        paths = discover_flat_image_paths(args.image_dir)
        if len(paths) != args.expected_count:
            raise ValueError(
                f"gallery count mismatch: expected {args.expected_count}, found {len(paths)}"
            )
        configure_model_cache(args.model_cache)
        device = _device(args.device)
        model = build_ssl_model(spec, pretrained=False)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if checkpoint.get("model_id") != spec.model_id:
            raise ValueError("checkpoint model id does not match requested model")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        loader = DataLoader(
            GalleryDataset(paths),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        chunks = []
        start = time.time()
        processed = 0
        with torch.inference_mode():
            for batch_index, images in enumerate(loader, start=1):
                images = images.to(device, non_blocking=True)
                features = torch.nn.functional.normalize(model.encode(images), dim=-1)
                chunks.append(features.cpu())
                processed += images.shape[0]
                if batch_index == 1 or batch_index % 20 == 0 or processed == len(paths):
                    elapsed = max(time.time() - start, 1e-6)
                    rate = processed / elapsed
                    eta = (len(paths) - processed) / max(rate, 1e-6)
                    print(
                        f"[extract] {processed}/{len(paths)} "
                        f"({processed/len(paths)*100:.1f}%) "
                        f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                        flush=True,
                    )
        matrix = torch.cat(chunks, dim=0)
        if matrix.shape != (len(paths), spec.feature_dim):
            raise ValueError(
                f"feature shape mismatch: expected {(len(paths), spec.feature_dim)}, "
                f"found {tuple(matrix.shape)}"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": matrix,
                "paths": [str(path) for path in paths],
                "model_id": spec.model_id,
                "checkpoint_epoch": checkpoint.get("epoch"),
            },
            args.output,
        )
        print(
            f"extraction_complete model={spec.model_id} "
            f"count={matrix.shape[0]} dim={matrix.shape[1]} output={args.output}",
            flush=True,
        )
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"self-supervised extraction failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
