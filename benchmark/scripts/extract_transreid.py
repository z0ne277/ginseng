#!/usr/bin/env python
"""Extract features from an explicitly supplied official TransReID checkpoint."""

import argparse
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Mapping, Optional, Sequence

from ginseng_benchmark.vision_features import (
    discover_flat_image_paths,
    l2_normalize_rows,
    validate_feature_collection,
)


PINNED_COMMIT = "dec55046fcdfadee14e2c28e2df89305d8f7557a"


@dataclass(frozen=True)
class CheckpointLayout:
    num_classes: int
    in_planes: int
    sie_embeddings: int


def _state_dict(payload: Mapping[str, object]) -> Mapping[str, object]:
    nested = payload.get("state_dict") if isinstance(payload, dict) else None
    value = nested if isinstance(nested, dict) else payload
    if not isinstance(value, dict) or not value:
        raise ValueError("checkpoint must contain a non-empty state dict")
    return {key.removeprefix("module."): tensor for key, tensor in value.items()}


def infer_checkpoint_layout(payload: Mapping[str, object]) -> CheckpointLayout:
    state = _state_dict(payload)
    classifier = state.get("classifier.weight")
    if classifier is None or getattr(classifier, "ndim", None) != 2:
        raise ValueError("checkpoint lacks a two-dimensional classifier.weight")
    num_classes, in_planes = map(int, classifier.shape)
    if num_classes <= 0 or in_planes <= 0:
        raise ValueError("checkpoint classifier has an invalid shape")
    sie = state.get("base.sie_embed")
    sie_embeddings = 0 if sie is None else int(sie.shape[0])
    return CheckpointLayout(num_classes, in_planes, sie_embeddings)


def expected_transreid_dim(*, in_planes: int, jpm: bool) -> int:
    if in_planes <= 0:
        raise ValueError("in_planes must be positive")
    return in_planes * (5 if jpm else 1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract official TransReID features.")
    parser.add_argument("--transreid-root", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--expected-count", type=int, default=12_787)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--camera-count", type=int)
    parser.add_argument("--view-count", type=int)
    parser.add_argument("--trusted-local-checkpoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_checkpoint(path: Path):
    import torch

    arguments = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        arguments["weights_only"] = False
    payload = torch.load(path, **arguments)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a dict")
    return payload


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError("TransReID root is not a readable Git checkout")
    return completed.stdout.strip()


def _atomic_json(payload, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_torch_save(payload, output: Path) -> None:
    import torch

    if output.suffix.casefold() != ".pt":
        raise ValueError("raw feature output must use the .pt suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _collate(batch):
    return batch


class _Dataset:
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from PIL import Image

        path = self.paths[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), str(path)


def _resolve_sie_counts(cfg, layout: CheckpointLayout, camera_count, view_count):
    use_camera = bool(cfg.MODEL.SIE_CAMERA)
    use_view = bool(cfg.MODEL.SIE_VIEW)
    if not use_camera and not use_view:
        return 0, 0
    if use_camera and use_view:
        if not camera_count or not view_count:
            raise ValueError("config enables camera and view SIE; provide both counts")
        if camera_count * view_count != layout.sie_embeddings:
            raise ValueError("camera/view counts do not match checkpoint SIE embeddings")
        return camera_count, view_count
    count = camera_count if use_camera else view_count
    if count is None:
        count = layout.sie_embeddings
    if count != layout.sie_embeddings or count <= 0:
        raise ValueError("SIE count does not match checkpoint embeddings")
    return (count, 0) if use_camera else (0, count)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        if not args.trusted_local_checkpoint:
            raise ValueError("pass --trusted-local-checkpoint for an explicitly trusted checkpoint")
        root = args.transreid_root.resolve(strict=True)
        config_file = args.config_file.resolve(strict=True)
        checkpoint = args.checkpoint.resolve(strict=True)
        head = _git_head(root)
        if head != PINNED_COMMIT:
            raise ValueError(f"TransReID commit mismatch: expected {PINNED_COMMIT}, found {head}")
        paths = discover_flat_image_paths(args.image_dir)
        if len(paths) != args.expected_count:
            raise ValueError(
                f"gallery count mismatch: expected {args.expected_count}, found {len(paths)}"
            )
        if args.dry_run:
            print(
                f"TransReID extraction dry-run: commit={head}, images={len(paths)}, "
                f"config={config_file.name}, checkpoint={checkpoint.name}"
            )
            return 0

        import torch
        from torch.utils.data import DataLoader
        from torchvision import transforms

        payload = _load_checkpoint(checkpoint)
        layout = infer_checkpoint_layout(payload)
        sys.path.insert(0, str(root))
        try:
            from config import cfg as official_cfg
            from model import make_model
        finally:
            sys.path.pop(0)
        cfg = official_cfg.clone()
        cfg.merge_from_file(str(config_file))
        cfg.MODEL.PRETRAIN_CHOICE = "self"
        cfg.freeze()
        camera_count, view_count = _resolve_sie_counts(
            cfg, layout, args.camera_count, args.view_count
        )
        model = make_model(
            cfg, num_class=layout.num_classes,
            camera_num=camera_count, view_num=view_count,
        )
        cleaned_state = _state_dict(payload)
        model.load_state_dict(cleaned_state, strict=True)
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if args.device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is unavailable")
            device = torch.device(args.device)
        model.eval().to(device)
        mean = [float(value) for value in cfg.INPUT.PIXEL_MEAN]
        std = [float(value) for value in cfg.INPUT.PIXEL_STD]
        size = [int(value) for value in cfg.INPUT.SIZE_TEST]
        transform = transforms.Compose(
            [transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        loader = DataLoader(
            _Dataset(paths, transform), batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=_collate,
            pin_memory=device.type == "cuda",
        )
        batches = []
        output_paths = []
        with torch.inference_mode():
            for batch in loader:
                images = torch.stack([item[0] for item in batch]).to(device, non_blocking=True)
                output_paths.extend(item[1] for item in batch)
                zeros = torch.zeros(images.shape[0], dtype=torch.long, device=device)
                features = model(images, cam_label=zeros, view_label=zeros)
                if not torch.is_tensor(features):
                    raise ValueError("TransReID eval forward did not return a tensor")
                batches.append(l2_normalize_rows(features).cpu())
        features = torch.cat(batches, dim=0).contiguous()
        expected_dim = expected_transreid_dim(
            in_planes=layout.in_planes, jpm=bool(cfg.MODEL.JPM)
        )
        validate_feature_collection(features, paths, expected_dim)
        metadata = {
            "official_commit": head,
            "config_name": config_file.name,
            "checkpoint_name": checkpoint.name,
            "checkpoint_sha256": _sha256(checkpoint),
            "feature_dim": expected_dim,
            "preprocessing": {
                "resize": size,
                "mean": mean,
                "std": std,
                "source": "official TransReID config",
            },
            "tta": {"enabled": False, "weights": [1.0]},
            "sie_camera": bool(cfg.MODEL.SIE_CAMERA),
            "sie_view": bool(cfg.MODEL.SIE_VIEW),
            "jpm": bool(cfg.MODEL.JPM),
        }
        _atomic_torch_save(
            {"features": features, "paths": output_paths, "metadata": metadata}, args.output
        )
        metadata_output = args.metadata_output or args.output.with_suffix(".metadata.json")
        _atomic_json(metadata, metadata_output)
        print(
            f"TransReID extraction passed: count={features.shape[0]}, "
            f"dim={features.shape[1]}, commit={head}"
        )
        return 0
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"TransReID extraction failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
