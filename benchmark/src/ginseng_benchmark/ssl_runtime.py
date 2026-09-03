"""Runtime construction helpers for self-supervised training and extraction."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
from typing import Tuple

import torch
from torch import nn

from ginseng_benchmark.ssl_models import DinoModel, SimSiamModel, VICRegModel
from ginseng_benchmark.ssl_training import SslModelSpec


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def seed_torch_checkpoint_cache(
    target: Path,
    source: Path,
    *,
    filenames,
) -> Tuple[Path, ...]:
    """Copy verified legacy Torch weights into the project-local model cache."""
    target = Path(target)
    source = Path(source)
    copied = []
    if not source.is_dir():
        return tuple()
    target.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        match = re.search(r"-([0-9a-f]{8})\.[^.]+$", filename)
        if not match:
            raise ValueError(f"checkpoint filename lacks a hash prefix: {filename}")
        source_path = source / filename
        target_path = target / filename
        if target_path.is_file():
            continue
        if not source_path.is_file():
            continue
        digest = _digest(source_path)
        if not digest.startswith(match.group(1)):
            raise ValueError(f"legacy checkpoint checksum mismatch: {source_path}")
        temporary = target_path.with_suffix(target_path.suffix + ".tmp")
        shutil.copyfile(source_path, temporary)
        os.replace(temporary, target_path)
        copied.append(target_path)
    return tuple(copied)


def configure_model_cache(root: Path) -> Path:
    root = Path(root).resolve()
    torch_home = root / "torch"
    hf_home = root / "huggingface"
    torch_home.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    seed_torch_checkpoint_cache(
        torch_home / "checkpoints",
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
        filenames=(
            "resnet50-11ad3fa6.pth",
            "resnet50-0676ba61.pth",
        ),
    )
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    torch.hub.set_dir(str(torch_home))
    return root


def build_backbone(
    name: str,
    *,
    pretrained: bool,
) -> Tuple[nn.Module, int]:
    if name == "resnet50":
        from torchvision import models

        if pretrained:
            try:
                weights = models.ResNet50_Weights.IMAGENET1K_V2
                encoder = models.resnet50(weights=weights)
            except AttributeError:
                encoder = models.resnet50(pretrained=True)
        else:
            try:
                encoder = models.resnet50(weights=None)
            except TypeError:
                encoder = models.resnet50(pretrained=False)
        dimension = int(encoder.fc.in_features)
        encoder.fc = nn.Identity()
        return encoder, dimension

    if name == "vit_small_patch16_224":
        import timm

        encoder = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        dimension = int(getattr(encoder, "num_features", 384))
        return encoder, dimension

    raise ValueError(f"unsupported self-supervised backbone: {name}")


def build_ssl_model(
    spec: SslModelSpec,
    *,
    pretrained: bool,
) -> nn.Module:
    encoder, dimension = build_backbone(spec.backbone, pretrained=pretrained)
    if dimension != spec.feature_dim:
        raise ValueError(
            f"configured feature_dim={spec.feature_dim} does not match "
            f"{spec.backbone} output={dimension}"
        )
    if spec.algorithm == "simsiam":
        return SimSiamModel(encoder, feature_dim=dimension)
    if spec.algorithm == "vicreg":
        return VICRegModel(encoder, feature_dim=dimension)
    if spec.algorithm == "dino":
        return DinoModel(
            encoder,
            feature_dim=dimension,
            output_dim=8192,
            hidden_dim=2048,
            bottleneck_dim=256,
        )
    raise ValueError(f"unsupported self-supervised algorithm: {spec.algorithm}")
