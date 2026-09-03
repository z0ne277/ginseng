#!/usr/bin/env python
"""Train image-only self-supervised baselines with visible progress."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Optional, Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ginseng_benchmark.ssl_models import update_dino_center
from ginseng_benchmark.ssl_runtime import (
    build_ssl_model,
    configure_model_cache,
)
from ginseng_benchmark.ssl_training import (
    dino_cross_view_loss,
    load_ssl_config,
    negative_cosine_similarity,
    validate_image_only_csv,
    vicreg_loss,
)


class ImageOnlyMultiViewDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = tuple(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            return self.transform(image)


class PairTransform:
    def __init__(self, *, image_size: int = 224):
        from torchvision import transforms

        color_jitter = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.15,
            hue=0.05,
        )
        self.view = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.2, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomApply([color_jitter], p=0.6),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                    p=0.5,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __call__(self, image):
        return self.view(image), self.view(image)


class DinoMultiCropTransform:
    def __init__(self, *, image_size: int = 224, local_crops: int = 4):
        from torchvision import transforms

        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        appearance = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.15,
                            hue=0.05,
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                    p=0.5,
                ),
                normalize,
            ]
        )
        self.global_view = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.4, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                appearance,
            ]
        )
        # Local content is cropped more tightly, then resized to the fixed ViT input.
        self.local_view = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.08, 0.4),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                appearance,
            ]
        )
        self.local_crops = local_crops

    def __call__(self, image):
        return [
            self.global_view(image),
            self.global_view(image),
            *[self.local_view(image) for _ in range(self.local_crops)],
        ]


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--dev-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-train-count", type=int, default=6425)
    parser.add_argument("--expected-val-count", type=int, default=802)
    parser.add_argument("--expected-dev-count", type=int, default=804)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _select_spec(config_path: Path, model_id: str):
    config = load_ssl_config(config_path)
    selected = [item for item in config.models if item.model_id == model_id]
    if len(selected) != 1:
        raise ValueError(f"unknown self-supervised model id: {model_id}")
    return selected[0]


def _atomic_torch_save(payload, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)


def _loss_for_batch(model, algorithm, batch, device, *, training: bool):
    views = [item.to(device, non_blocking=True) for item in batch]
    if algorithm == "simsiam":
        first_prediction, second_prediction, first_target, second_target = model(
            views[0], views[1]
        )
        loss = 0.5 * (
            negative_cosine_similarity(first_prediction, second_target)
            + negative_cosine_similarity(second_prediction, first_target)
        )
        return loss, {}
    if algorithm == "vicreg":
        first_projection, second_projection = model(views[0], views[1])
        return vicreg_loss(first_projection, second_projection)
    if algorithm == "dino":
        student_logits, teacher_logits = model(views)
        loss = dino_cross_view_loss(
            student_logits,
            teacher_logits,
            center=model.center,
            student_temperature=0.1,
            teacher_temperature=0.04,
        )
        if training:
            with torch.no_grad():
                combined_teacher = torch.cat(teacher_logits, dim=0)
                model.center.copy_(
                    update_dino_center(
                        model.center,
                        combined_teacher,
                        momentum=0.9,
                    )
                )
        return loss, {}
    raise ValueError(f"unsupported algorithm: {algorithm}")


def _run_epoch(
    model,
    loader,
    *,
    algorithm,
    device,
    optimizer,
    scaler,
    epoch,
    epochs,
    max_batches,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    count = 0
    start = time.time()
    limit = min(len(loader), max_batches) if max_batches else len(loader)
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > limit:
                break
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                loss, _ = _loss_for_batch(
                    model,
                    algorithm,
                    batch,
                    device,
                    training=training,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {algorithm} loss")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if algorithm == "dino":
                    model.update_teacher(momentum=0.996)
            total_loss += float(loss.detach().cpu())
            count += 1
            if batch_index == 1 or batch_index % 20 == 0 or batch_index == limit:
                elapsed = max(time.time() - start, 1e-6)
                rate = batch_index / elapsed
                remaining = (limit - batch_index) / max(rate, 1e-6)
                split = "train" if training else "eval"
                print(
                    f"[{split}] epoch={epoch}/{epochs} "
                    f"batch={batch_index}/{limit} "
                    f"loss={total_loss/count:.6f} "
                    f"elapsed={elapsed:.0f}s eta={remaining:.0f}s",
                    flush=True,
                )
    if count == 0:
        raise RuntimeError("data loader produced no batches")
    return total_loss / count


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        spec = _select_spec(args.config, args.model_id)
        train_paths = validate_image_only_csv(args.train_csv)
        val_paths = validate_image_only_csv(args.val_csv)
        dev_paths = validate_image_only_csv(args.dev_csv)
        expected = (
            ("train", len(train_paths), args.expected_train_count),
            ("val", len(val_paths), args.expected_val_count),
            ("dev", len(dev_paths), args.expected_dev_count),
        )
        mismatches = [
            f"{name} expected={wanted} found={found}"
            for name, found, wanted in expected
            if found != wanted
        ]
        if mismatches:
            raise ValueError("CSV count mismatch: " + "; ".join(mismatches))
        _set_seed(args.seed)
        configure_model_cache(args.model_cache)
        device = _resolve_device(args.device)
        print(
            f"model={spec.model_id} algorithm={spec.algorithm} "
            f"backbone={spec.backbone} device={device} "
            f"counts=train:{len(train_paths)},val:{len(val_paths)},dev:{len(dev_paths)}",
            flush=True,
        )
        transform = (
            DinoMultiCropTransform()
            if spec.algorithm == "dino"
            else PairTransform()
        )
        train_loader = DataLoader(
            ImageOnlyMultiViewDataset(train_paths, transform),
            batch_size=spec.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
        val_loader = DataLoader(
            ImageOnlyMultiViewDataset(val_paths, transform),
            batch_size=spec.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
        dev_loader = DataLoader(
            ImageOnlyMultiViewDataset(dev_paths, transform),
            batch_size=spec.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
        model = build_ssl_model(spec, pretrained=True).to(device)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=spec.learning_rate,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=spec.epochs,
            eta_min=spec.learning_rate * 0.01,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        best_path = args.output_dir / "best.pt"
        last_path = args.output_dir / "last.pt"
        history_path = args.output_dir / "history.json"
        start_epoch = 1
        best_val = math.inf
        history = []
        if args.resume and last_path.is_file():
            checkpoint = torch.load(last_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_val = float(checkpoint["best_val_loss"])
            history = list(checkpoint.get("history", []))
            print(f"resumed_from={last_path} start_epoch={start_epoch}", flush=True)
        for epoch in range(start_epoch, spec.epochs + 1):
            train_loss = _run_epoch(
                model,
                train_loader,
                algorithm=spec.algorithm,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                epochs=spec.epochs,
                max_batches=args.max_batches,
            )
            val_loss = _run_epoch(
                model,
                val_loader,
                algorithm=spec.algorithm,
                device=device,
                optimizer=None,
                scaler=scaler,
                epoch=epoch,
                epochs=spec.epochs,
                max_batches=args.max_batches,
            )
            scheduler.step()
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            payload = {
                "schema_version": 1,
                "model_id": spec.model_id,
                "algorithm": spec.algorithm,
                "backbone": spec.backbone,
                "feature_dim": spec.feature_dim,
                "epoch": epoch,
                "best_val_loss": min(best_val, val_loss),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "seed": args.seed,
                "information_condition": "image-only; no identity labels",
            }
            _atomic_torch_save(payload, last_path)
            if val_loss < best_val:
                best_val = val_loss
                payload["best_val_loss"] = best_val
                _atomic_torch_save(payload, best_path)
            history_path.write_text(
                json.dumps(history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"[epoch] {epoch}/{spec.epochs} train={train_loss:.6f} "
                f"val={val_loss:.6f} best={best_val:.6f}",
                flush=True,
            )
            if args.max_batches:
                print("max-batches smoke run completed after one epoch", flush=True)
                break
        if not best_path.is_file():
            raise RuntimeError("training did not produce best.pt")
        best = torch.load(best_path, map_location=device)
        model.load_state_dict(best["model_state_dict"])
        dev_loss = _run_epoch(
            model,
            dev_loader,
            algorithm=spec.algorithm,
            device=device,
            optimizer=None,
            scaler=scaler,
            epoch=int(best["epoch"]),
            epochs=spec.epochs,
            max_batches=args.max_batches,
        )
        print(
            f"training_complete model={spec.model_id} best_epoch={best['epoch']} "
            f"best_val={best_val:.6f} dev_loss={dev_loss:.6f} checkpoint={best_path}",
            flush=True,
        )
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"self-supervised training failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
