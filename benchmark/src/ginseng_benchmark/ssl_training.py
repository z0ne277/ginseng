"""Core utilities shared by image-only self-supervised baselines."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F


_ALGORITHMS = frozenset({"simsiam", "vicreg", "dino"})


@dataclass(frozen=True)
class SslModelSpec:
    model_id: str
    algorithm: str
    backbone: str
    conda_env: str
    feature_dim: int
    requires_identity_labels: bool
    batch_size: int
    epochs: int
    learning_rate: float


@dataclass(frozen=True)
class SslConfig:
    schema_version: int
    protocol_tag: str
    models: Tuple[SslModelSpec, ...]


def load_ssl_config(path: Path) -> SslConfig:
    """Load the training matrix and reject methods that need identity labels."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported self-supervised configuration schema")
    if payload.get("protocol_tag") != "271_1075_unlabeled":
        raise ValueError("self-supervised configuration uses the wrong protocol")

    models = []
    seen = set()
    for raw in payload.get("models", []):
        model_id = str(raw.get("id", "")).strip()
        algorithm = str(raw.get("algorithm", "")).strip().lower()
        if not model_id or model_id in seen:
            raise ValueError("model ids must be non-empty and unique")
        if bool(raw.get("requires_identity_labels", False)):
            raise ValueError(
                f"{model_id} requires identity labels and cannot enter the unlabeled protocol"
            )
        if algorithm not in _ALGORITHMS:
            raise ValueError(f"unsupported unlabeled algorithm: {algorithm}")
        feature_dim = int(raw.get("feature_dim", 0))
        batch_size = int(raw.get("batch_size", 0))
        epochs = int(raw.get("epochs", 0))
        learning_rate = float(raw.get("learning_rate", 0.0))
        if min(feature_dim, batch_size, epochs) <= 0 or learning_rate <= 0:
            raise ValueError(f"invalid numeric training setting for {model_id}")
        backbone = str(raw.get("backbone", "")).strip()
        conda_env = str(raw.get("conda_env", "")).strip()
        if not backbone or not conda_env:
            raise ValueError(f"missing backbone or conda environment for {model_id}")
        models.append(
            SslModelSpec(
                model_id=model_id,
                algorithm=algorithm,
                backbone=backbone,
                conda_env=conda_env,
                feature_dim=feature_dim,
                requires_identity_labels=False,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
            )
        )
        seen.add(model_id)
    if not models:
        raise ValueError("self-supervised configuration contains no models")
    return SslConfig(
        schema_version=1,
        protocol_tag="271_1075_unlabeled",
        models=tuple(models),
    )


def validate_image_only_csv(
    path: Path,
    *,
    require_files: bool = True,
) -> Tuple[Path, ...]:
    """Validate the exact information condition used by the training protocol."""
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["image"]:
            raise ValueError("training CSV must contain only the image column")
        images = []
        seen = set()
        for row_number, row in enumerate(reader, start=2):
            value = str(row.get("image", "")).strip()
            if not value:
                raise ValueError(f"empty image path at CSV row {row_number}")
            image = Path(value)
            key = str(image.resolve(strict=False)).casefold()
            if key in seen:
                raise ValueError(f"duplicate image path at CSV row {row_number}")
            if require_files and not image.is_file():
                raise ValueError(f"image path does not exist at CSV row {row_number}: {image}")
            seen.add(key)
            images.append(image)
    if not images:
        raise ValueError("training CSV contains no image rows")
    return tuple(images)


def negative_cosine_similarity(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """SimSiam stop-gradient negative cosine objective."""
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return -(prediction * target).sum(dim=-1).mean()


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    rows, columns = matrix.shape
    if rows != columns:
        raise ValueError("covariance matrix must be square")
    return matrix.flatten()[:-1].view(rows - 1, rows + 1)[:, 1:].flatten()


def vicreg_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    invariance_weight: float = 25.0,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
    epsilon: float = 1e-4,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """VICReg invariance, variance and covariance terms."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("VICReg projections must be matching two-dimensional tensors")
    if first.shape[0] < 2:
        raise ValueError("VICReg requires at least two samples per batch")

    invariance = F.mse_loss(first, second)
    centered_first = first - first.mean(dim=0)
    centered_second = second - second.mean(dim=0)
    std_first = torch.sqrt(centered_first.var(dim=0, unbiased=True) + epsilon)
    std_second = torch.sqrt(centered_second.var(dim=0, unbiased=True) + epsilon)
    variance = 0.5 * (
        F.relu(1.0 - std_first).mean() + F.relu(1.0 - std_second).mean()
    )

    denominator = first.shape[0] - 1
    covariance_first = centered_first.T @ centered_first / denominator
    covariance_second = centered_second.T @ centered_second / denominator
    covariance = (
        _off_diagonal(covariance_first).pow(2).sum()
        + _off_diagonal(covariance_second).pow(2).sum()
    ) / first.shape[1]
    components = {
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
    }
    total = (
        invariance_weight * invariance
        + variance_weight * variance
        + covariance_weight * covariance
    )
    return total, components


def dino_cross_view_loss(
    student_logits: Sequence[torch.Tensor],
    teacher_logits: Sequence[torch.Tensor],
    *,
    center: torch.Tensor,
    student_temperature: float,
    teacher_temperature: float,
) -> torch.Tensor:
    """DINO cross-entropy over non-matching student/teacher views."""
    if len(student_logits) < 2 or len(teacher_logits) < 2:
        raise ValueError("DINO requires at least two student and teacher views")
    teacher_probabilities = [
        F.softmax((logits.detach() - center) / teacher_temperature, dim=-1)
        for logits in teacher_logits
    ]
    student_log_probabilities = [
        F.log_softmax(logits / student_temperature, dim=-1)
        for logits in student_logits
    ]
    terms = []
    for teacher_index, teacher_probability in enumerate(teacher_probabilities):
        for student_index, student_log_probability in enumerate(
            student_log_probabilities
        ):
            if student_index == teacher_index:
                continue
            terms.append(
                -(teacher_probability * student_log_probability).sum(dim=-1).mean()
            )
    if not terms:
        raise ValueError("DINO cross-view loss has no valid view pairs")
    return torch.stack(terms).mean()
