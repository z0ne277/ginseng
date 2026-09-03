"""Shared validation helpers for frozen vision-encoder feature extraction."""

from pathlib import Path
from typing import Sequence, Tuple


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


def discover_flat_image_paths(root: Path) -> Tuple[Path, ...]:
    """Return supported images from a strictly flat gallery in stable order."""
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("image root must be a directory")
    nested_images = [
        path for path in root.rglob("*")
        if path.is_file() and path.parent != root and path.suffix.casefold() in IMAGE_SUFFIXES
    ]
    if nested_images:
        raise ValueError("gallery must be flat; nested image files are not allowed")
    images = tuple(
        sorted(
            (
                path.resolve(strict=True)
                for path in root.iterdir()
                if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    )
    if not images:
        raise ValueError("gallery contains no supported image files")
    names = [path.name.casefold() for path in images]
    if len(names) != len(set(names)):
        raise ValueError("gallery contains duplicate case-insensitive basenames")
    return images


def select_pooled_features(outputs):
    """Prefer an official pooler output and otherwise use the CLS token."""
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is None:
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None or getattr(hidden, "ndim", None) != 3 or hidden.shape[1] < 1:
            raise ValueError("model output has neither pooler_output nor CLS hidden state")
        pooled = hidden[:, 0]
    if getattr(pooled, "ndim", None) != 2 or pooled.shape[1] < 1:
        raise ValueError("pooled model feature must be a non-empty 2D tensor")
    return pooled


def forward_hf_features(model, model_inputs, extractor_kind: str):
    """Run one explicitly declared Hugging Face feature extraction strategy."""
    if extractor_kind == "pooler_or_cls":
        features = select_pooled_features(model(**model_inputs))
    elif extractor_kind == "model_image_features":
        extractor = getattr(model, "get_image_features", None)
        if not callable(extractor):
            raise ValueError(
                "model_image_features requires a callable model.get_image_features"
            )
        features = extractor(**model_inputs)
    else:
        raise ValueError(f"unsupported extractor kind: {extractor_kind}")
    if getattr(features, "ndim", None) != 2 or features.shape[1] < 1:
        raise ValueError("model feature output must be a non-empty 2D tensor")
    return features


def l2_normalize_rows(features):
    """Convert features to finite float32 rows with unit L2 norm."""
    import torch

    if getattr(features, "ndim", None) != 2 or features.shape[1] < 1:
        raise ValueError("features must be a non-empty 2D tensor")
    features = features.detach().to(dtype=torch.float32)
    if not torch.isfinite(features).all():
        raise ValueError("features contain NaN or Inf")
    norms = torch.linalg.vector_norm(features, ord=2, dim=1, keepdim=True)
    if torch.any(norms <= 0):
        raise ValueError("features contain a zero-norm row")
    normalized = features / norms
    if not torch.isfinite(normalized).all():
        raise ValueError("normalized features contain NaN or Inf")
    return normalized


def validate_feature_collection(features, paths: Sequence[Path], expected_dim: int) -> None:
    """Validate a complete extracted feature matrix before serialization."""
    import torch

    if getattr(features, "ndim", None) != 2:
        raise ValueError("feature collection must be 2D")
    if features.shape != (len(paths), expected_dim):
        raise ValueError(
            f"feature collection shape mismatch: expected {(len(paths), expected_dim)}, "
            f"found {tuple(features.shape)}"
        )
    if features.dtype != torch.float32 or not torch.isfinite(features).all():
        raise ValueError("feature collection must be finite float32")
    norms = torch.linalg.vector_norm(features, ord=2, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=1e-4):
        raise ValueError("feature collection is not L2 normalized")
