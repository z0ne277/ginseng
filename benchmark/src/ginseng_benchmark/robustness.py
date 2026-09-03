"""Controlled query perturbations and clean-gallery shift evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from ginseng_benchmark.cache import FeatureCache
from ginseng_benchmark.evaluation import _normalized_ks, _validated_protocol
from ginseng_benchmark.metrics import (
    aggregate_query_metrics,
    bootstrap_confidence_intervals,
    metric_definition_metadata,
    metrics_for_ranking,
)


_PERTURBATIONS = frozenset(
    {
        "mask_erode",
        "mask_dilate",
        "branch_occlusion",
        "boundary_jitter",
        "rotation",
        "gaussian_blur",
        "jpeg",
    }
)


def _scaled_radius(height: int, width: int, severity: int) -> int:
    factors = {1: 0.003, 2: 0.006, 3: 0.012}
    return max(1, int(round(min(height, width) * factors[severity])))


def apply_query_perturbation(
    image: np.ndarray,
    *,
    kind: str,
    severity: int,
    seed: int,
) -> np.ndarray:
    """Apply a deterministic mask or imaging perturbation."""
    if kind not in _PERTURBATIONS:
        raise ValueError(f"unsupported robustness perturbation: {kind}")
    if severity not in {1, 2, 3}:
        raise ValueError("robustness severity must be 1, 2 or 3")
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
        raise ValueError("robustness image must be an HxWx3 uint8 array")

    import cv2

    height, width = value.shape[:2]
    gray = cv2.cvtColor(value, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    radius = _scaled_radius(height, width, severity)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )

    if kind == "mask_erode":
        output_mask = cv2.erode(mask, kernel, iterations=1)
        return np.repeat(output_mask[..., None], 3, axis=2)
    if kind == "mask_dilate":
        output_mask = cv2.dilate(mask, kernel, iterations=1)
        return np.repeat(output_mask[..., None], 3, axis=2)
    if kind == "boundary_jitter":
        eroded = cv2.erode(mask, kernel, iterations=1)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        rng = np.random.default_rng(seed)
        selector = rng.random(mask.shape) > 0.5
        output_mask = np.where(selector, eroded, dilated).astype(np.uint8)
        return np.repeat(output_mask[..., None], 3, axis=2)
    if kind == "branch_occlusion":
        output = value.copy()
        foreground = np.argwhere(mask > 0)
        if foreground.size == 0:
            return output
        y_min, x_min = foreground.min(axis=0)
        y_max, x_max = foreground.max(axis=0)
        rng = np.random.default_rng(seed)
        fractions = {1: 0.08, 2: 0.14, 3: 0.22}
        box_width = max(1, int((x_max - x_min + 1) * fractions[severity]))
        box_height = max(1, int((y_max - y_min + 1) * fractions[severity]))
        center_x = int(rng.integers(x_min, x_max + 1))
        center_y = int(rng.integers(y_min, y_max + 1))
        left = max(0, center_x - box_width // 2)
        right = min(width, left + box_width)
        top = max(0, center_y - box_height // 2)
        bottom = min(height, top + box_height)
        output[top:bottom, left:right] = 0
        return output
    if kind == "rotation":
        angles = {1: 5.0, 2: 12.0, 3: 25.0}
        matrix = cv2.getRotationMatrix2D(
            (width / 2.0, height / 2.0),
            angles[severity],
            1.0,
        )
        return cv2.warpAffine(
            value,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
    if kind == "gaussian_blur":
        blur_radius = max(1, radius)
        kernel_size = 2 * blur_radius + 1
        return cv2.GaussianBlur(value, (kernel_size, kernel_size), sigmaX=0)
    if kind == "jpeg":
        qualities = {1: 70, 2: 40, 3: 15}
        success, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(value, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, qualities[severity]],
        )
        if not success:
            raise RuntimeError("unable to encode JPEG robustness sample")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    raise AssertionError("unreachable perturbation branch")


def _normalize_rows(features: np.ndarray, label: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError(f"{label} features must be a finite 2D matrix")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if np.any(norms == 0.0) or not np.isfinite(norms).all():
        raise ValueError(f"{label} features contain an invalid row")
    return np.ascontiguousarray(matrix / norms[:, None].astype(np.float32))


def evaluate_shifted_queries(
    clean_cache: FeatureCache,
    *,
    shifted_features: np.ndarray,
    shifted_paths: Sequence[str],
    query_protocol: Mapping[str, object],
    condition: str,
    ks: Iterable[int] = (1, 5, 10, 20),
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 42,
) -> Dict[str, object]:
    """Rank shifted query embeddings against the unchanged clean gallery."""
    normalized_ks = _normalized_ks(ks)
    groups = _validated_protocol(query_protocol)
    protocol_metadata = query_protocol["metadata"]
    if (
        clean_cache.metadata.get("dataset_manifest_sha256")
        != protocol_metadata["dataset_manifest_sha256"]
    ):
        raise ValueError("clean cache and query protocol manifest mismatch")
    if len(clean_cache.paths) != protocol_metadata["gallery_count"]:
        raise ValueError("clean gallery count does not match query protocol")

    clean = _normalize_rows(clean_cache.features, "clean gallery")
    shifted = _normalize_rows(shifted_features, "shifted query")
    if clean.shape[1] != shifted.shape[1]:
        raise ValueError("clean and shifted feature dimensions do not match")
    if len(shifted_paths) != shifted.shape[0]:
        raise ValueError("shifted feature/path count mismatch")

    clean_keys = [str(path).casefold() for path in clean_cache.paths]
    if len(clean_keys) != len(set(clean_keys)):
        raise ValueError("clean gallery contains duplicate basenames")
    clean_index = {key: index for index, key in enumerate(clean_keys)}
    shifted_names = [str(path).replace("\\", "/").rsplit("/", 1)[-1] for path in shifted_paths]
    shifted_keys = [name.casefold() for name in shifted_names]
    if len(shifted_keys) != len(set(shifted_keys)):
        raise ValueError("shifted query features contain duplicate basenames")
    shifted_index = {key: index for index, key in enumerate(shifted_keys)}

    per_query: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    cluster_ids = []
    for group in groups:
        query_name = str(group["query_image"])
        query_key = query_name.casefold()
        if query_key not in shifted_index:
            raise ValueError(f"shifted feature missing query: {query_name}")
        if query_key not in clean_index:
            raise ValueError(f"clean gallery missing query: {query_name}")
        similarities = shifted[shifted_index[query_key]] @ clean.T
        similarities[clean_index[query_key]] = -np.inf
        order = np.argsort(-similarities, kind="stable")
        ranked_names = [
            clean_cache.paths[int(index)]
            for index in order
            if int(index) != clean_index[query_key]
        ]
        positives = tuple(str(item) for item in group["same_ginsengs"])
        metrics = metrics_for_ranking(ranked_names, positives, ks=normalized_ks)
        metric_rows.append(metrics)
        cluster_ids.append(group["group_id"])
        per_query.append(
            {
                "group_id": group["group_id"],
                "name": group["name"],
                "query_image": query_name,
                "positive_count": len(positives),
                "top_results": ranked_names[:20],
                **metrics,
            }
        )

    aggregate = aggregate_query_metrics(metric_rows)
    bootstrap = bootstrap_confidence_intervals(
        metric_rows,
        list(aggregate["macro"].keys()),
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
        cluster_ids=cluster_ids,
    )
    return {
        "metadata": {
            "schema_version": 1,
            "model_id": clean_cache.metadata.get("model_id"),
            "dataset_manifest_sha256": clean_cache.metadata.get(
                "dataset_manifest_sha256"
            ),
            "query_protocol_sha256": protocol_metadata["query_protocol_sha256"],
            "condition": condition,
            "query_feature_source": "shifted",
            "gallery_feature_source": "clean",
            "query_count": len(groups),
            "gallery_count": len(clean_cache.paths),
            "candidate_count_per_query": len(clean_cache.paths) - 1,
            "feature_dim": int(clean.shape[1]),
            "similarity": "cosine",
            "ranking_scope": "full clean gallery",
            "self_exclusion": "clean gallery basename",
            "ks": list(normalized_ks),
            "metric_definitions": metric_definition_metadata(),
        },
        "aggregate": aggregate,
        "bootstrap": bootstrap,
        "per_query": per_query,
    }
