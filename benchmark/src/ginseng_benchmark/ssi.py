"""Group-level embedding cohesion analysis for multi-view retrieval."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Sequence

import numpy as np

from ginseng_benchmark.cache import FeatureCache
from ginseng_benchmark.evaluation import _validated_protocol


def compute_group_ssi(
    cache: FeatureCache,
    query_protocol: Mapping[str, object],
) -> List[Dict[str, object]]:
    """Compute mapped mean pairwise cosine similarity for every test identity."""
    groups = _validated_protocol(query_protocol)
    if (
        cache.metadata.get("dataset_manifest_sha256")
        != query_protocol["metadata"]["dataset_manifest_sha256"]
    ):
        raise ValueError("feature cache and query protocol manifest mismatch")
    if len(cache.paths) != query_protocol["metadata"]["gallery_count"]:
        raise ValueError("feature cache gallery count does not match protocol")

    features = np.asarray(cache.features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != len(cache.paths):
        raise ValueError("feature cache matrix/path mismatch")
    norms = np.linalg.norm(features.astype(np.float64), axis=1)
    if np.any(norms == 0.0) or not np.isfinite(norms).all():
        raise ValueError("feature cache contains invalid feature rows")
    normalized = features / norms[:, None].astype(np.float32)
    path_index = {path.casefold(): index for index, path in enumerate(cache.paths)}

    identity_images: Dict[str, set[str]] = defaultdict(set)
    for group in groups:
        identity_images[str(group["group_id"])].add(str(group["query_image"]))

    rows: List[Dict[str, object]] = []
    for group_id in sorted(identity_images, key=lambda value: (len(value), value)):
        names = sorted(identity_images[group_id], key=str.casefold)
        if len(names) < 2:
            raise ValueError(f"SSI requires at least two images: {group_id}")
        missing = [name for name in names if name.casefold() not in path_index]
        if missing:
            raise ValueError(f"SSI image missing from cache: {missing[0]}")
        indices = [path_index[name.casefold()] for name in names]
        group_features = normalized[indices]
        similarities = group_features @ group_features.T
        upper = similarities[np.triu_indices(len(indices), k=1)]
        mean_cosine = float(np.mean(upper))
        rows.append(
            {
                "group_id": group_id,
                "num_images": len(names),
                "mean_cosine": mean_cosine,
                "ssi": 0.5 * (1.0 + mean_cosine),
            }
        )
    if len(rows) != query_protocol["metadata"]["group_count"]:
        raise ValueError("SSI group count does not match query protocol")
    return rows


def attach_group_metrics(
    ssi_rows: Sequence[Mapping[str, object]],
    per_query: Sequence[Mapping[str, object]],
    *,
    metric_keys: Sequence[str] = ("map", "mrr", "recall@5", "recall@10"),
) -> List[Dict[str, object]]:
    """Average query metrics within each identity and attach them to SSI rows."""
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in per_query:
        grouped[str(row["group_id"])].append(row)
    output = []
    for ssi_row in ssi_rows:
        group_id = str(ssi_row["group_id"])
        query_rows = grouped.get(group_id)
        if not query_rows:
            raise ValueError(f"missing per-query metrics for SSI group: {group_id}")
        merged = dict(ssi_row)
        merged["query_count"] = len(query_rows)
        for metric in metric_keys:
            values = [float(row[metric]) for row in query_rows]
            merged[metric] = float(np.mean(values))
        output.append(merged)
    return output


def assign_tertile_bands(
    rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Assign near-equal Low/Mid/High bands without dropping tied samples."""
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (float(row["ssi"]), str(row["group_id"])),
    )
    labels = ("Low", "Mid", "High")
    for label, indices in zip(labels, np.array_split(np.arange(len(ordered)), 3)):
        for index in indices.tolist():
            ordered[index]["ssi_band"] = label
    return ordered
