"""Manifest-bound full-ranking evaluation for standardized feature caches."""

import csv
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ginseng_benchmark.cache import FeatureCache
from ginseng_benchmark.metrics import (
    aggregate_query_metrics,
    bootstrap_confidence_intervals,
    metric_definition_metadata,
    metrics_for_ranking,
)
from ginseng_benchmark.query_groups import _query_protocol_sha256


EVALUATION_SCHEMA_VERSION = 1
_HEX_DIGITS = frozenset("0123456789abcdef")


def _strict_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _basename(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string")
    # Protocols are generated on Windows but may be inspected by another OS.
    windows_name = PureWindowsPath(value).name
    posix_name = PurePosixPath(value).name
    name = windows_name if "\\" in value else posix_name
    if not name or name in {".", ".."}:
        raise ValueError(f"{label} does not contain a stable basename")
    return name


def _validated_protocol(payload: Mapping[str, object]) -> Tuple[Dict[str, object], ...]:
    if not isinstance(payload, dict):
        raise ValueError("query protocol must be a JSON object")
    metadata = payload.get("metadata")
    groups = payload.get("query_groups")
    if not isinstance(metadata, dict) or not isinstance(groups, list):
        raise ValueError("query protocol requires metadata and query_groups")
    required_metadata = {
        "dataset_manifest_sha256",
        "group_count",
        "query_count",
        "gallery_count",
        "positive_count_distribution",
        "query_protocol_sha256",
    }
    if not required_metadata.issubset(metadata):
        raise ValueError("query protocol metadata is incomplete")
    for key in ("dataset_manifest_sha256", "query_protocol_sha256"):
        value = metadata[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or not set(value).issubset(_HEX_DIGITS)
        ):
            raise ValueError(f"query protocol {key} must be a lowercase SHA-256 digest")
    for key in ("group_count", "query_count", "gallery_count"):
        value = metadata[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"query protocol {key} must be a positive integer")
    if metadata["query_count"] != len(groups):
        raise ValueError("query protocol query_count does not match query_groups")
    if metadata["gallery_count"] < 2:
        raise ValueError("query protocol gallery_count is invalid")

    normalized: List[Dict[str, object]] = []
    query_keys = set()
    identity_ids = set()
    positive_distribution: Dict[str, int] = {}
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"query group {index} must be an object")
        required = {"group_id", "name", "query_image", "same_ginsengs"}
        if not required.issubset(group):
            raise ValueError(f"query group {index} is missing required fields")
        group_id = group["group_id"]
        name = group["name"]
        if not isinstance(group_id, str) or not group_id or name != group_id:
            raise ValueError("query group name must equal its non-empty group_id")
        query_name = _basename(group["query_image"], "query_image")
        query_key = query_name.casefold()
        if query_key in query_keys:
            raise ValueError(f"duplicate query basename: {query_name}")
        query_keys.add(query_key)
        positives_value = group["same_ginsengs"]
        if isinstance(positives_value, (str, bytes)) or not isinstance(
            positives_value, list
        ):
            raise ValueError("same_ginsengs must be a non-empty list")
        positive_names = tuple(
            _basename(value, "same_ginsengs item") for value in positives_value
        )
        if not positive_names:
            raise ValueError("same_ginsengs must contain at least one positive")
        positive_keys = [value.casefold() for value in positive_names]
        if len(set(positive_keys)) != len(positive_keys):
            raise ValueError("same_ginsengs contains duplicate positives")
        if query_key in positive_keys:
            raise ValueError("query protocol contains a self-positive")
        identity_ids.add(group_id)
        distribution_key = str(len(positive_names))
        positive_distribution[distribution_key] = (
            positive_distribution.get(distribution_key, 0) + 1
        )
        normalized.append(
            {
                "group_id": group_id,
                "name": name,
                "query_image": query_name,
                "same_ginsengs": positive_names,
            }
        )

    if metadata["group_count"] != len(identity_ids):
        raise ValueError("query protocol group_count does not match identities")
    if metadata["positive_count_distribution"] != {
        key: positive_distribution[key]
        for key in sorted(positive_distribution, key=int)
    }:
        raise ValueError("query protocol positive_count_distribution is inconsistent")
    calculated_protocol_sha256 = _query_protocol_sha256(groups)
    if metadata["query_protocol_sha256"] != calculated_protocol_sha256:
        raise ValueError("query protocol sha256 does not match query_groups")
    return tuple(normalized)


def load_query_protocol(path: Path) -> Dict[str, object]:
    """Load and validate the canonical query protocol with strict JSON."""
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8-sig"),
            parse_constant=_strict_json_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("unable to read query protocol JSON") from error
    _validated_protocol(payload)
    return payload


def _normalized_ks(ks: Iterable[int]) -> Tuple[int, ...]:
    values = []
    seen = set()
    try:
        iterator = iter(ks)
    except TypeError as error:
        raise ValueError("ks must be an iterable of positive integers") from error
    for value in iterator:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("ks must contain only positive integers")
        if value not in seen:
            seen.add(value)
            values.append(value)
    if not values:
        raise ValueError("ks must not be empty")
    return tuple(values)


def evaluate_feature_cache(
    cache: FeatureCache,
    query_protocol: Mapping[str, object],
    *,
    ks: Iterable[int] = (1, 5, 10, 20),
    block_size: int = 32,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 42,
    confidence: float = 0.95,
) -> Dict[str, object]:
    """Evaluate every query against the complete gallery using cosine similarity."""
    normalized_ks = _normalized_ks(ks)
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    groups = _validated_protocol(query_protocol)
    protocol_metadata = query_protocol["metadata"]
    cache_metadata = cache.metadata
    if (
        cache_metadata.get("dataset_manifest_sha256")
        != protocol_metadata["dataset_manifest_sha256"]
    ):
        raise ValueError("cache and query protocol dataset manifest mismatch")
    if cache_metadata.get("num_images") != len(cache.paths):
        raise ValueError("cache metadata num_images is inconsistent")
    if protocol_metadata["gallery_count"] != len(cache.paths):
        raise ValueError("cache gallery count does not match query protocol")
    features = np.asarray(cache.features)
    if features.ndim != 2 or features.shape[0] != len(cache.paths):
        raise ValueError("cache features do not match cached paths")
    if not np.isfinite(features).all():
        raise ValueError("cache features contain non-finite values")
    norms = np.linalg.norm(features.astype(np.float64, copy=False), axis=1)
    if np.any(norms == 0.0) or not np.isfinite(norms).all():
        raise ValueError("cache contains a zero or invalid feature row")
    normalized_features = np.ascontiguousarray(
        features.astype(np.float32, copy=False) / norms[:, None].astype(np.float32)
    )

    if any(
        not isinstance(path, str)
        or not path
        or _basename(path, "cache path") != path
        for path in cache.paths
    ):
        raise ValueError("cache paths must be stable basenames")
    path_keys = [path.casefold() for path in cache.paths]
    if len(path_keys) != len(set(path_keys)):
        raise ValueError("cache paths contain duplicate casefold basenames")
    path_index = {key: index for index, key in enumerate(path_keys)}
    query_indices = []
    relevant_names = []
    for group in groups:
        query_key = str(group["query_image"]).casefold()
        if query_key not in path_index:
            raise ValueError(f"query image is missing from cache: {group['query_image']}")
        positives = tuple(str(value) for value in group["same_ginsengs"])
        missing = [value for value in positives if value.casefold() not in path_index]
        if missing:
            raise ValueError(f"positive image is missing from cache: {missing[0]}")
        query_indices.append(path_index[query_key])
        relevant_names.append(positives)

    top_result_count = min(20, max(0, len(cache.paths) - 1))
    per_query: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    for block_start in range(0, len(groups), block_size):
        block_end = min(block_start + block_size, len(groups))
        block_query_indices = np.asarray(query_indices[block_start:block_end], dtype=np.int64)
        similarities = normalized_features[block_query_indices] @ normalized_features.T
        similarities[np.arange(block_end - block_start), block_query_indices] = -np.inf
        orders = np.argsort(-similarities, axis=1, kind="stable")
        for local_index, order in enumerate(orders):
            global_index = block_start + local_index
            self_index = query_indices[global_index]
            ranked_indices = [int(index) for index in order if int(index) != self_index]
            ranked_names = [cache.paths[index] for index in ranked_indices]
            metrics = metrics_for_ranking(
                ranked_names,
                relevant_names[global_index],
                ks=normalized_ks,
            )
            metric_rows.append(metrics)
            group = groups[global_index]
            per_query.append(
                {
                    "group_id": group["group_id"],
                    "name": group["name"],
                    "query_image": group["query_image"],
                    "positive_count": len(relevant_names[global_index]),
                    "top_results": ranked_names[:top_result_count],
                    **metrics,
                }
            )

    aggregate = aggregate_query_metrics(metric_rows)
    macro_metric_keys = list(aggregate["macro"].keys())
    bootstrap = bootstrap_confidence_intervals(
        metric_rows,
        macro_metric_keys,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
        cluster_ids=[group["group_id"] for group in groups],
        confidence=confidence,
    )
    result = {
        "metadata": {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "model_id": cache_metadata.get("model_id"),
            "model_source": cache_metadata.get("model_source"),
            "dataset_manifest_sha256": cache_metadata.get("dataset_manifest_sha256"),
            "query_protocol_sha256": protocol_metadata["query_protocol_sha256"],
            "features_sha256": cache_metadata.get("features_sha256"),
            "paths_sha256": cache_metadata.get("paths_sha256"),
            "gallery_count": len(cache.paths),
            "query_count": len(groups),
            "candidate_count_per_query": len(cache.paths) - 1,
            "feature_dim": int(features.shape[1]),
            "similarity": "cosine",
            "ranking_scope": "full",
            "self_exclusion": "cache_index_to_negative_infinity",
            "tie_breaker": "canonical_cache_order",
            "ks": list(normalized_ks),
            "metric_definitions": metric_definition_metadata(),
        },
        "aggregate": aggregate,
        "bootstrap": bootstrap,
        "per_query": per_query,
    }
    # Enforce the public strict-JSON contract before returning a large result.
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result


def _atomic_text_path(output: Path) -> Tuple[Path, object]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    return Path(temporary.name), temporary


def write_evaluation_json_atomic(result: Mapping[str, object], output: Path) -> None:
    """Write strict evaluation JSON atomically."""
    temporary_path: Optional[Path] = None
    temporary_file = None
    try:
        temporary_path, temporary_file = _atomic_text_path(Path(output))
        with temporary_file:
            json.dump(
                result,
                temporary_file,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_per_query_csv_atomic(
    per_query: Sequence[Mapping[str, object]], output: Path
) -> None:
    """Write flattened per-query metrics as UTF-8-BOM CSV atomically."""
    rows = list(per_query)
    if not rows:
        raise ValueError("per_query CSV requires at least one row")
    base_fields = ["group_id", "name", "query_image", "positive_count"]
    excluded = set(base_fields) | {"top_results"}
    metric_fields = [key for key in rows[0] if key not in excluded]
    fieldnames = base_fields + metric_fields + ["top_results"]
    expected_keys = set(rows[0])
    temporary_path: Optional[Path] = None
    temporary_file = None
    try:
        temporary_path, temporary_file = _atomic_text_path(Path(output))
        with temporary_file:
            temporary_file.write("\ufeff")
            writer = csv.DictWriter(temporary_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                if set(row) != expected_keys:
                    raise ValueError("per_query rows have inconsistent fields")
                serialized = dict(row)
                serialized["top_results"] = json.dumps(
                    row["top_results"], ensure_ascii=False, separators=(",", ":")
                )
                writer.writerow(serialized)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
