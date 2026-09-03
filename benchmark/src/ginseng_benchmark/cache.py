"""Manifest-bound, pickle-free feature caches for retrieval evaluation."""

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ginseng_benchmark.protocol import AuditReport, digest


SCHEMA_VERSION = 1
_ARCHIVE_KEYS = frozenset({"features", "paths", "metadata_json"})
_REQUIRED_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "dataset_manifest_sha256",
        "model_id",
        "model_source",
        "preprocessing",
        "tta",
        "environment",
        "feature_normalization",
        "num_images",
        "feature_dim",
        "dtype",
        "path_format",
        "paths_sha256",
        "features_sha256",
    }
)
_OPTIONAL_METADATA_KEYS = frozenset({"checkpoint"})
_SENSITIVE_KEY_FRAGMENTS = (
    "token",
    "password",
    "secret",
    "api_key",
    "access_key",
)
_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class FeatureCache:
    features: np.ndarray
    paths: Tuple[str, ...]
    metadata: Dict[str, Any]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _features_sha256(features: np.ndarray) -> str:
    stable = np.ascontiguousarray(features, dtype=np.dtype("<f4"))
    return _sha256_bytes(stable.tobytes(order="C"))


def _paths_sha256(paths: Sequence[str]) -> str:
    canonical_json = json.dumps(
        list(paths),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _sha256_bytes(canonical_json.encode("utf-8"))


def _is_absolute_local_path(value: str) -> bool:
    if not value:
        return False
    return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()


def _validate_json_value(value: Any, location: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str) and _is_absolute_local_path(value):
            raise ValueError(f"absolute local paths are not allowed in metadata: {location}")
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"metadata numbers must be finite JSON values: {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"metadata keys must be JSON strings: {location}")
            normalized_key = key.casefold().replace("-", "_").replace(" ", "_")
            if any(fragment in normalized_key for fragment in _SENSITIVE_KEY_FRAGMENTS):
                raise ValueError(f"sensitive metadata key is forbidden: {location}.{key}")
            _validate_json_value(item, f"{location}.{key}")
        return
    raise ValueError(f"metadata contains a non-JSON value at {location}")


def _validate_metadata_object(value: Mapping[str, Any], label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} metadata must be a JSON object")
    _validate_json_value(value, label)
    # A JSON round-trip creates a detached tree containing only strict JSON types.
    return json.loads(
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    )


def _as_numpy_features(raw_features: Any) -> np.ndarray:
    value = raw_features
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    try:
        features = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError("features must be a numeric two-dimensional array") from error
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional (2D) matrix")
    if features.shape[1] <= 0:
        raise ValueError("feature dimension/width must be greater than zero")
    if (
        features.dtype.kind not in {"i", "u", "f"}
        or np.issubdtype(features.dtype, np.bool_)
        or np.issubdtype(features.dtype, np.complexfloating)
    ):
        raise ValueError("feature dtype must be real numeric, non-bool, and non-complex")
    try:
        normalized = np.ascontiguousarray(features, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("features cannot be represented as float32") from error
    if not np.isfinite(normalized).all():
        raise ValueError("features must contain only finite values")
    return normalized


def _validate_normalization(features: np.ndarray, mode: str) -> None:
    if mode not in {"l2", "none"}:
        raise ValueError("feature_normalization must be 'l2' or 'none'")
    if mode == "l2":
        norms = np.linalg.norm(features, axis=1)
        if np.any(norms == 0.0) or not np.allclose(
            norms,
            np.ones_like(norms),
            rtol=0.0,
            atol=1e-4,
        ):
            raise ValueError("L2 feature rows must be non-zero and normalized within 1e-4")


def _canonical_report_names(report: AuditReport) -> Tuple[str, ...]:
    if report.mismatches:
        raise ValueError("dataset audit must have zero mismatches before caching features")
    names = tuple(record.name for record in report.records)
    if len(names) != report.merged_count:
        raise ValueError("audit record count does not match merged image count")
    keys = [name.casefold() for name in names]
    if len(set(keys)) != len(keys):
        raise ValueError("audit manifest contains duplicate casefold basenames")
    for name in names:
        if not name or Path(name).name != name or _is_absolute_local_path(name):
            raise ValueError("audit manifest paths must be stable basenames")
    return names


def _validated_raw_path_keys(
    raw_paths: Sequence[os.PathLike],
    merged_root: Path,
) -> Tuple[str, ...]:
    if isinstance(raw_paths, (str, bytes, os.PathLike)):
        raise ValueError("raw paths must be a sequence, not a single path")
    try:
        path_values = tuple(raw_paths)
    except TypeError as error:
        raise ValueError("raw paths must be a finite sequence") from error
    resolved_root = Path(merged_root).resolve(strict=True)
    keys = []
    seen = set()
    for raw_value in path_values:
        try:
            raw_string = os.fspath(raw_value)
        except TypeError as error:
            raise ValueError("each raw path must be a string or path-like value") from error
        if isinstance(raw_string, bytes):
            raise ValueError("raw paths must be Unicode strings")
        raw_path = Path(raw_string)
        if raw_path.is_absolute():
            candidate = raw_path
        elif raw_string == raw_path.name and raw_path.name not in {"", ".", ".."}:
            candidate = resolved_root / raw_path.name
        else:
            raise ValueError("relative raw paths must be pure basenames")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"raw feature path does not exist: {raw_path.name}") from error
        if not resolved.is_file():
            raise ValueError(f"raw feature path is not a file: {raw_path.name}")
        if resolved.parent != resolved_root:
            raise ValueError(f"raw feature path escapes or is outside merged root: {raw_path.name}")
        key = resolved.name.casefold()
        if key in seen:
            raise ValueError(f"duplicate casefold raw path: {resolved.name}")
        seen.add(key)
        keys.append(key)
    return tuple(keys)


def _require_nonempty_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    stripped = value.strip()
    if _is_absolute_local_path(stripped):
        raise ValueError(f"{label} must not contain an absolute local path")
    return stripped


def build_feature_cache(
    raw_features: Any,
    raw_paths: Sequence[os.PathLike],
    report: AuditReport,
    *,
    model_id: str,
    model_source: str,
    feature_normalization: str,
    checkpoint: Optional[Path] = None,
    preprocessing: Optional[Mapping[str, Any]] = None,
    tta: Optional[Mapping[str, Any]] = None,
    environment: Optional[Mapping[str, Any]] = None,
    expected_feature_dim: Optional[int] = None,
) -> FeatureCache:
    """Validate and reorder raw features into the canonical audited manifest order."""
    features = _as_numpy_features(raw_features)
    try:
        raw_path_count = len(raw_paths)
    except TypeError as error:
        raise ValueError("raw paths must have a stable count") from error
    if features.shape[0] != raw_path_count:
        raise ValueError(
            "feature/path count mismatch: "
            f"features={features.shape[0]}, paths={raw_path_count}"
        )
    if expected_feature_dim is not None:
        if isinstance(expected_feature_dim, bool) or expected_feature_dim <= 0:
            raise ValueError("expected feature dimension must be a positive integer")
        if features.shape[1] != expected_feature_dim:
            raise ValueError(
                "feature dimension mismatch: "
                f"expected {expected_feature_dim}, found {features.shape[1]}"
            )

    canonical_names = _canonical_report_names(report)
    canonical_by_key = {name.casefold(): name for name in canonical_names}
    raw_keys = _validated_raw_path_keys(raw_paths, report.merged_root)
    raw_key_set = set(raw_keys)
    canonical_key_set = set(canonical_by_key)
    if raw_key_set != canonical_key_set:
        missing = len(canonical_key_set - raw_key_set)
        extra = len(raw_key_set - canonical_key_set)
        raise ValueError(
            "raw path set mismatch against dataset manifest: "
            f"missing={missing}, extra={extra}"
        )
    raw_index = {key: index for index, key in enumerate(raw_keys)}
    ordered = np.ascontiguousarray(
        features[[raw_index[name.casefold()] for name in canonical_names]],
        dtype=np.float32,
    )
    _validate_normalization(ordered, feature_normalization)

    clean_preprocessing = _validate_metadata_object(
        {} if preprocessing is None else preprocessing,
        "preprocessing",
    )
    clean_tta = _validate_metadata_object({} if tta is None else tta, "tta")
    clean_environment = _validate_metadata_object(
        {} if environment is None else environment,
        "environment",
    )
    metadata: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_manifest_sha256": report.manifest_sha256,
        "model_id": _require_nonempty_text(model_id, "model_id"),
        "model_source": _require_nonempty_text(model_source, "model_source"),
        "preprocessing": clean_preprocessing,
        "tta": clean_tta,
        "environment": clean_environment,
        "feature_normalization": feature_normalization,
        "num_images": len(canonical_names),
        "feature_dim": int(ordered.shape[1]),
        "dtype": "float32",
        "path_format": "basename",
        "paths_sha256": _paths_sha256(canonical_names),
        "features_sha256": _features_sha256(ordered),
    }
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise ValueError("checkpoint must be an existing file")
        metadata["checkpoint"] = {
            "name": checkpoint_path.name,
            "sha256": digest(checkpoint_path),
        }
    _validate_cache_components(ordered, canonical_names, metadata)
    return FeatureCache(features=ordered, paths=canonical_names, metadata=metadata)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_HEX_DIGITS)
    )


def _validate_cache_components(
    features: np.ndarray,
    paths: Sequence[str],
    metadata: Mapping[str, Any],
) -> None:
    if not isinstance(features, np.ndarray):
        raise ValueError("features must be a NumPy array")
    if features.ndim != 2 or features.shape[1] <= 0:
        raise ValueError("features must be a non-empty-width two-dimensional matrix")
    if features.dtype != np.dtype("float32"):
        raise ValueError("cached feature dtype must be float32")
    if not features.flags.c_contiguous:
        raise ValueError("cached features must be C contiguous")
    if not np.isfinite(features).all():
        raise ValueError("cached features must contain only finite values")

    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must decode to a JSON object")
    _validate_json_value(metadata, "metadata")
    metadata_keys = set(metadata)
    missing = _REQUIRED_METADATA_KEYS - metadata_keys
    unexpected = metadata_keys - (_REQUIRED_METADATA_KEYS | _OPTIONAL_METADATA_KEYS)
    if missing or unexpected:
        raise ValueError(
            "metadata schema mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported feature-cache schema_version")
    for key in (
        "dataset_manifest_sha256",
        "paths_sha256",
        "features_sha256",
    ):
        if not _valid_sha256(metadata[key]):
            raise ValueError(f"metadata {key} must be a lowercase SHA-256 digest")
    _require_nonempty_text(metadata["model_id"], "model_id")
    _require_nonempty_text(metadata["model_source"], "model_source")
    for key in ("preprocessing", "tta", "environment"):
        if not isinstance(metadata[key], dict):
            raise ValueError(f"metadata {key} must be a JSON object")
    if metadata["feature_normalization"] not in {"l2", "none"}:
        raise ValueError("metadata feature_normalization is invalid")
    if metadata["num_images"] != features.shape[0] or isinstance(
        metadata["num_images"], bool
    ):
        raise ValueError("metadata num_images does not match features")
    if metadata["feature_dim"] != features.shape[1] or isinstance(
        metadata["feature_dim"], bool
    ):
        raise ValueError("metadata feature_dim does not match features")
    if metadata["dtype"] != "float32":
        raise ValueError("metadata dtype must be float32")
    if metadata["path_format"] != "basename":
        raise ValueError("metadata path_format must be basename")

    if len(paths) != features.shape[0]:
        raise ValueError("cached feature/path count mismatch")
    path_keys = []
    for path in paths:
        if not isinstance(path, str):
            raise ValueError("cached paths must be Unicode strings")
        if not path or Path(path).name != path or _is_absolute_local_path(path):
            raise ValueError("cached paths must contain only basenames")
        path_keys.append(path.casefold())
    if len(set(path_keys)) != len(path_keys):
        raise ValueError("cached paths contain duplicate casefold basenames")
    if metadata["paths_sha256"] != _paths_sha256(paths):
        raise ValueError("paths_sha256 does not match cached paths")
    if metadata["features_sha256"] != _features_sha256(features):
        raise ValueError("features_sha256 does not match cached features")
    _validate_normalization(features, metadata["feature_normalization"])

    checkpoint = metadata.get("checkpoint")
    if checkpoint is not None:
        if not isinstance(checkpoint, dict) or set(checkpoint) != {"name", "sha256"}:
            raise ValueError("checkpoint metadata must contain only name and sha256")
        if (
            not isinstance(checkpoint["name"], str)
            or not checkpoint["name"]
            or Path(checkpoint["name"]).name != checkpoint["name"]
            or _is_absolute_local_path(checkpoint["name"])
        ):
            raise ValueError("checkpoint metadata name must be a basename")
        if not _valid_sha256(checkpoint["sha256"]):
            raise ValueError("checkpoint sha256 is invalid")


def write_feature_cache_atomic(cache: FeatureCache, output: Path) -> None:
    """Write a validated NPZ cache atomically without exposing pickle arrays."""
    output = Path(output)
    if output.suffix.casefold() != ".npz":
        raise ValueError("feature cache output must use the .npz suffix")
    _validate_cache_components(cache.features, cache.paths, cache.metadata)
    metadata_json = json.dumps(
        cache.metadata,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez_compressed(
                temporary_file,
                features=cache.features,
                paths=np.asarray(cache.paths, dtype=np.str_),
                metadata_json=np.asarray(metadata_json, dtype=np.str_),
            )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_feature_cache(
    path: Path,
    *,
    expected_dataset_manifest_sha256: Optional[str] = None,
    expected_paths: Optional[Sequence[str]] = None,
) -> FeatureCache:
    """Load and fully validate the standard pickle-free NPZ cache."""
    path = Path(path)
    if path.suffix.casefold() != ".npz":
        raise ValueError("feature cache must use the .npz suffix")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != _ARCHIVE_KEYS:
                raise ValueError("feature cache NPZ keys do not match the strict schema")
            try:
                raw_features = archive["features"]
                raw_paths = archive["paths"]
                raw_metadata_json = archive["metadata_json"]
            except ValueError as error:
                raise ValueError(
                    "feature cache contains an object/pickle array and is unsafe"
                ) from error
            if raw_features.dtype != np.dtype("float32"):
                raise ValueError("cached feature dtype must be float32")
            if not raw_features.flags.c_contiguous:
                raise ValueError("cached features must be C contiguous")
            features = np.array(raw_features, dtype=np.float32, order="C", copy=True)
            if raw_paths.ndim != 1 or raw_paths.dtype.kind != "U":
                raise ValueError("cached paths must be a one-dimensional Unicode array")
            paths = tuple(raw_paths.tolist())
            if raw_metadata_json.ndim != 0 or raw_metadata_json.dtype.kind != "U":
                raise ValueError("metadata_json must be a Unicode scalar")
            metadata_text = raw_metadata_json.item()
    except (OSError, KeyError) as error:
        raise ValueError("unable to read feature cache NPZ") from error
    try:
        metadata = json.loads(metadata_text)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("metadata_json is not valid JSON") from error
    _validate_cache_components(features, paths, metadata)

    if (
        expected_dataset_manifest_sha256 is not None
        and metadata["dataset_manifest_sha256"]
        != expected_dataset_manifest_sha256
    ):
        raise ValueError("dataset manifest does not match expected manifest")
    if expected_paths is not None:
        expected_tuple = tuple(expected_paths)
        expected_keys = [path.casefold() for path in expected_tuple]
        if len(expected_keys) != len(set(expected_keys)):
            raise ValueError("expected paths contain duplicate casefold basenames")
        if paths != expected_tuple:
            raise ValueError("cached paths do not strictly match expected paths")
    return FeatureCache(features=features, paths=paths, metadata=metadata)


def load_trusted_torch_cache(
    path: Path,
    *,
    trusted_local_pt: bool = False,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Load a locally trusted legacy torch pickle cache on the CPU."""
    path = Path(path)
    if path.suffix.casefold() != ".pt":
        raise ValueError("legacy raw cache must use the .pt suffix")
    if not trusted_local_pt:
        raise ValueError(
            "refusing torch pickle cache; pass trusted_local_pt=True only for a trusted local .pt file"
        )
    try:
        import torch
    except ImportError as error:
        raise ValueError("PyTorch is required to read a trusted local .pt cache") from error
    load_arguments: Dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_arguments["weights_only"] = False
    payload = torch.load(path, **load_arguments)
    if not isinstance(payload, dict) or not {"features", "paths"}.issubset(payload):
        raise ValueError("trusted torch cache must be a dict containing features and paths")
    features = _as_numpy_features(payload["features"])
    raw_paths = payload["paths"]
    if isinstance(raw_paths, (str, bytes, os.PathLike)):
        raise ValueError("trusted torch cache paths must be a sequence")
    try:
        paths = tuple(os.fspath(item) for item in raw_paths)
    except (TypeError, ValueError) as error:
        raise ValueError("trusted torch cache paths must be strings or path-like values") from error
    if any(isinstance(item, bytes) for item in paths):
        raise ValueError("trusted torch cache paths must be Unicode strings")
    return features, paths
