"""Build a deterministic, gallery-backed ginseng query protocol."""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, List, Mapping, Sequence, Tuple

from ginseng_benchmark.protocol import IMAGE_SUFFIXES, digest, image_files


QueryGroup = Dict[str, object]


def _require_directory(root: Path, label: str) -> Path:
    root = Path(root)
    if not root.exists():
        raise ValueError(f"{label} root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"{label} root is not a directory: {root}")
    return root.resolve()


def _name_sort_key(path: Path) -> Tuple[str, str]:
    return (path.name.casefold(), path.name)


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _unique_by_basename(paths: Sequence[Path], label: str) -> Dict[str, Path]:
    unique: Dict[str, Path] = {}
    for path in paths:
        key = path.name.casefold()
        if key in unique:
            raise ValueError(
                f"duplicate basename in {label}: "
                f"{unique[key].name} / {path.name}"
            )
        unique[key] = path
    return unique


def _numeric_group_directories(
    test_root: Path,
    expected_groups: int,
) -> Tuple[Path, ...]:
    if expected_groups < 1:
        raise ValueError("expected_groups must be at least 1")
    group_directories = tuple(path for path in test_root.iterdir() if path.is_dir())
    non_numeric = sorted(
        (path.name for path in group_directories if not path.name.isdigit()),
        key=lambda name: (name.casefold(), name),
    )
    if non_numeric:
        raise ValueError(
            "test group names must be numeric; found: " + ", ".join(non_numeric)
        )
    if len(group_directories) != expected_groups:
        raise ValueError(
            "test group count mismatch: "
            f"expected {expected_groups}, found {len(group_directories)}"
        )
    by_number = sorted(group_directories, key=lambda path: int(path.name))
    expected_names = [str(number) for number in range(1, expected_groups + 1)]
    actual_names = [path.name for path in by_number]
    if actual_names != expected_names:
        raise ValueError(
            f"test group ids must be exactly 1..{expected_groups}; "
            f"found: {', '.join(actual_names)}"
        )
    return tuple(by_number)


def _validated_group_images(
    test_root: Path,
    group_directories: Sequence[Path],
) -> Tuple[Tuple[Path, Tuple[Path, ...]], ...]:
    direct_groups = set(group_directories)
    all_test_images = image_files(test_root)
    for image in all_test_images:
        if image.parent not in direct_groups:
            raise ValueError(
                "test images must be directly inside a numeric group directory: "
                f"{image}"
            )

    seen: Dict[str, Path] = {}
    validated = []
    for group in group_directories:
        images = tuple(
            sorted(
                (
                    path
                    for path in group.iterdir()
                    if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
                ),
                key=_name_sort_key,
            )
        )
        if len(images) < 2:
            raise ValueError(
                f"test group {group.name} must contain at least 2 direct images"
            )
        for image in images:
            key = image.name.casefold()
            if key in seen:
                raise ValueError(
                    "duplicate basename in test groups: "
                    f"{seen[key].name} / {image.name}"
                )
            seen[key] = image
        validated.append((group, images))
    return tuple(validated)


def build_query_groups(
    test_root: Path,
    gallery_root: Path,
    expected_groups: int = 271,
) -> List[QueryGroup]:
    """Return one gallery-backed query for every direct test-group image."""
    test_root = _require_directory(test_root, "test")
    gallery_root = _require_directory(gallery_root, "merged gallery")
    group_directories = _numeric_group_directories(test_root, expected_groups)
    grouped_images = _validated_group_images(test_root, group_directories)
    gallery_by_name = _unique_by_basename(image_files(gallery_root), "merged gallery")

    query_groups: List[QueryGroup] = []
    for group, test_images in grouped_images:
        gallery_images = []
        for test_image in test_images:
            key = test_image.name.casefold()
            gallery_image = gallery_by_name.get(key)
            if gallery_image is None:
                raise ValueError(f"test image missing from merged gallery: {test_image.name}")
            if digest(test_image) != digest(gallery_image):
                raise ValueError(
                    "test and merged gallery content mismatch: "
                    f"{test_image.name}"
                )
            resolved_gallery_image = gallery_image.resolve()
            if not _is_within(resolved_gallery_image, gallery_root):
                raise ValueError(
                    "resolved gallery file is outside merged gallery root: "
                    f"{gallery_image.name}"
                )
            gallery_images.append(resolved_gallery_image)

        for query_image in gallery_images:
            query_key = query_image.name.casefold()
            positives = [
                str(candidate)
                for candidate in gallery_images
                if candidate.name.casefold() != query_key
            ]
            if not positives:
                raise ValueError(
                    f"query must have at least one positive: {query_image.name}"
                )
            query_groups.append(
                {
                    "group_id": group.name,
                    "name": group.name,
                    "query_image": str(query_image),
                    "same_ginsengs": positives,
                }
            )
    return query_groups


def _query_protocol_sha256(query_groups: Sequence[Mapping[str, object]]) -> str:
    canonical_groups = []
    for group in query_groups:
        query_name = Path(str(group["query_image"])).name
        positive_names = [Path(str(path)).name for path in group["same_ginsengs"]]
        canonical_groups.append(
            {
                "group_id": str(group["group_id"]),
                "name": str(group["name"]),
                "query_name": query_name,
                "same_ginseng_names": positive_names,
            }
        )
    canonical_json = json.dumps(
        canonical_groups,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def build_query_protocol(
    test_root: Path,
    gallery_root: Path,
    dataset_manifest_sha256: str,
    gallery_count: int,
    expected_groups: int = 271,
) -> Dict[str, object]:
    """Build the loader-compatible payload and root-independent metadata."""
    if (
        len(dataset_manifest_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in dataset_manifest_sha256)
    ):
        raise ValueError("dataset_manifest_sha256 must be a 64-character hex digest")
    if gallery_count < 1:
        raise ValueError("gallery_count must be at least 1")
    gallery_root = _require_directory(gallery_root, "merged gallery")
    actual_gallery_count = len(image_files(gallery_root))
    if gallery_count != actual_gallery_count:
        raise ValueError(
            "gallery_count does not match actual merged gallery files: "
            f"declared {gallery_count}, actual {actual_gallery_count}"
        )

    query_groups = build_query_groups(test_root, gallery_root, expected_groups)
    positive_distribution = Counter(
        len(group["same_ginsengs"]) for group in query_groups
    )
    metadata = {
        "schema_version": "1.0",
        "dataset_manifest_sha256": dataset_manifest_sha256.lower(),
        "group_count": expected_groups,
        "query_count": len(query_groups),
        "gallery_count": gallery_count,
        "positive_count_distribution": {
            str(count): positive_distribution[count]
            for count in sorted(positive_distribution)
        },
        "query_protocol_sha256": _query_protocol_sha256(query_groups),
    }
    return {"metadata": metadata, "query_groups": query_groups}


def write_query_protocol_atomic(payload: Mapping[str, object], output: Path) -> None:
    """Write stable UTF-8 JSON without exposing a partially written output."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=".query-groups-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(
                payload,
                temporary_file,
                ensure_ascii=False,
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
