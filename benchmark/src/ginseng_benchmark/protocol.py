"""Content-based dataset auditing for the ginseng retrieval benchmark."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple


IMAGE_SUFFIXES = frozenset(
    {
        ".bmp",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }
)
_DIGEST_CHUNK_SIZE = 1024 * 1024
_ERROR_NAME_LIMIT = 10


@dataclass(frozen=True)
class FileRecord:
    name: str
    path: Path
    size: int
    sha256: str
    source: str
    group_id: Optional[str] = None


@dataclass(frozen=True)
class Mismatch:
    name: str
    source: FileRecord
    merged: FileRecord


@dataclass(frozen=True)
class AuditReport:
    library_root: Path
    test_root: Path
    merged_root: Path
    library_count: int
    test_count: int
    merged_count: int
    group_count: int
    mismatches: Tuple[Mismatch, ...]
    records: Tuple[FileRecord, ...]
    manifest_sha256: str


def image_files(root: Path) -> Tuple[Path, ...]:
    """Return image files below *root* in stable relative-path order."""
    paths = (
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    )
    return tuple(
        sorted(
            paths,
            key=lambda path: (
                path.relative_to(root).as_posix().casefold(),
                path.relative_to(root).as_posix(),
            ),
        )
    )


def digest(path: Path) -> str:
    """Calculate a SHA-256 digest without loading the whole file into memory."""
    sha256 = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(_DIGEST_CHUNK_SIZE), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _require_directory(root: Path, label: str) -> None:
    if not root.exists():
        raise ValueError(f"{label} root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"{label} root is not a directory: {root}")


def _group_sort_key(path: Path) -> Tuple[int, object, str]:
    name = path.name
    if name.isdigit():
        return (0, int(name), name)
    return (1, name.casefold(), name)


def _unique_by_basename(paths: Iterable[Path], source: str) -> Dict[str, Path]:
    unique: Dict[str, Path] = {}
    duplicates: Dict[str, Set[str]] = {}
    for path in paths:
        basename_key = path.name.casefold()
        if basename_key in unique:
            duplicate_names = duplicates.setdefault(
                basename_key,
                {unique[basename_key].name},
            )
            duplicate_names.add(path.name)
        else:
            unique[basename_key] = path
    if duplicates:
        duplicate_groups = []
        for basename_key in sorted(duplicates):
            original_names = sorted(
                duplicates[basename_key],
                key=lambda name: (name.casefold(), name),
            )
            duplicate_groups.append(" / ".join(original_names))
        raise ValueError(
            f"duplicate basename(s) in {source}: {', '.join(duplicate_groups)}"
        )
    return unique


def _record(path: Path, source: str, group_id: Optional[str] = None) -> FileRecord:
    return FileRecord(
        name=path.name,
        path=path,
        size=path.stat().st_size,
        sha256=digest(path),
        source=source,
        group_id=group_id,
    )


def _manifest_digest(records: Sequence[FileRecord]) -> str:
    canonical_records = [
        {
            "group_id": record.group_id,
            "name": record.name,
            "sha256": record.sha256,
            "size": record.size,
            "source": record.source,
        }
        for record in records
    ]
    canonical_json = json.dumps(
        canonical_records,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _summarize_names(label: str, names: Sequence[str]) -> str:
    shown_names = names[:_ERROR_NAME_LIMIT]
    return (
        f"{label} (total={len(names)}, shown={len(shown_names)}): "
        + ", ".join(shown_names)
    )


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve(strict=False)
    second = second.resolve(strict=False)
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def _is_within(path: Path, root: Path) -> bool:
    path = path.resolve(strict=False)
    root = root.resolve(strict=False)
    return path == root or root in path.parents


def _atomic_verified_replace(
    source_path: Path,
    target_path: Path,
    expected_sha256: str,
    name: str,
    temporary_prefix: str,
    staging_error: str,
    final_error: str,
) -> None:
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target_path.parent,
            prefix=temporary_prefix,
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        shutil.copy2(source_path, temporary_path)
        if digest(temporary_path) != expected_sha256:
            raise OSError(f"{staging_error}: {name}")
        os.replace(temporary_path, target_path)
        temporary_path = None
        if digest(target_path) != expected_sha256:
            raise OSError(f"{final_error}: {name}")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _rollback_batch(
    mismatches: Sequence[Mismatch],
    backup_dir: Path,
) -> Tuple[Tuple[str, BaseException], ...]:
    failures = []
    for mismatch in mismatches:
        try:
            _atomic_verified_replace(
                source_path=backup_dir / mismatch.name,
                target_path=mismatch.merged.path,
                expected_sha256=mismatch.merged.sha256,
                name=mismatch.name,
                temporary_prefix=".ginseng-rollback-",
                staging_error="rollback staging verification failed",
                final_error="rollback verification failed",
            )
        except BaseException as error:
            failures.append((mismatch.name, error))
    return tuple(failures)


def _failure_summary(error: BaseException) -> str:
    message = str(error).strip()
    if message:
        return f"{type(error).__name__}: {message}"
    return type(error).__name__


def audit_sources(
    library: Path,
    test: Path,
    merged: Path,
    expected_groups: int,
) -> AuditReport:
    """Audit source trees against a merged gallery without modifying any input."""
    library = Path(library)
    test = Path(test)
    merged = Path(merged)
    for root, label in (
        (library, "library"),
        (test, "test"),
        (merged, "merged"),
    ):
        _require_directory(root, label)

    group_directories = tuple(
        sorted((path for path in test.iterdir() if path.is_dir()), key=_group_sort_key)
    )
    group_count = len(group_directories)
    if group_count != expected_groups:
        raise ValueError(
            f"test group count mismatch: expected {expected_groups}, found {group_count}"
        )

    library_by_key = _unique_by_basename(image_files(library), "library")
    test_by_key = _unique_by_basename(image_files(test), "test")
    merged_by_key = _unique_by_basename(image_files(merged), "merged")

    duplicate_source_keys = sorted(set(library_by_key).intersection(test_by_key))
    if duplicate_source_keys:
        duplicate_names = ", ".join(
            f"{library_by_key[key].name} / {test_by_key[key].name}"
            for key in duplicate_source_keys
        )
        raise ValueError(
            f"basename(s) present in both library and test: {duplicate_names}"
        )

    source_by_key = dict(library_by_key)
    source_by_key.update(test_by_key)
    source_keys = set(source_by_key)
    merged_keys = set(merged_by_key)
    missing_names = [
        source_by_key[key].name for key in sorted(source_keys.difference(merged_keys))
    ]
    extra_names = [
        merged_by_key[key].name for key in sorted(merged_keys.difference(source_keys))
    ]
    if missing_names or extra_names:
        details = []
        if missing_names:
            details.append(_summarize_names("missing from merged", missing_names))
        if extra_names:
            details.append(_summarize_names("extra in merged", extra_names))
        raise ValueError("merged basename set mismatch; " + "; ".join(details))

    immediate_groups = set(group_directories)
    records = []
    for path in library_by_key.values():
        records.append(_record(path, "library"))
    for path in test_by_key.values():
        if path.parent not in immediate_groups:
            raise ValueError(
                f"test image must be directly inside a group directory: {path}"
            )
        records.append(_record(path, "test", group_id=path.parent.name))
    records.sort(
        key=lambda record: (record.source, record.name.casefold(), record.name)
    )
    stable_records = tuple(records)

    records_by_key = {record.name.casefold(): record for record in stable_records}
    mismatches = []
    for basename_key in sorted(source_keys):
        source_record = records_by_key[basename_key]
        merged_record = _record(merged_by_key[basename_key], "merged")
        if source_record.sha256 != merged_record.sha256:
            mismatches.append(
                Mismatch(
                    name=source_record.name,
                    source=source_record,
                    merged=merged_record,
                )
            )

    return AuditReport(
        library_root=library.resolve(),
        test_root=test.resolve(),
        merged_root=merged.resolve(),
        library_count=len(library_by_key),
        test_count=len(test_by_key),
        merged_count=len(merged_by_key),
        group_count=group_count,
        mismatches=tuple(mismatches),
        records=stable_records,
        manifest_sha256=_manifest_digest(stable_records),
    )


def repair_mismatches(report: AuditReport, backup_dir: Path) -> int:
    """Back up mismatched merged files and restore their audited source bytes."""
    if not report.mismatches:
        return 0
    backup_dir = Path(backup_dir)
    for root in (report.library_root, report.test_root, report.merged_root):
        if _paths_overlap(backup_dir, root):
            raise ValueError("backup directory must be outside all data roots")
    backup_name_keys: Set[str] = set()
    for mismatch in report.mismatches:
        source_path = mismatch.source.path.resolve(strict=False)
        merged_path = mismatch.merged.path.resolve(strict=False)
        if source_path == merged_path:
            raise ValueError(
                f"source and merged resolve to the same file: {mismatch.name}"
            )
        if mismatch.source.source == "library":
            source_root = report.library_root
        elif mismatch.source.source == "test":
            source_root = report.test_root
        else:
            raise ValueError(f"unsupported mismatch source: {mismatch.source.source}")
        if not _is_within(source_path, source_root):
            raise ValueError(
                f"source file is outside its audited root: {mismatch.name}"
            )
        if mismatch.merged.source != "merged" or not _is_within(
            merged_path, report.merged_root
        ):
            raise ValueError(
                f"merged file is outside its audited root: {mismatch.name}"
            )
        if _paths_overlap(backup_dir, source_path.parent) or _paths_overlap(
            backup_dir, merged_path.parent
        ):
            raise ValueError("backup directory overlaps an audited file tree")
        if Path(mismatch.name).name != mismatch.name or mismatch.name in {
            "",
            ".",
            "..",
        }:
            raise ValueError("mismatch name is not a stable basename")
        backup_name_key = mismatch.name.casefold()
        if backup_name_key in backup_name_keys:
            raise ValueError(f"backup basename collision: {mismatch.name}")
        backup_name_keys.add(backup_name_key)
        if digest(mismatch.source.path) != mismatch.source.sha256:
            raise ValueError(f"source changed since audit: {mismatch.name}")
        if digest(mismatch.merged.path) != mismatch.merged.sha256:
            raise ValueError(f"merged file changed since audit: {mismatch.name}")
    backup_dir.mkdir(parents=True, exist_ok=False)
    for mismatch in report.mismatches:
        backup_path = backup_dir / mismatch.name
        shutil.copy2(mismatch.merged.path, backup_path)
        if digest(backup_path) != mismatch.merged.sha256:
            raise OSError(f"backup verification failed: {mismatch.name}")
    try:
        for mismatch in report.mismatches:
            if digest(mismatch.source.path) != mismatch.source.sha256:
                raise ValueError(f"source changed since audit: {mismatch.name}")
            if digest(mismatch.merged.path) != mismatch.merged.sha256:
                raise ValueError(f"merged file changed since audit: {mismatch.name}")
            _atomic_verified_replace(
                source_path=mismatch.source.path,
                target_path=mismatch.merged.path,
                expected_sha256=mismatch.source.sha256,
                name=mismatch.name,
                temporary_prefix=".ginseng-repair-",
                staging_error="repair staging verification failed",
                final_error="repair verification failed",
            )
    except BaseException as repair_error:
        rollback_failures = _rollback_batch(report.mismatches, backup_dir)
        if rollback_failures:
            rollback_summary = "; ".join(
                f"{name} ({_failure_summary(error)})"
                for name, error in rollback_failures
            )
            raise RuntimeError(
                "repair failed "
                f"({_failure_summary(repair_error)}); "
                f"rollback failed ({rollback_summary})"
            ) from repair_error
        raise
    return len(report.mismatches)
