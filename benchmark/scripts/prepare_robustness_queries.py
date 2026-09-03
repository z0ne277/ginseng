#!/usr/bin/env python
"""Materialize deterministic perturbations for canonical test queries only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Optional, Sequence

import numpy as np
from PIL import Image

from ginseng_benchmark.robustness import apply_query_perturbation


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a flat, perturbed copy of canonical query images."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--query-groups", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--severity", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-count", type=int, default=1075)
    return parser


def _stable_seed(base_seed: int, basename: str) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{basename.casefold()}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _source_index(source_root: Path) -> dict[str, Path]:
    root = Path(source_root).resolve(strict=True)
    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_SUFFIXES:
            continue
        key = path.name.casefold()
        if key in index:
            raise ValueError(
                f"duplicate query basename below source root: {path.name}"
            )
        index[key] = path
    if not index:
        raise ValueError("source root contains no supported images")
    return index


def _query_basenames(protocol: Mapping[str, object]) -> tuple[str, ...]:
    groups = protocol.get("query_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("query protocol requires a non-empty query_groups list")
    names = []
    seen = set()
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"query group {index} must be an object")
        raw_name = group.get("query_image")
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError(f"query group {index} has no query_image")
        name = Path(raw_name.replace("\\", "/")).name
        key = name.casefold()
        if key in seen:
            raise ValueError(f"duplicate query basename in protocol: {name}")
        seen.add(key)
        names.append(name)
    return tuple(names)


def _save_rgb(image: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pil_image = Image.fromarray(image, mode="RGB")
    suffix = output.suffix.casefold()
    if suffix in {".jpg", ".jpeg"}:
        pil_image.save(output, quality=95, subsampling=0)
    elif suffix in {".tif", ".tiff"}:
        pil_image.save(output, compression="tiff_deflate")
    else:
        pil_image.save(output)


def prepare_queries(
    *,
    source_root: Path,
    output_root: Path,
    protocol: Mapping[str, object],
    kind: str,
    severity: int,
    seed: int,
) -> int:
    """Create one perturbed image per canonical query and preserve its basename."""
    source_by_name = _source_index(source_root)
    query_names = _query_basenames(protocol)
    output_root = Path(output_root)
    started = time.time()
    for index, query_name in enumerate(query_names, start=1):
        source = source_by_name.get(query_name.casefold())
        if source is None:
            raise ValueError(f"query image is missing below source root: {query_name}")
        with Image.open(source) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        shifted = apply_query_perturbation(
            rgb,
            kind=kind,
            severity=severity,
            seed=_stable_seed(seed, query_name),
        )
        _save_rgb(shifted, output_root / query_name)
        if index == 1 or index % 100 == 0 or index == len(query_names):
            elapsed = max(time.time() - started, 1e-6)
            rate = index / elapsed
            eta = (len(query_names) - index) / max(rate, 1e-6)
            print(
                f"[prepare:{kind}_s{severity}] {index}/{len(query_names)} "
                f"({index / len(query_names) * 100:.1f}%) "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )
    return len(query_names)


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        protocol = json.loads(args.query_groups.read_text(encoding="utf-8-sig"))
        count = prepare_queries(
            source_root=args.source_root,
            output_root=args.output_root,
            protocol=protocol,
            kind=args.kind,
            severity=args.severity,
            seed=args.seed,
        )
        if count != args.expected_count:
            raise ValueError(
                f"query count mismatch: expected {args.expected_count}, found {count}"
            )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"robustness query preparation failed: {error}", file=sys.stderr)
        return 2
    print(
        f"robustness_query_preparation_complete condition={args.kind}_s{args.severity} "
        f"count={count} output={args.output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
