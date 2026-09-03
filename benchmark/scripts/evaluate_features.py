#!/usr/bin/env python
"""Evaluate one standardized feature cache with the canonical query protocol."""

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence, Tuple

from ginseng_benchmark.cache import load_feature_cache
from ginseng_benchmark.evaluation import (
    evaluate_feature_cache,
    load_query_protocol,
    write_evaluation_json_atomic,
    write_per_query_csv_atomic,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full-ranking manifest-bound retrieval evaluation."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--query-groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-query-csv", type=Path)
    parser.add_argument("--ks", default="1,5,10,20")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser


def _parse_ks(value: str) -> Tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("--ks must be comma-separated positive integers") from error
    if not values or any(item <= 0 for item in values):
        raise ValueError("--ks must be comma-separated positive integers")
    return values


def _validate_artifact_paths(args: argparse.Namespace) -> None:
    if args.output.suffix.casefold() != ".json":
        raise ValueError("--output must use the .json suffix")
    if args.per_query_csv is not None and args.per_query_csv.suffix.casefold() != ".csv":
        raise ValueError("--per-query-csv must use the .csv suffix")
    input_paths = {
        args.cache.resolve(strict=False),
        args.query_groups.resolve(strict=False),
    }
    output_paths = [args.output.resolve(strict=False)]
    if args.per_query_csv is not None:
        output_paths.append(args.per_query_csv.resolve(strict=False))
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("evaluation output paths must be distinct")
    if any(path in input_paths for path in output_paths):
        raise ValueError("evaluation outputs must not overwrite input artifacts")


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        _validate_artifact_paths(args)
        protocol = load_query_protocol(args.query_groups)
        cache = load_feature_cache(
            args.cache,
            expected_dataset_manifest_sha256=protocol["metadata"][
                "dataset_manifest_sha256"
            ],
        )
        result = evaluate_feature_cache(
            cache,
            protocol,
            ks=_parse_ks(args.ks),
            block_size=args.block_size,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
            confidence=args.confidence,
        )
        write_evaluation_json_atomic(result, args.output)
        if args.per_query_csv is not None:
            write_per_query_csv_atomic(result["per_query"], args.per_query_csv)
    except (OSError, TypeError, ValueError) as error:
        print(f"feature evaluation failed: {error}", file=sys.stderr)
        return 2

    macro = result["aggregate"]["macro"]
    print(
        "feature evaluation passed: "
        f"model={result['metadata']['model_id']}, "
        f"queries={result['metadata']['query_count']}, "
        f"gallery={result['metadata']['gallery_count']}, "
        f"mAP={macro['map']:.6f}, MRR={macro['mrr']:.6f}"
    )
    for k in result["metadata"]["ks"]:
        print(
            f"K={k}: recall={macro[f'recall@{k}']:.6f}, "
            f"hit={macro[f'hit@{k}']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
