#!/usr/bin/env python
"""Evaluate perturbed-query features against an unchanged clean gallery."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

from ginseng_benchmark.cache import load_feature_cache, load_trusted_torch_cache
from ginseng_benchmark.evaluation import (
    load_query_protocol,
    write_evaluation_json_atomic,
    write_per_query_csv_atomic,
)
from ginseng_benchmark.robustness import evaluate_shifted_queries


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate shifted query features against a clean feature cache."
    )
    parser.add_argument("--clean-cache", type=Path, required=True)
    parser.add_argument("--shifted-cache", type=Path, required=True)
    parser.add_argument("--query-groups", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-query-csv", type=Path)
    parser.add_argument("--ks", default="1,5,10,20")
    parser.add_argument("--expected-query-count", type=int, default=1075)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--trusted-local-pt", action="store_true")
    return parser


def _parse_ks(value: str) -> tuple[int, ...]:
    try:
        ks = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("--ks must contain comma-separated integers") from error
    if not ks or any(item <= 0 for item in ks):
        raise ValueError("--ks must contain positive integers")
    return ks


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        if not args.trusted_local_pt:
            raise ValueError(
                "refusing pickle-backed shifted cache; pass --trusted-local-pt "
                "only for a trusted local extractor output"
            )
        protocol = load_query_protocol(args.query_groups)
        clean_cache = load_feature_cache(
            args.clean_cache,
            expected_dataset_manifest_sha256=protocol["metadata"][
                "dataset_manifest_sha256"
            ],
        )
        shifted_features, shifted_paths = load_trusted_torch_cache(
            args.shifted_cache,
            trusted_local_pt=True,
        )
        if shifted_features.shape[0] != args.expected_query_count:
            raise ValueError(
                "shifted query count mismatch: "
                f"expected {args.expected_query_count}, "
                f"found {shifted_features.shape[0]}"
            )
        result = evaluate_shifted_queries(
            clean_cache,
            shifted_features=shifted_features,
            shifted_paths=shifted_paths,
            query_protocol=protocol,
            condition=args.condition,
            ks=_parse_ks(args.ks),
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
        )
        write_evaluation_json_atomic(result, args.output)
        if args.per_query_csv is not None:
            write_per_query_csv_atomic(result["per_query"], args.per_query_csv)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"query-shift evaluation failed: {error}", file=sys.stderr)
        return 2

    macro = result["aggregate"]["macro"]
    print(
        "query_shift_evaluation_complete "
        f"model={result['metadata']['model_id']} "
        f"condition={args.condition} "
        f"queries={result['metadata']['query_count']} "
        f"mAP={macro['map']:.6f} MRR={macro['mrr']:.6f}",
        flush=True,
    )
    for k in result["metadata"]["ks"]:
        print(
            f"K={k}: recall={macro[f'recall@{k}']:.6f}, "
            f"hit={macro[f'hit@{k}']:.6f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
