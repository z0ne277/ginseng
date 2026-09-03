#!/usr/bin/env python
"""Paired identity-cluster bootstrap for two retrieval result CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import fmean
from typing import Dict, Iterable, Mapping, Sequence


DEFAULT_METRICS = ("map", "mrr", "hit@1", "recall@1", "recall@5", "recall@10")


def _indexed(rows: Iterable[Mapping[str, object]]) -> Dict[str, Mapping[str, object]]:
    indexed: Dict[str, Mapping[str, object]] = {}
    for row in rows:
        query = str(row["query_image"])
        if query in indexed:
            raise ValueError(f"duplicate query: {query}")
        indexed[query] = row
    return indexed


def _number(row: Mapping[str, object], metric: str, query: str) -> float:
    try:
        value = float(row[metric])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {metric} for query {query}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite {metric} for query {query}")
    return value


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_cluster_bootstrap(
    baseline_rows: Iterable[Mapping[str, object]],
    challenger_rows: Iterable[Mapping[str, object]],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    iterations: int = 10000,
    seed: int = 42,
) -> dict:
    """Return challenger-minus-baseline differences with cluster bootstrap CIs."""
    if iterations < 20:
        raise ValueError("iterations must be at least 20")
    baseline = _indexed(baseline_rows)
    challenger = _indexed(challenger_rows)
    if set(baseline) != set(challenger):
        raise ValueError("baseline and challenger query sets do not match")

    groups: Dict[str, list[str]] = {}
    for query, baseline_row in baseline.items():
        challenger_row = challenger[query]
        baseline_group = str(baseline_row["group_id"])
        challenger_group = str(challenger_row["group_id"])
        if baseline_group != challenger_group:
            raise ValueError(f"group mismatch for query {query}")
        groups.setdefault(baseline_group, []).append(query)
    if not groups:
        raise ValueError("no paired queries")

    group_ids = sorted(groups)
    per_group: Dict[str, Dict[str, float]] = {}
    for group_id, queries in groups.items():
        per_group[group_id] = {}
        for metric in metrics:
            per_group[group_id][metric] = fmean(
                _number(challenger[query], metric, query)
                - _number(baseline[query], metric, query)
                for query in queries
            )

    observed: Dict[str, float] = {}
    for metric in metrics:
        observed[metric] = fmean(
            _number(challenger[query], metric, query)
            - _number(baseline[query], metric, query)
            for query in baseline
        )

    rng = random.Random(seed)
    samples = {metric: [] for metric in metrics}
    for _ in range(iterations):
        sampled_groups = rng.choices(group_ids, k=len(group_ids))
        for metric in metrics:
            weighted_sum = 0.0
            sampled_queries = 0
            for group_id in sampled_groups:
                group_size = len(groups[group_id])
                weighted_sum += per_group[group_id][metric] * group_size
                sampled_queries += group_size
            samples[metric].append(weighted_sum / sampled_queries)

    metric_results = {}
    for metric in metrics:
        distribution = samples[metric]
        non_positive = sum(value <= 0 for value in distribution) / iterations
        non_negative = sum(value >= 0 for value in distribution) / iterations
        metric_results[metric] = {
            "difference": observed[metric],
            "ci_lower": _percentile(distribution, 0.025),
            "ci_upper": _percentile(distribution, 0.975),
            "two_sided_p": min(1.0, 2 * min(non_positive, non_negative)),
        }

    return {
        "sampling_unit": "identity_group",
        "confidence": 0.95,
        "iterations": iterations,
        "seed": seed,
        "group_count": len(group_ids),
        "query_count": len(baseline),
        "metrics": metric_results,
    }


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare retrieval systems with paired identity-cluster bootstrap."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--challenger-name", required=True)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    metrics = tuple(part.strip() for part in args.metrics.split(",") if part.strip())
    try:
        result = paired_cluster_bootstrap(
            _read_csv(args.baseline),
            _read_csv(args.challenger),
            metrics=metrics,
            iterations=args.iterations,
            seed=args.seed,
        )
        result["baseline"] = args.baseline_name
        result["challenger"] = args.challenger_name
        result["difference_direction"] = "challenger_minus_baseline"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except (OSError, ValueError) as error:
        print(f"paired bootstrap failed: {error}")
        return 2
    print(
        f"paired bootstrap passed: groups={result['group_count']}, "
        f"queries={result['query_count']}, output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
