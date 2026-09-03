#!/usr/bin/env python
"""Recompute SSI and group retrieval difficulty on the canonical protocol."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Optional, Sequence

import numpy as np

from ginseng_benchmark.cache import load_feature_cache
from ginseng_benchmark.evaluation import evaluate_feature_cache, load_query_protocol
from ginseng_benchmark.ssi import (
    assign_tertile_bands,
    attach_group_metrics,
    compute_group_ssi,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--query-groups", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    return parser


def _atomic_json(payload, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_csv(rows, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _draw(rows, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Low": "#C65D4B", "Mid": "#D8A03D", "High": "#3F7F73"}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), dpi=220)
    for band in ("Low", "Mid", "High"):
        subset = [row for row in rows if row["ssi_band"] == band]
        axes[0].scatter(
            [row["ssi"] for row in subset],
            [row["map"] for row in subset],
            s=16,
            alpha=0.65,
            color=colors[band],
            label=f"{band} SSI",
            edgecolors="none",
        )
    axes[0].set_xlabel("SSI")
    axes[0].set_ylabel("Group-level mAP")
    axes[0].grid(alpha=0.18)
    axes[0].legend(frameon=False, fontsize=8)

    labels = ("Low", "Mid", "High")
    means = [
        float(np.mean([row["map"] for row in rows if row["ssi_band"] == label]))
        for label in labels
    ]
    axes[1].bar(labels, means, color=[colors[label] for label in labels], width=0.62)
    axes[1].set_ylabel("Mean group-level mAP")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(axis="y", alpha=0.18)
    for index, value in enumerate(means):
        axes[1].text(index, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        protocol = load_query_protocol(args.query_groups)
        cache = load_feature_cache(
            args.cache,
            expected_dataset_manifest_sha256=protocol["metadata"][
                "dataset_manifest_sha256"
            ],
        )
        evaluation = evaluate_feature_cache(
            cache,
            protocol,
            ks=(1, 5, 10, 20),
            block_size=32,
            bootstrap_iterations=100,
            bootstrap_seed=42,
        )
        rows = assign_tertile_bands(
            attach_group_metrics(
                compute_group_ssi(cache, protocol),
                evaluation["per_query"],
            )
        )
        ssi_values = np.asarray([row["ssi"] for row in rows], dtype=np.float64)
        bands = {}
        for label in ("Low", "Mid", "High"):
            subset = [row for row in rows if row["ssi_band"] == label]
            bands[label] = {
                "group_count": len(subset),
                "ssi_min": min(float(row["ssi"]) for row in subset),
                "ssi_max": max(float(row["ssi"]) for row in subset),
                "map": float(np.mean([row["map"] for row in subset])),
                "recall@5": float(
                    np.mean([row["recall@5"] for row in subset])
                ),
                "recall@10": float(
                    np.mean([row["recall@10"] for row in subset])
                ),
            }
        payload = {
            "metadata": {
                "model_id": cache.metadata["model_id"],
                "group_count": len(rows),
                "query_count": protocol["metadata"]["query_count"],
                "dataset_manifest_sha256": cache.metadata[
                    "dataset_manifest_sha256"
                ],
                "query_protocol_sha256": protocol["metadata"][
                    "query_protocol_sha256"
                ],
                "definition": "0.5 * (1 + mean pairwise cosine similarity)",
                "interpretation": (
                    "post-hoc embedding cohesion, not an independent physical "
                    "shape annotation"
                ),
            },
            "summary": {
                "mean": float(np.mean(ssi_values)),
                "median": float(np.median(ssi_values)),
                "minimum": float(np.min(ssi_values)),
                "maximum": float(np.max(ssi_values)),
                "bands": bands,
            },
            "groups": rows,
        }
        _atomic_json(payload, args.output_json)
        _write_csv(rows, args.output_csv)
        if args.figure is not None:
            _draw(rows, args.figure)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"SSI analysis failed: {error}", file=sys.stderr)
        return 2

    summary = payload["summary"]
    print(
        f"ssi_analysis_complete groups={len(rows)} "
        f"mean={summary['mean']:.4f} median={summary['median']:.4f} "
        f"min={summary['minimum']:.4f} max={summary['maximum']:.4f}",
        flush=True,
    )
    for label, band in summary["bands"].items():
        print(
            f"{label}: groups={band['group_count']} "
            f"mAP={band['map']:.4f} R@5={band['recall@5']:.4f} "
            f"R@10={band['recall@10']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
