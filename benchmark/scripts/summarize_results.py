#!/usr/bin/env python
"""Validate evaluation JSON files and build paper-ready Markdown/CSV tables."""

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


KS = (1, 5, 10, 20)
METRICS = ("map", "mrr") + tuple(
    metric for k in KS for metric in (f"recall@{k}", f"hit@{k}")
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize validated retrieval results.")
    parser.add_argument("--config", type=Path, default=Path("configs/existing_models.json"))
    parser.add_argument("--input", type=Path, default=Path("artifacts/results"))
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("artifacts/tables/existing_models.md"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/tables/existing_models.csv"),
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path.name}")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not (-float("inf") < number < float("inf")):
        raise ValueError(f"{label} must be finite")
    return number


def _validated_result(
    payload: Mapping[str, object], expected_model_id: str
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]], Mapping[str, object]]:
    metadata = payload.get("metadata")
    aggregate = payload.get("aggregate")
    bootstrap = payload.get("bootstrap")
    if not isinstance(metadata, dict) or not isinstance(aggregate, dict) or not isinstance(bootstrap, dict):
        raise ValueError(f"result for {expected_model_id} is missing required sections")
    if metadata.get("model_id") != expected_model_id:
        raise ValueError(f"result model_id mismatch for {expected_model_id}")
    if metadata.get("ranking_scope") != "full":
        raise ValueError(f"result for {expected_model_id} is not full-ranking")
    if metadata.get("query_count") != 1075 or metadata.get("gallery_count") != 12787:
        raise ValueError(f"result protocol counts are invalid for {expected_model_id}")
    macro = aggregate.get("macro")
    ci_metrics = bootstrap.get("metrics")
    if not isinstance(macro, dict) or not isinstance(ci_metrics, dict):
        raise ValueError(f"result metrics are incomplete for {expected_model_id}")
    values: Dict[str, float] = {}
    intervals: Dict[str, Tuple[float, float]] = {}
    for metric in METRICS:
        if metric not in macro or metric not in ci_metrics:
            raise ValueError(f"result for {expected_model_id} lacks {metric}")
        values[metric] = _finite_number(macro[metric], f"{expected_model_id}.{metric}")
        interval = ci_metrics[metric]
        if not isinstance(interval, dict):
            raise ValueError(f"bootstrap interval is invalid for {expected_model_id}.{metric}")
        lower = _finite_number(interval.get("lower"), f"{expected_model_id}.{metric}.lower")
        upper = _finite_number(interval.get("upper"), f"{expected_model_id}.{metric}.upper")
        point = _finite_number(
            interval.get("point_estimate"), f"{expected_model_id}.{metric}.point"
        )
        if abs(point - values[metric]) > 1e-12 or lower > upper:
            raise ValueError(f"bootstrap interval disagrees with macro metric {expected_model_id}.{metric}")
        intervals[metric] = (lower, upper)
    return values, intervals, metadata


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _formatted(value: float, interval: Tuple[float, float]) -> str:
    return f"{value:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]"


def summarize(
    config: Mapping[str, object], input_root: Path, *, allow_missing: bool
) -> List[Dict[str, object]]:
    if config.get("protocol_tag") != "271_1075" or not isinstance(config.get("models"), list):
        raise ValueError("model config is not the 271_1075 protocol")
    rows = []
    shared_manifest = None
    shared_protocol = None
    for model in config["models"]:
        model_id = model["id"]
        result_path = input_root / f"{model_id}_271_1075.json"
        if not result_path.is_file():
            if not allow_missing:
                raise ValueError(f"missing result: {result_path.name}")
            rows.append(
                {"model_id": model_id, "display_name": model["display_name"], "status": "PENDING"}
            )
            continue
        payload = _load_json(result_path)
        values, intervals, metadata = _validated_result(payload, model_id)
        manifest = metadata.get("dataset_manifest_sha256")
        protocol = metadata.get("query_protocol_sha256")
        if shared_manifest is None:
            shared_manifest, shared_protocol = manifest, protocol
        elif manifest != shared_manifest or protocol != shared_protocol:
            raise ValueError("result files do not share one dataset/query protocol")
        row: Dict[str, object] = {
            "model_id": model_id,
            "display_name": model["display_name"],
            "status": "OK",
        }
        for metric in METRICS:
            row[metric] = values[metric]
            row[f"{metric}_ci_lower"] = intervals[metric][0]
            row[f"{metric}_ci_upper"] = intervals[metric][1]
        rows.append(row)
    return rows


def write_tables(rows: Sequence[Mapping[str, object]], markdown: Path, csv_path: Path) -> None:
    csv_fields = ["model_id", "display_name", "status"] + [
        field for metric in METRICS for field in (metric, f"{metric}_ci_lower", f"{metric}_ci_upper")
    ]

    def write_csv(handle) -> None:
        handle.write("\ufeff")
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    def write_markdown(handle) -> None:
        columns = ("map", "mrr", "recall@1", "recall@5", "recall@10", "recall@20", "hit@1")
        handle.write("# Existing-model results (271 groups / 1075 queries)\n\n")
        handle.write("All values use full-gallery ranking. Brackets are 95% identity-cluster bootstrap CIs.\n\n")
        handle.write("| Model | " + " | ".join(columns) + " |\n")
        handle.write("|---|" + "---:|" * len(columns) + "\n")
        for row in rows:
            if row["status"] != "OK":
                cells = ["PENDING"] * len(columns)
            else:
                cells = [
                    _formatted(
                        float(row[metric]),
                        (float(row[f"{metric}_ci_lower"]), float(row[f"{metric}_ci_upper"])),
                    )
                    for metric in columns
                ]
            handle.write(f"| {row['display_name']} | " + " | ".join(cells) + " |\n")

    _atomic_write(csv_path, write_csv)
    _atomic_write(markdown, write_markdown)


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        config = _load_json(args.config)
        rows = summarize(config, args.input, allow_missing=args.allow_missing)
        write_tables(rows, args.output_markdown, args.output_csv)
    except (OSError, TypeError, ValueError) as error:
        print(f"result summary failed: {error}")
        return 2
    completed = sum(row["status"] == "OK" for row in rows)
    print(f"result summary passed: completed={completed}, total={len(rows)}")
    print(f"markdown={args.output_markdown}")
    print(f"csv={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
