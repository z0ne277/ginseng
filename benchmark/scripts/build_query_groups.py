#!/usr/bin/env python
"""Audit the canonical dataset and atomically build query_groups.json."""

import argparse
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from ginseng_benchmark.env import load_env_file
from ginseng_benchmark.protocol import AuditReport, audit_sources
from ginseng_benchmark.query_groups import (
    build_query_protocol,
    write_query_protocol_atomic,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic query_groups.json after a full data audit."
    )
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/manifests/query_groups.json"),
    )
    parser.add_argument("--expected-groups", type=int, default=271)
    parser.add_argument("--expected-library-count", type=int, default=11_712)
    parser.add_argument("--expected-query-count", type=int, default=1_075)
    parser.add_argument("--expected-gallery-count", type=int, default=12_787)
    parser.add_argument(
        "--expected-positive-distribution",
        default="1:6,2:9,3:1060",
        help="Comma-separated positive-count:query-count pairs.",
    )
    return parser


def _required_path(environment: Dict[str, str], key: str) -> Path:
    value = environment.get(key, "").strip()
    if not value:
        raise ValueError(f"missing required path in env file: {key}")
    return Path(value)


def _parse_distribution(raw_value: str) -> Dict[str, int]:
    distribution: Dict[str, int] = {}
    for raw_item in raw_value.split(","):
        item = raw_item.strip()
        if not item or ":" not in item:
            raise ValueError(
                "expected positive distribution must use count:queries pairs"
            )
        raw_positive_count, raw_query_count = item.split(":", 1)
        try:
            positive_count = int(raw_positive_count.strip())
            query_count = int(raw_query_count.strip())
        except ValueError as error:
            raise ValueError(
                "expected positive distribution values must be integers"
            ) from error
        if positive_count < 1 or query_count < 1:
            raise ValueError(
                "expected positive distribution values must be positive"
            )
        key = str(positive_count)
        if key in distribution:
            raise ValueError(f"duplicate positive count in distribution: {key}")
        distribution[key] = query_count
    return {key: distribution[key] for key in sorted(distribution, key=int)}


def _audit_errors(
    report: AuditReport,
    expected_library_count: int,
    expected_query_count: int,
    expected_gallery_count: int,
) -> Sequence[str]:
    checks = (
        ("library count", expected_library_count, report.library_count),
        ("query count", expected_query_count, report.test_count),
        ("gallery count", expected_gallery_count, report.merged_count),
    )
    errors = [
        f"{label} mismatch: expected {expected}, found {actual}"
        for label, expected, actual in checks
        if expected != actual
    ]
    if report.mismatches:
        errors.append(
            "dataset audit mismatch count must be 0 before query generation; "
            f"found {len(report.mismatches)}"
        )
    return errors


def _protocol_errors(
    payload: Mapping[str, object],
    expected_groups: int,
    expected_query_count: int,
    expected_gallery_count: int,
    expected_positive_distribution: Mapping[str, int],
) -> Sequence[str]:
    metadata = payload["metadata"]
    errors = []
    checks = (
        ("group count", expected_groups, metadata["group_count"]),
        ("query count", expected_query_count, metadata["query_count"]),
        ("gallery count", expected_gallery_count, metadata["gallery_count"]),
    )
    errors.extend(
        f"{label} mismatch: expected {expected}, found {actual}"
        for label, expected, actual in checks
        if expected != actual
    )
    actual_distribution = metadata["positive_count_distribution"]
    if actual_distribution != expected_positive_distribution:
        errors.append(
            "positive count distribution mismatch: "
            f"expected {dict(expected_positive_distribution)}, "
            f"found {actual_distribution}"
        )
    return errors


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(arguments)
    try:
        environment = load_env_file(args.env)
        library_root = _required_path(environment, "LIBRARY_BINARY")
        test_root = _required_path(environment, "TEST_BINARY_ROOT")
        gallery_root = _required_path(environment, "MERGED_GALLERY")
        expected_distribution = _parse_distribution(
            args.expected_positive_distribution
        )
        report = audit_sources(
            library_root,
            test_root,
            gallery_root,
            expected_groups=args.expected_groups,
        )
        audit_errors = _audit_errors(
            report,
            args.expected_library_count,
            args.expected_query_count,
            args.expected_gallery_count,
        )
        if audit_errors:
            raise ValueError("; ".join(audit_errors))

        payload = build_query_protocol(
            test_root,
            gallery_root,
            dataset_manifest_sha256=report.manifest_sha256,
            gallery_count=report.merged_count,
            expected_groups=args.expected_groups,
        )
        protocol_errors = _protocol_errors(
            payload,
            args.expected_groups,
            args.expected_query_count,
            args.expected_gallery_count,
            expected_distribution,
        )
        if protocol_errors:
            raise ValueError("; ".join(protocol_errors))

        write_query_protocol_atomic(payload, args.output)
        metadata = payload["metadata"]
        print(
            f"groups={metadata['group_count']} "
            f"queries={metadata['query_count']} "
            f"gallery={metadata['gallery_count']} "
            "missing=0 self_positive=0 "
            f"protocol_sha256={metadata['query_protocol_sha256']}"
        )
        return 0
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
