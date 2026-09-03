#!/usr/bin/env python
"""Safely preview or repair content mismatches in the merged gallery."""

import argparse
from pathlib import Path
import sys
from typing import Dict, Optional, Sequence

from ginseng_benchmark.env import load_env_file
from ginseng_benchmark.protocol import audit_sources, repair_mismatches


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview or repair audited merged-gallery content mismatches."
    )
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--expected-groups", type=int, default=271)
    parser.add_argument("--expected-library-count", type=int, default=11_712)
    parser.add_argument("--expected-test-count", type=int, default=1_075)
    parser.add_argument("--expected-merged-count", type=int, default=12_787)
    parser.add_argument("--expected-mismatches", type=int, default=107)
    parser.add_argument("--backup-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser


def _required_path(environment: Dict[str, str], key: str) -> Path:
    value = environment.get(key, "").strip()
    if not value:
        raise ValueError(f"missing required path in env file: {key}")
    return Path(value)


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(arguments)
    if args.apply and args.backup_dir is None:
        print("repair failed: --backup-dir is required with --apply", file=sys.stderr)
        return 2
    try:
        environment = load_env_file(args.env)
        report = audit_sources(
            library=_required_path(environment, "LIBRARY_BINARY"),
            test=_required_path(environment, "TEST_BINARY_ROOT"),
            merged=_required_path(environment, "MERGED_GALLERY"),
            expected_groups=args.expected_groups,
        )
        count_checks = (
            ("library", args.expected_library_count, report.library_count),
            ("test", args.expected_test_count, report.test_count),
            ("merged", args.expected_merged_count, report.merged_count),
        )
        for label, expected, actual in count_checks:
            if actual != expected:
                raise ValueError(
                    f"{label} count mismatch: expected {expected}, found {actual}"
                )
        mismatch_count = len(report.mismatches)
        if mismatch_count != args.expected_mismatches:
            raise ValueError(
                "mismatch count mismatch: "
                f"expected {args.expected_mismatches}, found {mismatch_count}"
            )
        repaired = None
        if args.apply:
            repaired = repair_mismatches(report, args.backup_dir)
    except (OSError, ValueError) as error:
        print(f"repair failed: {error}", file=sys.stderr)
        return 2

    if repaired is not None:
        print(f"repaired={repaired} backup_dir={args.backup_dir.resolve(strict=False)}")
        return 0
    print(f"would_repair={mismatch_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
