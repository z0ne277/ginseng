#!/usr/bin/env python
"""Audit the immutable ginseng benchmark sources against the merged gallery."""

import argparse
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Dict, Optional, Sequence

from ginseng_benchmark.env import load_env_file
from ginseng_benchmark.protocol import AuditReport, FileRecord, Mismatch, audit_sources


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit source and merged ginseng datasets by SHA-256 content hash."
    )
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/manifests/audit.json"),
    )
    parser.add_argument("--expected-groups", type=int, default=271)
    parser.add_argument("--expected-library-count", type=int, default=11_712)
    parser.add_argument("--expected-test-count", type=int, default=1_075)
    parser.add_argument("--expected-merged-count", type=int, default=12_787)
    parser.add_argument("--expected-mismatches", type=int)
    return parser


def _required_path(environment: Dict[str, str], key: str) -> Path:
    value = environment.get(key, "").strip()
    if not value:
        raise ValueError(f"missing required path in env file: {key}")
    return Path(value)


def _record_payload(record: FileRecord) -> Dict[str, object]:
    if record.source == "test":
        if record.group_id is None:
            raise ValueError("test record is missing group_id")
        serialized_path = PurePosixPath(record.group_id, record.name).as_posix()
    elif record.source in {"library", "merged"}:
        serialized_path = record.name
    else:
        raise ValueError(f"unsupported record source: {record.source}")
    return {
        "group_id": record.group_id,
        "name": record.name,
        "path": serialized_path,
        "sha256": record.sha256,
        "size": record.size,
        "source": record.source,
    }


def _mismatch_payload(mismatch: Mismatch) -> Dict[str, object]:
    return {
        "merged": _record_payload(mismatch.merged),
        "name": mismatch.name,
        "source": _record_payload(mismatch.source),
    }


def _report_payload(report: AuditReport) -> Dict[str, object]:
    return {
        "manifest_sha256": report.manifest_sha256,
        "mismatches": [
            _mismatch_payload(mismatch) for mismatch in report.mismatches
        ],
        "records": [_record_payload(record) for record in report.records],
        "report": {
            "group_count": report.group_count,
            "library_count": report.library_count,
            "merged_count": report.merged_count,
            "mismatch_count": len(report.mismatches),
            "record_count": len(report.records),
            "test_count": report.test_count,
        },
    }


def _validation_errors(
    report: AuditReport,
    expected_library_count: int,
    expected_test_count: int,
    expected_merged_count: int,
    expected_mismatches: Optional[int],
) -> Sequence[str]:
    checks = (
        ("library count", expected_library_count, report.library_count),
        ("test count", expected_test_count, report.test_count),
        ("merged count", expected_merged_count, report.merged_count),
    )
    errors = [
        f"{label} mismatch: expected {expected}, found {actual}"
        for label, expected, actual in checks
        if expected != actual
    ]
    if expected_mismatches is not None and expected_mismatches != len(
        report.mismatches
    ):
        errors.append(
            "mismatch count mismatch: "
            f"expected {expected_mismatches}, found {len(report.mismatches)}"
        )
    return errors


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        environment = load_env_file(args.env)
        report = audit_sources(
            library=_required_path(environment, "LIBRARY_BINARY"),
            test=_required_path(environment, "TEST_BINARY_ROOT"),
            merged=_required_path(environment, "MERGED_GALLERY"),
            expected_groups=args.expected_groups,
        )
        payload = _report_payload(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as error:
        print(f"audit failed: {error}", file=sys.stderr)
        return 2

    errors = _validation_errors(
        report,
        expected_library_count=args.expected_library_count,
        expected_test_count=args.expected_test_count,
        expected_merged_count=args.expected_merged_count,
        expected_mismatches=args.expected_mismatches,
    )
    if errors:
        for error in errors:
            print(f"audit validation failed: {error}", file=sys.stderr)
        return 1

    print(
        "audit passed: "
        f"library={report.library_count}, "
        f"test={report.test_count}, "
        f"merged={report.merged_count}, "
        f"groups={report.group_count}, "
        f"mismatches={len(report.mismatches)}, "
        f"records={len(report.records)}"
    )
    print(f"manifest_sha256={report.manifest_sha256}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
