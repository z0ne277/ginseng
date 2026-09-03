#!/usr/bin/env python
"""Convert a trusted legacy torch cache into the audited standard NPZ format."""

import argparse
import base64
import binascii
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence

from ginseng_benchmark.cache import (
    build_feature_cache,
    load_trusted_torch_cache,
    write_feature_cache_atomic,
)
from ginseng_benchmark.env import load_env_file
from ginseng_benchmark.protocol import AuditReport, audit_sources


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stamp a trusted local .pt feature cache as a manifest-bound NPZ."
    )
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-source", required=True)
    parser.add_argument(
        "--feature-normalization",
        choices=("l2", "none"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--preprocessing-json")
    parser.add_argument("--tta-json")
    parser.add_argument("--environment-json")
    parser.add_argument("--preprocessing-json-base64")
    parser.add_argument("--tta-json-base64")
    parser.add_argument("--environment-json-base64")
    parser.add_argument("--expected-feature-dim", type=int)
    parser.add_argument(
        "--trusted-local-pt",
        action="store_true",
        help="Acknowledge that torch .pt uses pickle and the local file is trusted.",
    )
    parser.add_argument("--expected-groups", type=int, default=271)
    parser.add_argument("--expected-library-count", type=int, default=11_712)
    parser.add_argument("--expected-test-count", type=int, default=1_075)
    parser.add_argument("--expected-merged-count", type=int, default=12_787)
    return parser


def _required_path(environment: Dict[str, str], key: str) -> Path:
    value = environment.get(key, "").strip()
    if not value:
        raise ValueError(f"missing required dataset path in env file: {key}")
    return Path(value)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _json_object(value: str, label: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(value, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} must be strict JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _json_object_argument(
    plain_value: Optional[str],
    base64_value: Optional[str],
    label: str,
) -> Dict[str, Any]:
    if plain_value is not None and base64_value is not None:
        raise ValueError(f"use only one of {label} and {label}-base64")
    value = plain_value if plain_value is not None else "{}"
    if base64_value is not None:
        try:
            padding = "=" * (-len(base64_value) % 4)
            value = base64.b64decode(
                (base64_value + padding).encode("ascii"),
                altchars=b"-_",
                validate=True,
            ).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError, binascii.Error) as error:
            raise ValueError(f"{label}-base64 must contain URL-safe UTF-8 JSON") from error
    return _json_object(value, label)


def _validate_audit_counts(
    report: AuditReport,
    *,
    expected_library_count: int,
    expected_test_count: int,
    expected_merged_count: int,
) -> None:
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
    if report.mismatches:
        errors.append(
            f"merged content mismatch count must be zero, found {len(report.mismatches)}"
        )
    if errors:
        raise ValueError("; ".join(errors))


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        if not args.trusted_local_pt:
            raise ValueError(
                "refusing pickle-backed .pt; pass --trusted-local-pt only for a trusted local cache"
            )
        preprocessing = _json_object_argument(
            args.preprocessing_json,
            args.preprocessing_json_base64,
            "preprocessing-json",
        )
        tta = _json_object_argument(
            args.tta_json,
            args.tta_json_base64,
            "tta-json",
        )
        environment_metadata = _json_object_argument(
            args.environment_json,
            args.environment_json_base64,
            "environment-json",
        )
        environment = load_env_file(args.env)
        report = audit_sources(
            library=_required_path(environment, "LIBRARY_BINARY"),
            test=_required_path(environment, "TEST_BINARY_ROOT"),
            merged=_required_path(environment, "MERGED_GALLERY"),
            expected_groups=args.expected_groups,
        )
        _validate_audit_counts(
            report,
            expected_library_count=args.expected_library_count,
            expected_test_count=args.expected_test_count,
            expected_merged_count=args.expected_merged_count,
        )
        raw_features, raw_paths = load_trusted_torch_cache(
            args.raw_cache,
            trusted_local_pt=True,
        )
        cache = build_feature_cache(
            raw_features=raw_features,
            raw_paths=raw_paths,
            report=report,
            model_id=args.model_id,
            model_source=args.model_source,
            feature_normalization=args.feature_normalization,
            checkpoint=args.checkpoint,
            preprocessing=preprocessing,
            tta=tta,
            environment=environment_metadata,
            expected_feature_dim=args.expected_feature_dim,
        )
        write_feature_cache_atomic(cache, args.output)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"feature-cache stamp failed: {error}", file=sys.stderr)
        return 2

    print(
        "feature-cache stamp passed: "
        f"model={cache.metadata['model_id']}, "
        f"count={cache.metadata['num_images']}, "
        f"dim={cache.metadata['feature_dim']}"
    )
    print(f"manifest_sha256={cache.metadata['dataset_manifest_sha256']}")
    print(f"features_sha256={cache.metadata['features_sha256']}")
    print(f"paths_sha256={cache.metadata['paths_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
