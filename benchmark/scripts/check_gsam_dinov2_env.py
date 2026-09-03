#!/usr/bin/env python
"""Verify DINOv2 support without allowing the legacy torch stack to drift."""

from importlib import metadata
import sys


EXPECTED = {
    "torch": "1.13.1",
    "torchvision": "0.14.1",
    "transformers": "4.35.2",
    "tokenizers": "0.15.2",
}


def _base_version(value: str) -> str:
    return value.split("+", 1)[0]


def main() -> int:
    errors = []
    print(f"environment=gsam python={sys.version.split()[0]}")
    for package, expected in EXPECTED.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            errors.append(f"missing package: {package}=={expected}")
            continue
        print(f"{package}={actual}")
        if _base_version(actual) != expected:
            errors.append(f"{package} expected {expected}, found {actual}")

    try:
        import torch
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        print(f"cuda_available={torch.cuda.is_available()} cuda_runtime={torch.version.cuda}")
        if not torch.cuda.is_available():
            errors.append("CUDA is unavailable in gsam")
        if "dinov2" not in CONFIG_MAPPING:
            errors.append("transformers does not register the dinov2 architecture")
    except Exception as error:  # pragma: no cover - diagnostic boundary
        errors.append(f"runtime import check failed: {error}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print("gsam DINOv2 environment check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
