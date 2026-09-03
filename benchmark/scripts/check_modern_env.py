#!/usr/bin/env python
"""Fail-fast verification for the isolated public-baseline environment."""

from importlib import metadata
import sys


EXPECTED = {
    "torch": "2.7.1",
    "torchvision": "0.22.1",
    "transformers": "4.57.0",
    "huggingface-hub": "0.35.3",
    "safetensors": "0.6.2",
    "timm": "1.0.17",
    "yacs": "0.1.8",
    "pillow": "11.3.0",
    "numpy": "2.1.3",
}


def _base_version(value: str) -> str:
    return value.split("+", 1)[0]


def main() -> int:
    errors = []
    print(f"environment=ginseng-baselines python={sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 11):
        errors.append(f"python expected 3.11, found {sys.version.split()[0]}")

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
            errors.append("CUDA is unavailable in ginseng-baselines")
        else:
            print(f"gpu={torch.cuda.get_device_name(0)}")
        required_model_types = ("siglip", "clip", "swinv2", "convnextv2")
        for model_type in required_model_types:
            if model_type not in CONFIG_MAPPING:
                errors.append(
                    f"transformers does not register the {model_type} architecture"
                )
    except Exception as error:  # pragma: no cover - diagnostic boundary
        errors.append(f"runtime import check failed: {error}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print("modern environment check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
