#!/usr/bin/env python
"""Extract deterministic frozen Hugging Face vision embeddings into a trusted raw cache."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Callable, Optional, Sequence

from ginseng_benchmark.env import load_env_file
from ginseng_benchmark.vision_features import (
    discover_flat_image_paths,
    forward_hf_features,
    l2_normalize_rows,
    validate_feature_collection,
)


_HF_TIMEOUT_DEFAULTS = {
    "HF_HUB_DOWNLOAD_TIMEOUT": "120",
    "HF_HUB_ETAG_TIMEOUT": "60",
}
_HF_ENV_KEYS = (
    "HF_ENDPOINT",
    "HF_HUB_DISABLE_SYMLINKS_WARNING",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract frozen HF vision features.")
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--expected-dim", type=int, required=True)
    parser.add_argument(
        "--extractor-kind",
        choices=("pooler_or_cls", "model_image_features"),
        default="pooler_or_cls",
    )
    parser.add_argument("--expected-count", type=int, default=12_787)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--token-env-key", default="HF_TOKEN")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _collate(batch):
    return batch


def _atomic_torch_save(payload, output: Path) -> None:
    import torch

    output = Path(output)
    if output.suffix.casefold() != ".pt":
        raise ValueError("raw feature output must use the .pt suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class _ImageDataset:
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from PIL import Image

        path = self.paths[index]
        with Image.open(path) as image:
            return image.convert("RGB"), str(path)


def _resolve_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _load_options(revision: str, token: str):
    common = {"revision": revision}
    if token:
        common["token"] = token
    processor_options = dict(common)
    # Pin the saved slow processor path; Transformers warns that the implicit
    # default may change and yield slightly different pixels in future releases.
    processor_options["use_fast"] = False
    model_options = dict(common)
    model_options["use_safetensors"] = None
    return processor_options, model_options


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _format_progress(
    *,
    processed: int,
    total: int,
    elapsed_seconds: float,
    width: int = 16,
) -> str:
    if total <= 0 or processed < 0 or processed > total:
        raise ValueError("progress counts are invalid")
    fraction = processed / total
    filled = min(width, int(fraction * width))
    bar = "#" * filled + "-" * (width - filled)
    rate = processed / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining = (total - processed) / rate if rate > 0 else 0.0
    return (
        f"[{bar}] {processed}/{total} ({fraction * 100:5.1f}%) "
        f"elapsed={_format_duration(elapsed_seconds)} "
        f"ETA={_format_duration(remaining)}"
    )


class _ProgressReporter:
    def __init__(
        self,
        *,
        total: int,
        update_every: int,
        clock: Callable[[], float] = time.monotonic,
        emit: Callable[[str], None] = lambda line: print(line, flush=True),
    ) -> None:
        if total <= 0 or update_every <= 0:
            raise ValueError("progress total/update interval must be positive")
        self.total = total
        self.update_every = update_every
        self.clock = clock
        self.emit = emit
        self.started_at = clock()
        self.next_update = update_every
        self.has_reported = False

    def update(self, processed: int) -> None:
        should_report = (
            not self.has_reported
            or processed >= self.total
            or processed >= self.next_update
        )
        if not should_report:
            return
        elapsed = self.clock() - self.started_at
        self.emit(
            "[stage 4/5] Extracting features "
            + _format_progress(
                processed=min(processed, self.total),
                total=self.total,
                elapsed_seconds=elapsed,
            )
        )
        self.has_reported = True
        while self.next_update <= processed:
            self.next_update += self.update_every


def _configure_hf_runtime(environment, *, repo_root: Optional[Path] = None):
    """Apply non-secret Hub settings before transformers imports huggingface_hub."""
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    default_hf_home = (repo_root / "artifacts" / "models" / "huggingface").resolve()
    hf_home = Path(str(environment.get("HF_HOME") or default_hf_home)).expanduser().resolve()
    hub_cache = Path(
        str(environment.get("HF_HUB_CACHE") or (hf_home / "hub"))
    ).expanduser().resolve()
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = str(
        environment.get("HF_HUB_DISABLE_PROGRESS_BARS") or "0"
    )

    for key in _HF_ENV_KEYS:
        value = str(environment.get(key, "")).strip()
        if value:
            os.environ[key] = value

    settings = {}
    for key, default in _HF_TIMEOUT_DEFAULTS.items():
        raw_value = str(environment.get(key) or os.environ.get(key) or default).strip()
        try:
            value = int(raw_value)
        except ValueError as error:
            raise ValueError(f"{key} must be a positive integer") from error
        if value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        os.environ[key] = str(value)
        settings[key] = value

    return {
        "endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        "download_timeout": settings["HF_HUB_DOWNLOAD_TIMEOUT"],
        "etag_timeout": settings["HF_HUB_ETAG_TIMEOUT"],
        "proxy_configured": bool(os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")),
        "hf_home": str(hf_home),
        "hub_cache": str(hub_cache),
    }


def _format_hf_error(error: Exception) -> str:
    message = str(error)
    normalized = message.casefold()
    if any(marker in normalized for marker in ("gatedrepoerror", "401", "403", "access to model")):
        return (
            "Hugging Face 模型访问被拒绝。请先在模型页面接受许可，再在 .env 中填写"
            "有读取权限的 HF_TOKEN。原始错误: " + message
        )
    if any(marker in normalized for marker in ("read timed out", "readtimeout", "connecttimeout")):
        return (
            "Hugging Face 下载超时。请检查代理，并在 .env 中增大 "
            "HF_HUB_DOWNLOAD_TIMEOUT/HF_HUB_ETAG_TIMEOUT。原始错误: " + message
        )
    return message


def main(arguments: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        if args.expected_dim <= 0 or args.expected_count <= 0:
            raise ValueError("expected dimension/count must be positive")
        if args.batch_size <= 0 or args.num_workers < 0:
            raise ValueError("batch size must be positive and workers non-negative")
        print(f"[stage 1/5] Scanning gallery: {args.image_dir}", flush=True)
        paths = discover_flat_image_paths(args.image_dir)
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("limit must be positive")
            paths = paths[: args.limit]
        elif len(paths) != args.expected_count:
            raise ValueError(
                f"gallery count mismatch: expected {args.expected_count}, found {len(paths)}"
            )
        print(f"[stage 1/5] Gallery ready: {len(paths)} images", flush=True)
        if args.dry_run:
            print(
                f"HF extraction dry-run: model={args.model}@{args.revision}, "
                f"images={len(paths)}, dim={args.expected_dim}, "
                f"extractor={args.extractor_kind}, output={args.output}"
            )
            return 0

        environment = load_env_file(args.env)
        hub_settings = _configure_hf_runtime(environment)
        token = environment.get(args.token_env_key, "").strip()
        print(
            "HF Hub settings: "
            f"endpoint={hub_settings['endpoint']}, "
            f"download_timeout={hub_settings['download_timeout']}s, "
            f"etag_timeout={hub_settings['etag_timeout']}s, "
            f"proxy={'set' if hub_settings['proxy_configured'] else 'unset'}, "
            f"token={'set' if token else 'unset'}, "
            f"cache={hub_settings['hub_cache']}",
            flush=True,
        )

        import torch
        from torch.utils.data import DataLoader
        import transformers
        from transformers import AutoImageProcessor, AutoModel

        processor_options, model_options = _load_options(args.revision, token)
        print(
            f"[stage 2/5] Checking/downloading model: {args.model}@{args.revision}",
            flush=True,
        )
        processor = AutoImageProcessor.from_pretrained(args.model, **processor_options)
        model = AutoModel.from_pretrained(args.model, **model_options)
        device = _resolve_device(args.device)
        model.eval().to(device)
        print(
            f"[stage 3/5] Model ready: device={device}, batch_size={args.batch_size}, "
            f"workers={args.num_workers}",
            flush=True,
        )
        loader = DataLoader(
            _ImageDataset(paths),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_collate,
            pin_memory=device.type == "cuda",
        )
        feature_batches = []
        output_paths = []
        progress = _ProgressReporter(
            total=len(paths),
            update_every=max(args.batch_size, math.ceil(len(paths) / 100)),
        )
        with torch.inference_mode():
            for batch in loader:
                images = [item[0] for item in batch]
                output_paths.extend(item[1] for item in batch)
                inputs = processor(images=images, return_tensors="pt")
                inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
                pooled = forward_hf_features(model, inputs, args.extractor_kind)
                feature_batches.append(l2_normalize_rows(pooled).cpu())
                progress.update(len(output_paths))
        features = torch.cat(feature_batches, dim=0).contiguous()
        print(
            f"[stage 5/5] Validating and saving cache: {args.output}",
            flush=True,
        )
        validate_feature_collection(features, paths, args.expected_dim)
        resolved_commit = getattr(model.config, "_commit_hash", None) or args.revision
        payload = {
            "features": features,
            "paths": output_paths,
            "metadata": {
                "model": args.model,
                "requested_revision": args.revision,
                "resolved_revision": resolved_commit,
                "feature_extractor": args.extractor_kind,
                "normalization": "l2",
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
            },
        }
        _atomic_torch_save(payload, args.output)
        print(
            f"HF extraction passed: model={args.model}, count={features.shape[0]}, "
            f"dim={features.shape[1]}, revision={resolved_commit}",
            flush=True,
        )
        return 0
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"HF extraction failed: {_format_hf_error(error)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
