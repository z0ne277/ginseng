import ast
import json
import re
from typing import Any, Dict, List

from PIL import Image
from torchvision import transforms


def load_grayscale_rgb(img_path: str) -> Image.Image:
    with Image.open(img_path) as img:
        return img.convert("L").convert("RGB")


def build_tensor_transform(cfg: Dict[str, Any]):
    preprocess_cfg = cfg.get("image_preprocess", {})
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=preprocess_cfg.get("mean", [0.5, 0.5, 0.5]),
                std=preprocess_cfg.get("std", [0.5, 0.5, 0.5]),
            ),
        ]
    )


def _parse_tta_mode(mode_text: str, default_size: int) -> Dict[str, Any]:
    token = str(mode_text).strip().lower()
    match = re.fullmatch(r"(stretch|contain)(\d+)?", token)
    if not match:
        raise ValueError(
            f"Unsupported TTA mode '{mode_text}'. Expected values like stretch224 or contain256."
        )

    mode = match.group(1)
    size = int(match.group(2) or default_size)
    if size <= 0:
        raise ValueError(f"Invalid TTA image size: {size}")

    return {"name": f"{mode}{size}", "mode": mode, "size": size}


def _normalize_raw_tta_modes(raw_modes: Any, default_size: int) -> List[str]:
    if raw_modes is None:
        return [f"stretch{default_size}"]

    if isinstance(raw_modes, (list, tuple)):
        return [str(item).strip() for item in raw_modes if str(item).strip()]

    if not isinstance(raw_modes, str):
        return [str(raw_modes).strip()]

    text = raw_modes.strip()

    # Repeatedly peel off wrapping quotes so cmd / powershell / json-string inputs all work.
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", "\""}:
        text = text[1:-1].strip()

    if not text:
        return [f"stretch{default_size}"]

    if text.startswith("[") and text.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip().strip("'\"") for item in parsed if str(item).strip()]

    if "," in text:
        return [part.strip().strip("'\"") for part in text.split(",") if part.strip()]

    return [text.strip().strip("'\"")]


def _normalize_raw_tta_weights(raw_weights: Any) -> List[float]:
    if raw_weights is None:
        return []

    if isinstance(raw_weights, (list, tuple)):
        items = list(raw_weights)
    elif isinstance(raw_weights, str):
        text = raw_weights.strip()
        while len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", "\""}:
            text = text[1:-1].strip()

        if not text:
            return []

        if text.startswith("[") and text.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                except Exception:
                    continue
                if isinstance(parsed, (list, tuple)):
                    items = list(parsed)
                    break
            else:
                items = [part.strip() for part in text[1:-1].split(",") if part.strip()]
        elif "," in text:
            items = [part.strip() for part in text.split(",") if part.strip()]
        else:
            items = [text]
    else:
        items = [raw_weights]

    normalized = []
    for item in items:
        token = str(item).strip().strip("'\"")
        if not token:
            continue
        normalized.append(float(token))
    return normalized


def build_tta_specs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    preprocess_cfg = cfg.get("image_preprocess", {})
    default_size = int(preprocess_cfg.get("resize", 224))

    if bool(cfg.get("tta_enabled", False)):
        raw_modes = cfg.get("tta_modes", [f"stretch{default_size}"])
    else:
        raw_modes = [f"stretch{default_size}"]

    normalized_modes = _normalize_raw_tta_modes(raw_modes, default_size)
    specs = [_parse_tta_mode(mode_text, default_size) for mode_text in normalized_modes]
    if not specs:
        raise ValueError("No valid TTA modes configured")
    return specs


def build_tta_weights(cfg: Dict[str, Any], specs: List[Dict[str, Any]]) -> List[float]:
    if not specs:
        raise ValueError("No TTA specs available for weight assignment")

    weights = _normalize_raw_tta_weights(cfg.get("tta_weights"))
    if not weights:
        weights = [1.0] * len(specs)

    if len(weights) != len(specs):
        raise ValueError(
            f"TTA weights count {len(weights)} does not match TTA modes count {len(specs)}"
        )
    if any(weight < 0 for weight in weights):
        raise ValueError("TTA weights must be non-negative")

    total = sum(weights)
    if total <= 0:
        raise ValueError("Sum of TTA weights must be positive")

    return [float(weight) / float(total) for weight in weights]


def resize_with_mode(img: Image.Image, spec: Dict[str, Any], pad_value: int = 0) -> Image.Image:
    target_size = int(spec["size"])
    mode = str(spec["mode"]).lower()
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR

    if mode == "stretch":
        return img.resize((target_size, target_size), resample=resample)

    if mode == "contain":
        scale = min(target_size / img.width, target_size / img.height)
        new_width = max(1, int(round(img.width * scale)))
        new_height = max(1, int(round(img.height * scale)))
        resized = img.resize((new_width, new_height), resample=resample)

        fill = int(pad_value)
        canvas = Image.new("RGB", (target_size, target_size), (fill, fill, fill))
        offset_x = (target_size - new_width) // 2
        offset_y = (target_size - new_height) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas

    raise ValueError(f"Unsupported resize mode: {mode}")
