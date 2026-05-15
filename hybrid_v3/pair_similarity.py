\
\
\
\
\
\
\
\


from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import load_config
from model import MoCoV3HybridTopo


config: Dict[str, Any] = {}
BASE_DIR = Path(__file__).resolve().parent
MAIN_CODE_DIR = BASE_DIR.parent
GINSENG_EXTRACTOR_DIR = MAIN_CODE_DIR / "ginseng_extractor"
GINSENG_EXTRACTOR_SCRIPT = GINSENG_EXTRACTOR_DIR / "ginseng_extractor.py"
CONVERT_IMAGE_SCRIPT = GINSENG_EXTRACTOR_DIR / "convert_image.py"
PIL_TRANSPOSE = getattr(Image, "Transpose", Image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute similarity between two images with hybrid_v3."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config JSON path.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override key=value (repeatable).",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def load_runtime_config() -> Dict[str, Any]:
    args = parse_args()
    cfg = load_config("pair_similarity", config_path=args.config, kv_overrides=args.override)
    if "use_topology" not in cfg:
        cfg["use_topology"] = True
    return cfg


def get_target_size(preprocess_cfg: Dict[str, Any]) -> Tuple[int, int]:
    resize = preprocess_cfg.get("resize", 224)
    width = int(preprocess_cfg.get("resize_width", resize))
    height = int(preprocess_cfg.get("resize_height", resize))
    return max(1, width), max(1, height)


def apply_rotation(img: Image.Image, rotation_deg: int) -> Image.Image:
    rotation = int(rotation_deg) % 360
    if rotation == 0:
        return img
    if rotation == 90:
        return img.transpose(PIL_TRANSPOSE.ROTATE_90)
    if rotation == 180:
        return img.transpose(PIL_TRANSPOSE.ROTATE_180)
    if rotation == 270:
        return img.transpose(PIL_TRANSPOSE.ROTATE_270)
    return img.rotate(rotation, expand=True)


def resize_with_mode(
    img: Image.Image, preprocess_cfg: Dict[str, Any]
) -> Tuple[Image.Image, Dict[str, Any]]:
    mode = str(preprocess_cfg.get("resize_mode", "stretch")).lower()
    target_width, target_height = get_target_size(preprocess_cfg)
    pad_value = int(preprocess_cfg.get("pad_value", 0))
    canvas_color = (pad_value, pad_value, pad_value)
    resample = Image.BILINEAR

    if mode == "none":
        return img, {"mode": mode, "canvas_size": list(img.size), "content_size": list(img.size)}

    if mode == "stretch":
        resized = img.resize((target_width, target_height), resample=resample)
        return resized, {
            "mode": mode,
            "canvas_size": [target_width, target_height],
            "content_size": [target_width, target_height],
        }

    if mode in {"contain", "letterbox", "pad"}:
        scale = min(target_width / img.width, target_height / img.height)
        new_width = max(1, int(round(img.width * scale)))
        new_height = max(1, int(round(img.height * scale)))
        resized = img.resize((new_width, new_height), resample=resample)
        canvas = Image.new("RGB", (target_width, target_height), canvas_color)
        offset_x = (target_width - new_width) // 2
        offset_y = (target_height - new_height) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas, {
            "mode": "contain",
            "canvas_size": [target_width, target_height],
            "content_size": [new_width, new_height],
            "offset": [offset_x, offset_y],
        }

    if mode == "long_edge":
        target_long_edge = max(target_width, target_height)
        scale = target_long_edge / max(img.width, img.height)
        new_width = max(1, int(round(img.width * scale)))
        new_height = max(1, int(round(img.height * scale)))
        resized = img.resize((new_width, new_height), resample=resample)
        return resized, {
            "mode": mode,
            "canvas_size": [new_width, new_height],
            "content_size": [new_width, new_height],
        }

    if mode == "short_edge":
        target_short_edge = min(target_width, target_height)
        scale = target_short_edge / min(img.width, img.height)
        new_width = max(1, int(round(img.width * scale)))
        new_height = max(1, int(round(img.height * scale)))
        resized = img.resize((new_width, new_height), resample=resample)
        return resized, {
            "mode": mode,
            "canvas_size": [new_width, new_height],
            "content_size": [new_width, new_height],
        }

    raise ValueError(
        "Unsupported resize_mode. Expected one of: none, stretch, contain, long_edge, short_edge."
    )


def tensor_from_image(img: Image.Image) -> torch.Tensor:
    preprocess_cfg = config.get("image_preprocess", {})
    mean = preprocess_cfg.get("mean", [0.5, 0.5, 0.5])
    std = preprocess_cfg.get("std", [0.5, 0.5, 0.5])
    tensor_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return tensor_transform(img).unsqueeze(0)


def build_model(device: torch.device) -> MoCoV3HybridTopo:
    model = MoCoV3HybridTopo(
        feature_dim=config["feature_dim"],
        topo_dim=config["topo_dim"],
        K=config["K"],
        m=config["m"],
        T=config["T"],
        topo_weight=config["topo_weight"],
        use_topology=config.get("use_topology", True),
        use_legacy_branch=config["use_legacy_branch"],
        use_skeleton_branch=config["use_skeleton_branch"],
        use_edge_branch=config["use_edge_branch"],
        use_frequency_branch=config["use_frequency_branch"],
        device=device,
    )

    checkpoint_path = resolve_path(config["model_path"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def extract_feature(
    model: MoCoV3HybridTopo, img_path: str, device: torch.device, rotation_deg: int
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    img = Image.open(img_path).convert("L").convert("RGB")
    original_size = img.size
    rotated_img = apply_rotation(img, rotation_deg)
    rotated_size = rotated_img.size
    resized_img, resize_meta = resize_with_mode(rotated_img, config.get("image_preprocess", {}))
    model_input_size = resized_img.size
    img_tensor = tensor_from_image(resized_img).to(device)

    with torch.no_grad():
        feature_type = str(config.get("feature_type", "both")).lower()

        if feature_type == "visual":
            feat = model.extract_features(
                img_tensor,
                use_query_encoder=True,
                feature_type="visual",
            )
            return feat.squeeze(0), {
                "original_size": {"width": int(original_size[0]), "height": int(original_size[1])},
                "rotation_deg": int(rotation_deg),
                "rotated_size": {"width": int(rotated_size[0]), "height": int(rotated_size[1])},
                "model_input_size": {"width": int(model_input_size[0]), "height": int(model_input_size[1])},
                "resize_meta": resize_meta,
            }

        if feature_type == "topo":
            if not model.use_topology:
                raise ValueError("Topology branch is disabled, cannot extract topo-only features.")
            feat = model.extract_features(
                img_tensor,
                use_query_encoder=True,
                feature_type="topo",
            )
            return feat.squeeze(0), {
                "original_size": {"width": int(original_size[0]), "height": int(original_size[1])},
                "rotation_deg": int(rotation_deg),
                "rotated_size": {"width": int(rotated_size[0]), "height": int(rotated_size[1])},
                "model_input_size": {"width": int(model_input_size[0]), "height": int(model_input_size[1])},
                "resize_meta": resize_meta,
            }

        visual_feat = model.extract_features(
            img_tensor,
            use_query_encoder=True,
            feature_type="visual",
        )
        if not model.use_topology:
            return visual_feat.squeeze(0), {
                "original_size": {"width": int(original_size[0]), "height": int(original_size[1])},
                "rotation_deg": int(rotation_deg),
                "rotated_size": {"width": int(rotated_size[0]), "height": int(rotated_size[1])},
                "model_input_size": {"width": int(model_input_size[0]), "height": int(model_input_size[1])},
                "resize_meta": resize_meta,
            }

        topo_feat = model.extract_features(
            img_tensor,
            use_query_encoder=True,
            feature_type="topo",
        )

        alpha = float(config.get("fusion_alpha", 0.15))
        alpha = max(0.0, min(1.0, alpha))
        visual_feat = F.normalize(visual_feat, dim=1)
        topo_feat = F.normalize(topo_feat, dim=1)
        fused_feat = torch.cat([(1.0 - alpha) * visual_feat, alpha * topo_feat], dim=1)
        fused_feat = F.normalize(fused_feat, dim=1)
        return fused_feat.squeeze(0), {
            "original_size": {"width": int(original_size[0]), "height": int(original_size[1])},
            "rotation_deg": int(rotation_deg),
            "rotated_size": {"width": int(rotated_size[0]), "height": int(rotated_size[1])},
            "model_input_size": {"width": int(model_input_size[0]), "height": int(model_input_size[1])},
            "resize_meta": resize_meta,
        }


def validate_inputs() -> None:
    required_keys = ["model_path", "image_a", "image_b"]
    missing = [key for key in required_keys if not str(config.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Missing required config value(s): {', '.join(missing)}")

    for key in ["model_path", "image_a", "image_b"]:
        resolved = resolve_path(str(config[key]))
        if not resolved.exists():
            raise FileNotFoundError(f"{key} not found: {resolved}")


def run_subprocess(cmd: list[str], cwd: Path, step_name: str) -> None:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode == 0:
        return

    stdout_tail = "\n".join(result.stdout.strip().splitlines()[-40:])
    stderr_tail = "\n".join(result.stderr.strip().splitlines()[-40:])
    raise RuntimeError(
        f"{step_name} failed.\n"
        f"Command: {' '.join(str(x) for x in cmd)}\n"
        f"STDOUT:\n{stdout_tail}\n"
        f"STDERR:\n{stderr_tail}"
    )


def build_pipeline_root() -> Path:
    pipeline_root_raw = str(config.get("pipeline_root", "./pair_similarity_runs")).strip()
    pipeline_root = resolve_path(pipeline_root_raw)
    run_name = str(config.get("run_name", "")).strip()
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = pipeline_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def save_input_as_jpg(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        img.convert("RGB").save(dst_path, format="JPEG", quality=95)


def run_extract_binary_pipeline(image_a_path: Path, image_b_path: Path) -> Dict[str, str]:
    if not GINSENG_EXTRACTOR_SCRIPT.exists():
        raise FileNotFoundError(f"ginseng_extractor.py not found: {GINSENG_EXTRACTOR_SCRIPT}")
    if not CONVERT_IMAGE_SCRIPT.exists():
        raise FileNotFoundError(f"convert_image.py not found: {CONVERT_IMAGE_SCRIPT}")

    run_root = build_pipeline_root()
    input_dir = run_root / "input"
    cutout_dir = run_root / "cutout"
    processed_dir = run_root / "processed"
    gray_dir = run_root / "gray"
    binary_dir = run_root / "binary"
    grounded_dir = run_root / "groundsam"
    json_dir = run_root / "json"
    mask_dir = run_root / "mask"
    raw_dir = run_root / "raw"
    temp_dir = run_root / "temp"
    transparent_dir = run_root / "transparent"

    staged_a = input_dir / "image_a.jpg"
    staged_b = input_dir / "image_b.jpg"
    save_input_as_jpg(image_a_path, staged_a)
    save_input_as_jpg(image_b_path, staged_b)

    text_prompt = str(config.get("text_prompt", "Yellow tree root")).strip() or "Yellow tree root"
    extractor_cmd = [
        sys.executable,
        str(GINSENG_EXTRACTOR_SCRIPT),
        "--input-root",
        str(input_dir),
        "--output-root",
        str(cutout_dir),
        "--processed-root",
        str(processed_dir),
        "--json-root",
        str(json_dir),
        "--groundsam-root",
        str(grounded_dir),
        "--transparent-root",
        str(transparent_dir),
        "--raw-root",
        str(raw_dir),
        "--mask-root",
        str(mask_dir),
        "--temp-root",
        str(temp_dir),
        "--batch-size",
        "2",
        "--text-prompt",
        text_prompt,
        "--no-transparent",
    ]
    run_subprocess(extractor_cmd, GINSENG_EXTRACTOR_DIR, "ginseng extraction")

    binary_threshold = int(config.get("binary_threshold", 128))
    convert_cmd = [
        sys.executable,
        str(CONVERT_IMAGE_SCRIPT),
        "--input-root",
        str(processed_dir),
        "--output-gray",
        str(gray_dir),
        "--output-binary",
        str(binary_dir),
        "--threshold",
        str(binary_threshold),
    ]
    run_subprocess(convert_cmd, GINSENG_EXTRACTOR_DIR, "binary conversion")

    processed_a = processed_dir / "image_a.jpg"
    processed_b = processed_dir / "image_b.jpg"
    binary_a = binary_dir / "image_a.jpg"
    binary_b = binary_dir / "image_b.jpg"
    for path in [processed_a, processed_b, binary_a, binary_b]:
        if not path.exists():
            raise FileNotFoundError(f"Pipeline output not found: {path}")

    return {
        "run_root": str(run_root),
        "staged_image_a": str(staged_a),
        "staged_image_b": str(staged_b),
        "processed_image_a": str(processed_a),
        "processed_image_b": str(processed_b),
        "binary_image_a": str(binary_a),
        "binary_image_b": str(binary_b),
    }


def maybe_save_json(payload: Dict[str, Any]) -> None:
    output_json = str(config.get("output_json", "")).strip()
    if not output_json:
        return

    output_path = resolve_path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"结果已保存到: {output_path}")


def main() -> None:
    global config
    config = load_runtime_config()
    validate_inputs()

    use_gpu = bool(config.get("use_gpu", True)) and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    resize_mode = str(config.get("image_preprocess", {}).get("resize_mode", "stretch")).lower()
    target_width, target_height = get_target_size(config.get("image_preprocess", {}))
    preprocess_pipeline = str(config.get("preprocess_pipeline", "none")).lower()

    image_a_path = resolve_path(config["image_a"])
    image_b_path = resolve_path(config["image_b"])
    pipeline_outputs: Dict[str, str] = {}

    if preprocess_pipeline == "extract_binary":
        pipeline_outputs = run_extract_binary_pipeline(image_a_path, image_b_path)
        feature_image_a = pipeline_outputs["binary_image_a"]
        feature_image_b = pipeline_outputs["binary_image_b"]
    elif preprocess_pipeline == "none":
        feature_image_a = str(image_a_path)
        feature_image_b = str(image_b_path)
    else:
        raise ValueError("preprocess_pipeline only supports 'none' or 'extract_binary'.")

    print(f"\n{'=' * 70}")
    print("hybrid_v3 两图相似度计算")
    print(f"{'=' * 70}")
    print(f"设备: {device}")
    print(f"模型: {resolve_path(config['model_path'])}")
    print(f"原图A: {image_a_path}")
    print(f"原图B: {image_b_path}")
    print(f"送模图A: {feature_image_a}")
    print(f"送模图B: {feature_image_b}")
    print(f"preprocess_pipeline: {preprocess_pipeline}")
    print(f"feature_type: {config.get('feature_type', 'both')}")
    print(f"fusion_alpha: {config.get('fusion_alpha', 0.15)}")
    print(f"resize_mode: {resize_mode}")
    print(f"target_size: {target_width} x {target_height}")
    print(f"rotation_a: {int(config.get('rotation_a', 0))}")
    print(f"rotation_b: {int(config.get('rotation_b', 0))}")
    if pipeline_outputs:
        print(f"pipeline_run_root: {pipeline_outputs['run_root']}")

    model = build_model(device)

    feat_a, meta_a = extract_feature(
        model,
        feature_image_a,
        device,
        int(config.get("rotation_a", 0)),
    )
    feat_b, meta_b = extract_feature(
        model,
        feature_image_b,
        device,
        int(config.get("rotation_b", 0)),
    )

    feat_a = F.normalize(feat_a.unsqueeze(0), dim=1)
    feat_b = F.normalize(feat_b.unsqueeze(0), dim=1)
    cosine_similarity = float(torch.sum(feat_a * feat_b).item())
    score_0_1 = max(0.0, min(1.0, (cosine_similarity + 1.0) / 2.0))

    result = {
        "model_path": str(resolve_path(config["model_path"])),
        "raw_image_a": str(image_a_path),
        "raw_image_b": str(image_b_path),
        "feature_image_a": str(feature_image_a),
        "feature_image_b": str(feature_image_b),
        "preprocess_pipeline": preprocess_pipeline,
        "pipeline_outputs": pipeline_outputs,
        "image_preprocess": {
            "resize_mode": resize_mode,
            "resize_width": target_width,
            "resize_height": target_height,
            "pad_value": int(config.get("image_preprocess", {}).get("pad_value", 0)),
        },
        "feature_type": str(config.get("feature_type", "both")).lower(),
        "fusion_alpha": float(config.get("fusion_alpha", 0.15)),
        "image_a_meta": meta_a,
        "image_b_meta": meta_b,
        "feature_dim": int(feat_a.shape[1]),
        "cosine_similarity": cosine_similarity,
        "score_0_1": score_0_1,
    }

    print("\n图像A信息:")
    print(json.dumps(meta_a, indent=2, ensure_ascii=False))
    print("\n图像B信息:")
    print(json.dumps(meta_b, indent=2, ensure_ascii=False))
    print("\n相似度结果:")
    print(f"  cosine_similarity: {cosine_similarity:.6f}")
    print(f"  score_0_1: {score_0_1:.6f}")
    print(f"  feature_dim: {feat_a.shape[1]}")
    print(f"\nJSON结果:\n{json.dumps(result, indent=2, ensure_ascii=False)}")

    maybe_save_json(result)


if __name__ == "__main__":
    main()
