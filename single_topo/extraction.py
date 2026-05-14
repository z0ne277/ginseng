import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from config import load_config
from model import ImprovedMoCoV3WithTopoSideline
from preprocess_utils import (
    build_tensor_transform,
    build_tta_specs,
    build_tta_weights,
    load_grayscale_rgb,
    resize_with_mode,
)


IMAGE_SUFFIXES_DEFAULT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

config = {}
tensor_transform = None
tta_specs = []
tta_weights = []


def parse_args():
    parser = argparse.ArgumentParser(description="Extract gallery features for 2025-11-8 model")
    parser.add_argument("--config", type=str, default=None, help="Optional config JSON path")
    parser.add_argument("--override", action="append", default=[], help="Override key=value")
    return parser.parse_args()


def load_runtime_config():
    args = parse_args()
    cfg = load_config("extraction", config_path=args.config, kv_overrides=args.override)
    cfg["img_suffix"] = tuple(cfg.get("img_suffix", IMAGE_SUFFIXES_DEFAULT))
    return cfg


def get_all_img_paths(root_dir, suffixes):
    root = Path(root_dir)
    return sorted(
        str(path)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def extract_single_embedding(model, img_tensor):
    feature_type = str(config.get("feature_type", "both")).lower()
    if feature_type == "visual":
        feat = model.extract_features(
            img_tensor,
            use_query_encoder=True,
            feature_type="visual",
        )
        return F.normalize(feat, dim=1)

    if feature_type == "topo":
        feat = model.extract_features(
            img_tensor,
            use_query_encoder=True,
            feature_type="topo",
        )
        return F.normalize(feat, dim=1)

    visual_feat = model.extract_features(
        img_tensor,
        use_query_encoder=True,
        feature_type="visual",
    )
    topo_feat = model.extract_features(
        img_tensor,
        use_query_encoder=True,
        feature_type="topo",
    )
    fused_feat = torch.cat([visual_feat, topo_feat], dim=1)
    return F.normalize(fused_feat, dim=1)


def get_fused_features(model, img_path, device):
    img = load_grayscale_rgb(img_path)
    pad_value = int(config.get("tta_pad_value", 0))
    view_features = []

    with torch.no_grad():
        for spec in tta_specs:
            processed_img = resize_with_mode(img, spec, pad_value=pad_value)
            img_tensor = tensor_transform(processed_img).unsqueeze(0).to(device)
            view_features.append(extract_single_embedding(model, img_tensor).squeeze(0))

    weight_tensor = torch.tensor(
        tta_weights,
        dtype=view_features[0].dtype,
        device=view_features[0].device,
    ).view(-1, 1)
    fused_feat = (torch.stack(view_features, dim=0) * weight_tensor).sum(dim=0, keepdim=True)
    fused_feat = F.normalize(fused_feat, dim=1)
    return fused_feat.squeeze(0).cpu()


def load_model(device):
    model = ImprovedMoCoV3WithTopoSideline(
        feature_dim=config["feature_dim"],
        topo_dim=config["topo_dim"],
        K=config["K"],
        m=config["m"],
        T=config["T"],
        topo_weight=config["topo_weight"],
        num_erosion_levels=config["num_erosion_levels"],
        device=device,
    )

    checkpoint = torch.load(config["model_path"], map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def main():
    global config, tensor_transform, tta_specs, tta_weights
    config = load_runtime_config()
    tensor_transform = build_tensor_transform(config)
    tta_specs = build_tta_specs(config)
    tta_weights = build_tta_weights(config, tta_specs)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Start extracting gallery features for single_topo")
    print(f"TTA modes: {[spec['name'] for spec in tta_specs]}")
    print(f"TTA weights: {[round(weight, 4) for weight in tta_weights]}")

    use_gpu = bool(config.get("use_gpu", True)) and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"Device: {device}")

    model = load_model(device)
    print(f"Model loaded: {config['model_path']}")

    img_paths = get_all_img_paths(config["image_dir"], config["img_suffix"])
    print(f"Images found: {len(img_paths)}")
    if not img_paths:
        raise FileNotFoundError(f"No images found under {config['image_dir']}")

    features = []
    valid_paths = []

    for idx, img_path in enumerate(img_paths, start=1):
        try:
            feat = get_fused_features(model, img_path, device)
            features.append(feat)
            valid_paths.append(img_path)
        except Exception as exc:
            print(f"Skip {img_path}: {exc}")
            continue

        if idx % 100 == 0 or idx == len(img_paths):
            print(f"Progress: {idx}/{len(img_paths)}")

    if not features:
        raise RuntimeError("No valid features were extracted")

    features_tensor = torch.stack(features)
    output_path = Path(config["output_feats"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features_tensor,
            "paths": valid_paths,
            "feature_dim": config["feature_dim"],
            "topo_dim": config["topo_dim"],
            "total_dim": int(features_tensor.shape[1]),
            "num_images": len(valid_paths),
            "feature_type": str(config.get("feature_type", "both")).lower(),
            "num_erosion_levels": config["num_erosion_levels"],
            "tta_enabled": bool(config.get("tta_enabled", False)),
            "tta_modes": [spec["name"] for spec in tta_specs],
            "tta_weights": [float(weight) for weight in tta_weights],
        },
        output_path,
    )
    print(f"Saved features: {output_path}")
    print(f"Feature tensor shape: {tuple(features_tensor.shape)}")


if __name__ == "__main__":
    main()
