import argparse
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import load_config
from model import MoCoV3Ginseng


IMAGE_SUFFIXES_DEFAULT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

config = {}
image_preprocess = None


def parse_args():
    parser = argparse.ArgumentParser(description="Extract gallery features (MoCo-CBAM)")
    parser.add_argument("--config", type=str, default=None, help="Optional config JSON path")
    parser.add_argument("--override", action="append", default=[], help="Override key=value (repeatable)")
    return parser.parse_args()


def load_runtime_config():
    args = parse_args()
    cfg = load_config("extraction", config_path=args.config, kv_overrides=args.override)
    cfg["img_suffix"] = tuple(cfg.get("img_suffix", IMAGE_SUFFIXES_DEFAULT))
    return cfg


def build_preprocess(cfg):
    preprocess_cfg = cfg.get("image_preprocess", {})
    return transforms.Compose(
        [
            transforms.Resize(
                (int(preprocess_cfg.get("resize", 224)), int(preprocess_cfg.get("resize", 224)))
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=preprocess_cfg.get("mean", [0.5, 0.5, 0.5]),
                std=preprocess_cfg.get("std", [0.5, 0.5, 0.5]),
            ),
        ]
    )


def get_all_img_paths(root_dir, suffix):
    file_list = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(suffix):
                file_list.append(os.path.join(dirpath, name))
    return file_list


def get_features(model, img_path, device):
    img = Image.open(img_path).convert("L").convert("RGB")
    img_tensor = image_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.extract_features(img_tensor, use_query_encoder=True)
        feat = F.normalize(feat, dim=1)
    return feat.cpu().squeeze()


def main():
    global config, image_preprocess
    config = load_runtime_config()
    image_preprocess = build_preprocess(config)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 70}")
    print(f"[{timestamp}] Start MoCo-CBAM feature extraction")
    print(f"{'=' * 70}")

    device = torch.device("cuda" if config["use_gpu"] else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = MoCoV3Ginseng(
        feature_dim=config["feature_dim"],
        K=config["K"],
        m=config["m"],
        T=config["T"],
        device=device,
    )

    try:
        model.load_state_dict(torch.load(config["model_path"], map_location=device))
        print(f"Model loaded: {config['model_path']}")
    except Exception as exc:
        print(f"Model load failed: {exc}")
        return

    model.to(device)
    model.eval()

    print(f"\nScanning: {config['image_dir']}")
    img_paths = get_all_img_paths(config["image_dir"], config["img_suffix"])
    print(f"Found {len(img_paths)} images")

    if not img_paths:
        print("No images found, exit")
        return

    features = []
    valid_paths = []

    for idx, img_path in enumerate(img_paths, start=1):
        try:
            feat = get_features(model, img_path, device)
            features.append(feat)
            valid_paths.append(img_path)
            if idx % 20 == 0 or idx == len(img_paths):
                progress_pct = idx / len(img_paths) * 100
                print(f"[{idx:>4d}/{len(img_paths)}] ({progress_pct:>5.1f}%) {os.path.basename(img_path)}")
        except Exception as exc:
            print(f"[{idx:>4d}] Skip: {os.path.basename(img_path)} -> {str(exc)[:80]}")

    print("\nSaving features...")
    features_tensor = torch.stack(features)
    os.makedirs(os.path.dirname(config["output_feats"]), exist_ok=True)
    torch.save(
        {
            "features": features_tensor,
            "paths": valid_paths,
            "feature_dim": config["feature_dim"],
            "num_images": len(valid_paths),
            "model_config": {"feature_dim": config["feature_dim"]},
        },
        config["output_feats"],
    )

    print(f"Saved: {config['output_feats']}")
    print(f"Images: {len(valid_paths)}")
    print(f"Feature shape: {features_tensor.shape}")


if __name__ == "__main__":
    main()
