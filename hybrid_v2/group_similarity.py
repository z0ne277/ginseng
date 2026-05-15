\
\
\
\
\
\
\


import argparse
import csv
import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import MoCoV3HybridTopo

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def list_group_dirs(root_dir: str) -> List[str]:
    root = Path(root_dir)
    if not root.exists():
        return []
    return [str(p) for p in root.iterdir() if p.is_dir()]


def list_images(folder: str, suffixes=IMAGE_SUFFIXES, recursive: bool = False) -> List[str]:
    root = Path(folder)
    if recursive:
        return [str(p) for p in root.rglob('*') if p.is_file() and p.suffix.lower() in suffixes]
    return [str(p) for p in root.iterdir() if p.is_file() and p.suffix.lower() in suffixes]


def sample_images(paths: List[str], sample_k: int, seed: int, strategy: str = "random") -> List[str]:
    if sample_k <= 0 or len(paths) <= sample_k:
        return sorted(paths)
    if strategy == "first":
        return sorted(paths)[:sample_k]
    rng = random.Random(seed)
    return rng.sample(paths, sample_k)


def build_model(config, device):
    model = MoCoV3HybridTopo(
        feature_dim=config['feature_dim'],
        topo_dim=config['topo_dim'],
        K=config['K'],
        m=config['m'],
        T=config['T'],
        topo_weight=config['topo_weight'],
        use_legacy_branch=config['use_legacy_branch'],
        use_skeleton_branch=config['use_skeleton_branch'],
        use_edge_branch=config['use_edge_branch'],
        use_frequency_branch=config['use_frequency_branch'],
        device=device
    )

    checkpoint = torch.load(config['model_path'], map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def get_fused_features(model, img_path: str, device, preprocess) -> torch.Tensor:
    img = Image.open(img_path).convert("L").convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        visual_feat = model.extract_features(
            img_tensor,
            use_query_encoder=True,
            feature_type='visual'
        )
        topo_feat = model.extract_features(
            img_tensor,
            use_query_encoder=True,
            feature_type='topo'
        )
        fused_feat = torch.cat([visual_feat, topo_feat], dim=1)
        fused_feat = F.normalize(fused_feat, dim=1)

    return fused_feat.squeeze(0).cpu()


def mean_pairwise_cosine(features: torch.Tensor) -> float:
    if features.size(0) < 2:
        return 0.0
    feats = F.normalize(features, dim=1)
    sims = torch.mm(feats, feats.t())
    idx = torch.triu_indices(sims.size(0), sims.size(1), offset=1)
    vals = sims[idx[0], idx[1]]
    return float(vals.mean().item()) if vals.numel() > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Folder-level similarity with hybrid_v2 features.")
    parser.add_argument("--input-root", required=True, help="Root folder containing group subfolders.")
    parser.add_argument("--model-path", required=True, help="Path to trained hybrid_v2 weights.")
    parser.add_argument("--output-csv", default="group_similarity.csv", help="Output CSV path.")
    parser.add_argument("--sample", type=int, default=4, help="Images per folder (0=all).")
    parser.add_argument("--sample-strategy", choices=["random", "first"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recursive", action="store_true", help="Scan nested images in each group folder.")
    parser.add_argument("--suffixes", default=None, help="Comma-separated suffixes, e.g. .jpg,.png")
    args = parser.parse_args()

    suffixes = IMAGE_SUFFIXES
    if args.suffixes:
        suffixes = tuple(s.strip().lower() for s in args.suffixes.split(',') if s.strip())

    config = {
        'model_path': args.model_path,
        'feature_dim': 256,
        'topo_dim': 128,
        'K': 4096,
        'm': 0.999,
        'T': 0.07,
        'topo_weight': 0.35,
        'use_legacy_branch': True,
        'use_skeleton_branch': False,
        'use_edge_branch': True,
        'use_frequency_branch': False,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    model = build_model(config, device)

    group_dirs = list_group_dirs(args.input_root)
    if not group_dirs:
        print(f"No subfolders found under: {args.input_root}")
        return

    rows: List[Tuple[str, int, int, str, str, str, str]] = []

    for group_dir in group_dirs:
        image_paths = list_images(group_dir, suffixes=suffixes, recursive=args.recursive)
        used = sample_images(image_paths, args.sample, args.seed, args.sample_strategy)
        if len(image_paths) < 2 or len(used) < 2:
            rows.append((
                os.path.basename(group_dir),
                len(image_paths),
                len(used),
                "",
                "",
                "",
                "insufficient_images"
            ))
            continue
        feats = []
        for p in used:
            try:
                feats.append(get_fused_features(model, p, device, preprocess))
            except Exception as e:
                print(f"Skip {p}: {e}")
        if len(feats) < 2:
            rows.append((
                os.path.basename(group_dir),
                len(image_paths),
                len(used),
                "",
                "",
                "",
                "feature_extract_failed"
            ))
            continue
        feats_tensor = torch.stack(feats)
        mean_sim = mean_pairwise_cosine(feats_tensor)
        score = (mean_sim + 1.0) / 2.0
        score = max(0.0, min(1.0, score))
        diversity = 1.0 - score
        rows.append((
            os.path.basename(group_dir),
            len(image_paths),
            len(used),
            f"{mean_sim:.6f}",
            f"{score:.6f}",
            f"{diversity:.6f}",
            "ok"
        ))

    rows.sort(key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0]))

    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["group", "num_images", "used_images", "mean_cosine", "score_0_1", "diversity_0_1", "status"])
        writer.writerows(rows)

    print(f"Saved: {args.output_csv} (groups={len(rows)})")


if __name__ == "__main__":
    main()
