from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "test_pair_simalarity_extractor"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "resolution_impact_runs"
BACKEND_DIR = Path(r"d:\post_code\ginseng_retrieve_system\backend")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze resolution impact using ginseng_retrieve_system backend retrieval flow."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing extracted/binarized images for similarity analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory. Defaults to resolution_impact_runs/<timestamp>_backend_flow.",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=1920,
        help="Resize anchor image to this height while preserving aspect ratio.",
    )
    return parser.parse_args()


def load_backend_settings():
    original_cwd = Path.cwd()
    os.chdir(BACKEND_DIR)
    try:
        if str(BACKEND_DIR) not in sys.path:
            sys.path.insert(0, str(BACKEND_DIR))
        from app.core.settings import settings
        from app.utils.hybrid_imports import ensure_hybrid_path

        ensure_hybrid_path()
        from model import MoCoV3HybridTopo

        return settings, MoCoV3HybridTopo
    finally:
        os.chdir(original_cwd)


def build_output_dir(path_str: str) -> Path:
    if path_str.strip():
        output_dir = Path(path_str)
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_backend_flow"
        output_dir = DEFAULT_OUTPUT_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def group_images(image_dir: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        group_key = path.stem.split("-", 1)[0]
        groups[group_key].append(path)
    return dict(sorted(groups.items(), key=lambda item: int(item[0])))


def build_model(settings, model_cls):
    device = torch.device("cuda" if settings.use_gpu and torch.cuda.is_available() else "cpu")
    model = model_cls(
        feature_dim=settings.feature_dim,
        topo_dim=settings.topo_dim,
        K=settings.K,
        m=settings.m,
        T=settings.T,
        topo_weight=settings.topo_weight,
        use_legacy_branch=settings.use_legacy_branch,
        use_skeleton_branch=settings.use_skeleton_branch,
        use_edge_branch=settings.use_edge_branch,
        use_frequency_branch=settings.use_frequency_branch,
        device=device,
    )
    checkpoint = torch.load(settings.hybrid_model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model, device


def build_preprocess():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def resize_anchor_image(src_path: Path, dst_path: Path, target_height: int) -> Dict[str, int]:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        gray = img.convert("L")
        width, height = gray.size
        scale = float(target_height) / float(height)
        new_width = max(1, int(round(width * scale)))
        resized = gray.resize((new_width, target_height), Image.LANCZOS)
        binary = resized.point(lambda x: 0 if x < 128 else 255, "1")
        binary.save(dst_path)
    return {"width": new_width, "height": target_height}


def prepare_resized_anchors(
    groups: Dict[str, List[Path]], output_dir: Path, target_height: int
) -> Tuple[Dict[Path, Path], Dict[str, Dict[str, int]]]:
    resized_root = output_dir / "anchor_resized_h1920"
    resized_map: Dict[Path, Path] = {}
    size_map: Dict[str, Dict[str, int]] = {}
    for paths in groups.values():
        for src_path in paths:
            dst_path = resized_root / f"{src_path.stem}_h{target_height}.png"
            size_map[str(src_path)] = resize_anchor_image(src_path, dst_path, target_height)
            resized_map[src_path] = dst_path
    return resized_map, size_map


def extract_fused_feature(model, device: torch.device, preprocess, img_path: Path) -> Dict[str, Any]:
    img = Image.open(img_path).convert("L").convert("RGB")
    original_size = img.size
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
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
        fused_feat = F.normalize(fused_feat, dim=1)
    return {
        "feature": fused_feat.squeeze(0).cpu(),
        "meta": {
            "original_width": int(original_size[0]),
            "original_height": int(original_size[1]),
            "model_input_width": 224,
            "model_input_height": 224,
        },
    }


def cosine_similarity(feat_a: torch.Tensor, feat_b: torch.Tensor) -> float:
    return float(torch.sum(feat_a * feat_b).item())


def summarize_results(df: pd.DataFrame) -> Dict[str, Any]:
    group_summary_df = (
        df.groupby("group_id")
        .agg(
            pair_count=("pair_id", "count"),
            mean_similarity_original=("similarity_original", "mean"),
            mean_similarity_h1920=("similarity_h1920", "mean"),
            mean_delta_similarity=("delta_similarity", "mean"),
            median_delta_similarity=("delta_similarity", "median"),
            positive_delta_count=("delta_similarity", lambda s: int((s > 0).sum())),
            negative_delta_count=("delta_similarity", lambda s: int((s < 0).sum())),
        )
        .reset_index()
    )
    group_summary: List[Dict[str, Any]] = []
    for row in group_summary_df.to_dict(orient="records"):
        group_summary.append(
            {
                "group_id": str(row["group_id"]),
                "pair_count": int(row["pair_count"]),
                "mean_similarity_original": float(row["mean_similarity_original"]),
                "mean_similarity_h1920": float(row["mean_similarity_h1920"]),
                "mean_delta_similarity": float(row["mean_delta_similarity"]),
                "median_delta_similarity": float(row["median_delta_similarity"]),
                "positive_delta_count": int(row["positive_delta_count"]),
                "negative_delta_count": int(row["negative_delta_count"]),
            }
        )

    return {
        "pair_count": int(len(df)),
        "group_count": int(df["group_id"].nunique()),
        "image_count": int(pd.unique(df[["anchor_image", "other_image"]].values.ravel("K")).size),
        "mean_similarity_original": float(df["similarity_original"].mean()),
        "mean_similarity_h1920": float(df["similarity_h1920"].mean()),
        "mean_delta_similarity": float(df["delta_similarity"].mean()),
        "median_delta_similarity": float(df["delta_similarity"].median()),
        "std_delta_similarity": float(df["delta_similarity"].std(ddof=0)),
        "positive_delta_ratio": float((df["delta_similarity"] > 0).mean()),
        "negative_delta_ratio": float((df["delta_similarity"] < 0).mean()),
        "group_summary": group_summary,
    }


def make_summary_figure(df: pd.DataFrame, summary: Dict[str, Any], output_path: Path) -> None:
    group_means = (
        df.groupby("group_id")["delta_similarity"]
        .mean()
        .reset_index()
        .sort_values("group_id", key=lambda s: s.astype(int))
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)

    axes[0].boxplot(
        [df["similarity_original"], df["similarity_h1920"]],
        tick_labels=["3k*4k", "h1920 Anchor"],
        patch_artist=True,
        boxprops={"facecolor": "#cfe3d6"},
        medianprops={"color": "#8b1e3f", "linewidth": 1.5},
    )
    axes[0].set_title("Similarity Distribution")
    axes[0].set_ylabel("Similarity")
    axes[0].grid(axis="y", alpha=0.25)

    colors = ["#2a6f97" if x >= 0 else "#c44536" for x in group_means["delta_similarity"]]
    axes[1].bar(group_means["group_id"].astype(str), group_means["delta_similarity"], color=colors)
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_title("Mean Delta per Group (h1920 - 3k*4k)")
    axes[1].set_xlabel("Group")
    axes[1].set_ylabel("Mean Delta Similarity")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(
        (
            "Retrieval Backend Flow Resolution Impact\n"
            f"Pairs={summary['pair_count']}  "
            f"Mean 3k*4k={summary['mean_similarity_original']:.4f}  "
            f"Mean h1920={summary['mean_similarity_h1920']:.4f}  "
            f"Mean Delta={summary['mean_delta_similarity']:.4f}"
        ),
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_summary_markdown(
    summary: Dict[str, Any],
    output_path: Path,
    input_dir: Path,
    model_path: Path,
    target_height: int,
) -> None:
    group_lines = []
    for row in summary["group_summary"]:
        direction = "升高" if row["mean_delta_similarity"] > 0 else "降低" if row["mean_delta_similarity"] < 0 else "基本不变"
        group_lines.append(
            f"- 第 {row['group_id']} 组：{row['pair_count']} 条有向配对，3k*4k 均值 {row['mean_similarity_original']:.4f}，h{target_height} 均值 {row['mean_similarity_h1920']:.4f}，平均变化 {row['mean_delta_similarity']:+.4f}（{direction}）"
        )

    mean_delta = summary["mean_delta_similarity"]
    if mean_delta > 0.01:
        overall_text = f"按检索系统后端流程计算后，把锚点图缩到高 {target_height}，整体相似度明显升高。"
    elif mean_delta < -0.01:
        overall_text = f"按检索系统后端流程计算后，把锚点图缩到高 {target_height}，整体相似度明显降低。"
    else:
        overall_text = f"按检索系统后端流程计算后，把锚点图缩到高 {target_height}，整体相似度变化较小。"

    content = "\n".join(
        [
            "# 检索系统后端流程分辨率影响实验",
            "",
            f"- 输入目录：`{input_dir}`",
            f"- 模型：`{model_path}`",
            f"- 锚点缩放目标高度：`{target_height}`",
            "- 计算流程：`hybrid_v2 + Resize(224,224) + visual/topo拼接 + L2归一化 + 点积相似度`",
            "",
            "## 总体结论",
            "",
            overall_text,
            "",
            f"- 3k*4k 平均相似度：`{summary['mean_similarity_original']:.6f}`",
            f"- h{target_height} 平均相似度：`{summary['mean_similarity_h1920']:.6f}`",
            f"- 平均变化量：`{summary['mean_delta_similarity']:+.6f}`",
            f"- 中位变化量：`{summary['median_delta_similarity']:+.6f}`",
            f"- 正向变化占比：`{summary['positive_delta_ratio']:.2%}`",
            f"- 负向变化占比：`{summary['negative_delta_ratio']:.2%}`",
            "",
            "## 分组结果",
            "",
            *group_lines,
            "",
            "## 说明",
            "",
            "- 本实验直接使用已提取、已二值化的人参图像，跳过提取步骤。",
            "- 但特征提取与相似度计算严格对齐 `ginseng_retrieve_system/backend` 的检索流程。",
            "- 每条记录都是有向配对：只缩小锚点图，其余同组图始终保持原尺寸。",
        ]
    )
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = build_output_dir(args.output_dir)
    settings, model_cls = load_backend_settings()
    model, device = build_model(settings, model_cls)
    preprocess = build_preprocess()
    groups = group_images(input_dir)
    resized_map, resized_sizes = prepare_resized_anchors(groups, output_dir, args.target_height)

    print("=" * 70)
    print("按 ginseng_retrieve_system 后端流程计算分辨率影响")
    print("=" * 70)
    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"model_path: {settings.hybrid_model_path}")
    print(f"device: {device}")
    print(
        "branches:",
        {
            "legacy": settings.use_legacy_branch,
            "skeleton": settings.use_skeleton_branch,
            "edge": settings.use_edge_branch,
            "frequency": settings.use_frequency_branch,
        },
    )
    print(f"group_count: {len(groups)}")
    print(f"image_count: {sum(len(v) for v in groups.values())}")
    print(f"target_height: {args.target_height}")

    original_cache: Dict[Path, Dict[str, Any]] = {}
    resized_cache: Dict[Path, Dict[str, Any]] = {}
    total_images = sum(len(v) for v in groups.values())
    current = 0
    for group_id, paths in groups.items():
        for image_path in paths:
            current += 1
            print(f"[{current}/{total_images}] original feature: {group_id}/{image_path.name}")
            original_cache[image_path] = extract_fused_feature(model, device, preprocess, image_path)
            print(f"[{current}/{total_images}] resized feature: {group_id}/{image_path.name}")
            resized_cache[image_path] = extract_fused_feature(model, device, preprocess, resized_map[image_path])

    records: List[Dict[str, Any]] = []
    for group_id, paths in groups.items():
        for anchor_path in paths:
            for other_path in paths:
                if anchor_path == other_path:
                    continue
                anchor_original = original_cache[anchor_path]
                anchor_resized = resized_cache[anchor_path]
                other_original = original_cache[other_path]

                sim_original = cosine_similarity(
                    anchor_original["feature"], other_original["feature"]
                )
                sim_resized = cosine_similarity(
                    anchor_resized["feature"], other_original["feature"]
                )

                records.append(
                    {
                        "pair_id": f"{anchor_path.stem}__to__{other_path.stem}",
                        "group_id": group_id,
                        "anchor_image": anchor_path.name,
                        "other_image": other_path.name,
                        "anchor_original_width": anchor_original["meta"]["original_width"],
                        "anchor_original_height": anchor_original["meta"]["original_height"],
                        "anchor_h1920_width": resized_sizes[str(anchor_path)]["width"],
                        "anchor_h1920_height": resized_sizes[str(anchor_path)]["height"],
                        "other_original_width": other_original["meta"]["original_width"],
                        "other_original_height": other_original["meta"]["original_height"],
                        "similarity_original": sim_original,
                        "similarity_h1920": sim_resized,
                        "delta_similarity": sim_resized - sim_original,
                    }
                )

    df = pd.DataFrame(records)
    df["_group_order"] = df["group_id"].astype(int)
    df = df.sort_values(by=["_group_order", "anchor_image", "other_image"]).drop(
        columns=["_group_order"]
    )

    csv_path = output_dir / "resolution_impact_backend_flow_results.csv"
    summary_json_path = output_dir / "resolution_impact_backend_flow_summary.json"
    summary_md_path = output_dir / "resolution_impact_backend_flow_summary.md"
    figure_path = output_dir / "resolution_impact_backend_flow_summary.png"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary = summarize_results(df)
    summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_summary_markdown(
        summary=summary,
        output_path=summary_md_path,
        input_dir=input_dir,
        model_path=Path(settings.hybrid_model_path),
        target_height=args.target_height,
    )
    make_summary_figure(df, summary, figure_path)

    print("\n输出文件:")
    print(f"- CSV: {csv_path}")
    print(f"- Figure: {figure_path}")
    print(f"- Summary JSON: {summary_json_path}")
    print(f"- Summary Markdown: {summary_md_path}")
    print("\n总体统计:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
