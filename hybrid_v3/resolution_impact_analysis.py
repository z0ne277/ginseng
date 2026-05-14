from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

import pair_similarity as ps
from config import load_config


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "test_pair_simalarity_extractor"
DEFAULT_RAW_DIR = BASE_DIR / "test_pair_simalarity"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "resolution_impact_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the impact of anchor image resolution on hybrid_v3 pair similarity."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "extract-single"],
        default="batch",
        help="batch: run the full experiment; extract-single: extract one feature in an isolated subprocess.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing extracted/binarized ginseng images.",
    )
    parser.add_argument(
        "--fallback-raw-dir",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help="Fallback directory containing raw ginseng images if input-dir is unavailable.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory. Defaults to resolution_impact_runs/<timestamp>.",
    )
    parser.add_argument(
        "--downscale-long-edge",
        type=int,
        default=1000,
        help="Downscale the anchor image so its long edge equals this value.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU if available.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Optional model checkpoint override.",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="",
        help="Optional feature_type override: visual/topo/both.",
    )
    parser.add_argument(
        "--fusion-alpha",
        type=float,
        default=-1.0,
        help="Optional fusion alpha override for feature_type=both.",
    )
    parser.add_argument(
        "--single-image",
        type=str,
        default="",
        help="Worker mode only: image path for single feature extraction.",
    )
    parser.add_argument(
        "--single-feature-out",
        type=str,
        default="",
        help="Worker mode only: output .pt path for extracted feature.",
    )
    parser.add_argument(
        "--single-meta-out",
        type=str,
        default="",
        help="Worker mode only: output .json path for feature metadata.",
    )
    return parser.parse_args()


def choose_input_dir(preferred: Path, fallback: Path) -> Path:
    if preferred.exists() and any(preferred.iterdir()):
        return preferred
    if fallback.exists() and any(fallback.iterdir()):
        return fallback
    raise FileNotFoundError(
        f"No usable input directory found. Checked: {preferred} and {fallback}"
    )


def build_output_dir(path_str: str) -> Path:
    if path_str.strip():
        output_dir = Path(path_str)
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_runtime_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config("pair_similarity")
    cfg["preprocess_pipeline"] = "none"
    cfg["use_gpu"] = bool(args.use_gpu)
    cfg.setdefault("image_preprocess", {})
    cfg["image_preprocess"]["resize_mode"] = "none"
    if args.model_path.strip():
        cfg["model_path"] = args.model_path.strip()
    if args.feature_type.strip():
        cfg["feature_type"] = args.feature_type.strip().lower()
    if args.fusion_alpha >= 0.0:
        cfg["fusion_alpha"] = float(args.fusion_alpha)
    return cfg


def get_device(cfg: Dict[str, Any]) -> torch.device:
    use_gpu = bool(cfg.get("use_gpu", True)) and torch.cuda.is_available()
    return torch.device("cuda" if use_gpu else "cpu")


def extract_single_feature(args: argparse.Namespace) -> None:
    if not args.single_image.strip():
        raise ValueError("--single-image is required in extract-single mode.")
    if not args.single_feature_out.strip():
        raise ValueError("--single-feature-out is required in extract-single mode.")
    if not args.single_meta_out.strip():
        raise ValueError("--single-meta-out is required in extract-single mode.")

    cfg = build_runtime_config(args)
    ps.config = cfg
    device = get_device(cfg)
    model = ps.build_model(device)

    image_path = Path(args.single_image)
    feature_out = Path(args.single_feature_out)
    meta_out = Path(args.single_meta_out)
    feature_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)

    feature_data = extract_feature_cached(model, image_path, device)
    torch.save(feature_data["feature"], feature_out)
    meta_out.write_text(
        json.dumps(feature_data["meta"], indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_feature_worker(
    image_path: Path,
    feature_out: Path,
    meta_out: Path,
    args: argparse.Namespace,
) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "extract-single",
        "--single-image",
        str(image_path),
        "--single-feature-out",
        str(feature_out),
        "--single-meta-out",
        str(meta_out),
    ]
    if args.use_gpu:
        cmd.append("--use-gpu")
    if args.model_path.strip():
        cmd.extend(["--model-path", args.model_path.strip()])
    if args.feature_type.strip():
        cmd.extend(["--feature-type", args.feature_type.strip()])
    if args.fusion_alpha >= 0.0:
        cmd.extend(["--fusion-alpha", str(args.fusion_alpha)])

    result = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode == 0:
        return

    raise RuntimeError(
        "Single-image feature extraction failed.\n"
        f"Image: {image_path}\n"
        f"Feature out: {feature_out}\n"
        f"STDOUT:\n{result.stdout[-4000:]}\n"
        f"STDERR:\n{result.stderr[-4000:]}"
    )


def load_feature_cache(feature_path: Path, meta_path: Path) -> Dict[str, Any]:
    feature = torch.load(feature_path, map_location="cpu")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {"feature": feature, "meta": meta}


def group_images(image_dir: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        group_key = path.stem.split("-", 1)[0]
        groups[group_key].append(path)
    return dict(sorted(groups.items(), key=lambda item: int(item[0])))


def resize_binary_image(src_path: Path, dst_path: Path, long_edge: int) -> Tuple[int, int]:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        gray = img.convert("L")
        width, height = gray.size
        scale = float(long_edge) / float(max(width, height))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = gray.resize((new_width, new_height), Image.LANCZOS)
        binary = resized.point(lambda x: 255 if x >= 128 else 0, mode="L")
        binary.save(dst_path, format="PNG")
    return new_width, new_height


def prepare_downscaled_variants(
    groups: Dict[str, List[Path]], output_dir: Path, long_edge: int
) -> Tuple[Dict[Path, Path], Dict[str, Dict[str, int]]]:
    variant_dir = output_dir / "downscaled_anchor_1k"
    variant_map: Dict[Path, Path] = {}
    size_map: Dict[str, Dict[str, int]] = {}
    for paths in groups.values():
        for src_path in paths:
            dst_path = variant_dir / f"{src_path.stem}_1k.png"
            new_width, new_height = resize_binary_image(src_path, dst_path, long_edge)
            variant_map[src_path] = dst_path
            size_map[str(src_path)] = {"width": new_width, "height": new_height}
    return variant_map, size_map


def extract_feature_cached(
    model: ps.MoCoV3HybridTopo,
    image_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    feat, meta = ps.extract_feature(model, str(image_path), device, rotation_deg=0)
    feat = torch.nn.functional.normalize(feat.unsqueeze(0), dim=1).squeeze(0).cpu()
    gc.collect()
    return {"feature": feat, "meta": meta}


def cosine_similarity(feat_a: torch.Tensor, feat_b: torch.Tensor) -> float:
    return float(torch.sum(feat_a * feat_b).item())


def summarize_results(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        raise ValueError("Result dataframe is empty.")

    group_summary_df = (
        df.groupby("group_id")
        .agg(
            pair_count=("pair_id", "count"),
            mean_cosine_original=("cosine_original", "mean"),
            mean_cosine_1k=("cosine_1k", "mean"),
            mean_delta_cosine=("delta_cosine", "mean"),
            median_delta_cosine=("delta_cosine", "median"),
            positive_delta_count=("delta_cosine", lambda s: int((s > 0).sum())),
            negative_delta_count=("delta_cosine", lambda s: int((s < 0).sum())),
        )
        .reset_index()
    )
    group_summary: List[Dict[str, Any]] = []
    for row in group_summary_df.to_dict(orient="records"):
        group_summary.append(
            {
                "group_id": str(row["group_id"]),
                "pair_count": int(row["pair_count"]),
                "mean_cosine_original": float(row["mean_cosine_original"]),
                "mean_cosine_1k": float(row["mean_cosine_1k"]),
                "mean_delta_cosine": float(row["mean_delta_cosine"]),
                "median_delta_cosine": float(row["median_delta_cosine"]),
                "positive_delta_count": int(row["positive_delta_count"]),
                "negative_delta_count": int(row["negative_delta_count"]),
            }
        )

    summary = {
        "pair_count": int(len(df)),
        "group_count": int(df["group_id"].nunique()),
        "image_count": int(pd.unique(df[["anchor_image", "other_image"]].values.ravel("K")).size),
        "mean_cosine_original": float(df["cosine_original"].mean()),
        "mean_cosine_1k": float(df["cosine_1k"].mean()),
        "mean_delta_cosine": float(df["delta_cosine"].mean()),
        "median_delta_cosine": float(df["delta_cosine"].median()),
        "std_delta_cosine": float(df["delta_cosine"].std(ddof=0)),
        "positive_delta_ratio": float((df["delta_cosine"] > 0).mean()),
        "negative_delta_ratio": float((df["delta_cosine"] < 0).mean()),
        "group_summary": group_summary,
    }
    return summary


def make_summary_figure(df: pd.DataFrame, summary: Dict[str, Any], output_path: Path) -> None:
    group_means = (
        df.groupby("group_id")["delta_cosine"]
        .mean()
        .reset_index()
        .sort_values("group_id", key=lambda s: s.astype(int))
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)

    axes[0].boxplot(
        [df["cosine_original"], df["cosine_1k"]],
        tick_labels=["Original", "1k Anchor"],
        patch_artist=True,
        boxprops={"facecolor": "#c9d7f0"},
        medianprops={"color": "#b22222", "linewidth": 1.5},
    )
    axes[0].set_title("Similarity Distribution")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].grid(axis="y", alpha=0.25)

    bar_colors = ["#3a7ca5" if x >= 0 else "#d1495b" for x in group_means["delta_cosine"]]
    axes[1].bar(group_means["group_id"].astype(str), group_means["delta_cosine"], color=bar_colors)
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_title("Mean Delta per Group (1k - Original)")
    axes[1].set_xlabel("Group")
    axes[1].set_ylabel("Mean Delta Cosine")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(
        (
            "Hybrid_v3 Resolution Impact Summary\n"
            f"Pairs={summary['pair_count']}  "
            f"Mean Original={summary['mean_cosine_original']:.4f}  "
            f"Mean 1k={summary['mean_cosine_1k']:.4f}  "
            f"Mean Delta={summary['mean_delta_cosine']:.4f}"
        ),
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_summary_markdown(
    summary: Dict[str, Any],
    output_path: Path,
    source_dir: Path,
    model_path: Path,
    downscale_long_edge: int,
) -> None:
    group_lines = []
    for row in summary["group_summary"]:
        direction = "升高" if row["mean_delta_cosine"] > 0 else "降低" if row["mean_delta_cosine"] < 0 else "基本不变"
        group_lines.append(
            (
                f"- 第 {row['group_id']} 组：{row['pair_count']} 个有向配对，"
                f"原始均值 {row['mean_cosine_original']:.4f}，"
                f"1k 均值 {row['mean_cosine_1k']:.4f}，"
                f"平均变化 {row['mean_delta_cosine']:+.4f}（{direction}）"
            )
        )

    mean_delta = summary["mean_delta_cosine"]
    if mean_delta > 0.01:
        overall_text = "将锚点图像降到 1k 后，整体相似度有明显上升。"
    elif mean_delta < -0.01:
        overall_text = "将锚点图像降到 1k 后，整体相似度有明显下降。"
    else:
        overall_text = "将锚点图像降到 1k 后，整体相似度变化较小。"

    content = "\n".join(
        [
            "# hybrid_v3 分辨率影响实验总结",
            "",
            f"- 数据目录：`{source_dir}`",
            f"- 模型：`{model_path}`",
            f"- 锚点图像下采样长边：`{downscale_long_edge}`",
            f"- 有向配对数：`{summary['pair_count']}`",
            f"- 组数：`{summary['group_count']}`",
            "",
            "## 总体结论",
            "",
            overall_text,
            "",
            f"- 原始分辨率平均余弦相似度：`{summary['mean_cosine_original']:.6f}`",
            f"- 1k 锚点平均余弦相似度：`{summary['mean_cosine_1k']:.6f}`",
            f"- 平均变化量（1k - 原始）：`{summary['mean_delta_cosine']:+.6f}`",
            f"- 中位变化量：`{summary['median_delta_cosine']:+.6f}`",
            f"- 正向变化占比：`{summary['positive_delta_ratio']:.2%}`",
            f"- 负向变化占比：`{summary['negative_delta_ratio']:.2%}`",
            "",
            "## 分组结果",
            "",
            *group_lines,
            "",
            "## 说明",
            "",
            "- 本实验使用你已经提取并二值化的人参图像，不再重复做人参提取。",
            "- 每条记录都是有向配对：只缩小锚点图像，另一张同组图像保持原分辨率不变。",
            "- 下采样时保持纵横比不变，并将长边缩放到 1000，然后再二值化保存为临时副本。",
        ]
    )
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.mode == "extract-single":
        extract_single_feature(args)
        return

    input_dir = choose_input_dir(Path(args.input_dir), Path(args.fallback_raw_dir))
    output_dir = build_output_dir(args.output_dir)
    records_path = output_dir / "resolution_impact_results.csv"
    figure_path = output_dir / "resolution_impact_summary.png"
    summary_json_path = output_dir / "resolution_impact_summary.json"
    summary_md_path = output_dir / "resolution_impact_summary.md"

    cfg = build_runtime_config(args)
    ps.config = cfg
    device = get_device(cfg)
    model_path = Path(ps.resolve_path(cfg["model_path"]))

    groups = group_images(input_dir)
    variant_map, downscaled_sizes = prepare_downscaled_variants(
        groups, output_dir, args.downscale_long_edge
    )

    print("=" * 70)
    print("hybrid_v3 分辨率影响批量实验")
    print("=" * 70)
    print(f"source_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"model_path: {model_path}")
    print(f"device: {device}")
    print(f"group_count: {len(groups)}")
    print(f"image_count: {sum(len(v) for v in groups.values())}")
    print(f"downscale_long_edge: {args.downscale_long_edge}")
    print("feature extraction mode: preprocess_pipeline=none, resize_mode=none")

    original_feature_cache: Dict[Path, Dict[str, Any]] = {}
    downscaled_feature_cache: Dict[Path, Dict[str, Any]] = {}
    feature_cache_dir = output_dir / "feature_cache"
    original_cache_dir = feature_cache_dir / "original"
    downscaled_cache_dir = feature_cache_dir / "anchor_1k"

    total_images = sum(len(v) for v in groups.values())
    processed_images = 0
    for group_id, paths in groups.items():
        for image_path in paths:
            processed_images += 1
            original_feature_path = original_cache_dir / f"{image_path.stem}.pt"
            original_meta_path = original_cache_dir / f"{image_path.stem}.json"
            downscaled_feature_path = downscaled_cache_dir / f"{image_path.stem}_1k.pt"
            downscaled_meta_path = downscaled_cache_dir / f"{image_path.stem}_1k.json"

            if original_feature_path.exists() and original_meta_path.exists():
                print(f"[{processed_images}/{total_images}] cache hit original: {group_id}/{image_path.name}")
            else:
                print(f"[{processed_images}/{total_images}] extracting original feature: {group_id}/{image_path.name}")
                run_feature_worker(
                    image_path=image_path,
                    feature_out=original_feature_path,
                    meta_out=original_meta_path,
                    args=args,
                )

            if downscaled_feature_path.exists() and downscaled_meta_path.exists():
                print(f"[{processed_images}/{total_images}] cache hit 1k: {group_id}/{image_path.name}")
            else:
                print(f"[{processed_images}/{total_images}] extracting 1k feature: {group_id}/{image_path.name}")
                run_feature_worker(
                    image_path=variant_map[image_path],
                    feature_out=downscaled_feature_path,
                    meta_out=downscaled_meta_path,
                    args=args,
                )

            original_feature_cache[image_path] = load_feature_cache(
                original_feature_path, original_meta_path
            )
            downscaled_feature_cache[image_path] = load_feature_cache(
                downscaled_feature_path, downscaled_meta_path
            )

    records: List[Dict[str, Any]] = []
    for group_id, paths in groups.items():
        for anchor_path in paths:
            for other_path in paths:
                if anchor_path == other_path:
                    continue

                anchor_original = original_feature_cache[anchor_path]
                anchor_1k = downscaled_feature_cache[anchor_path]
                other_original = original_feature_cache[other_path]

                cosine_original = cosine_similarity(
                    anchor_original["feature"], other_original["feature"]
                )
                cosine_1k = cosine_similarity(anchor_1k["feature"], other_original["feature"])
                score_original = max(0.0, min(1.0, (cosine_original + 1.0) / 2.0))
                score_1k = max(0.0, min(1.0, (cosine_1k + 1.0) / 2.0))

                anchor_meta = anchor_original["meta"]
                other_meta = other_original["meta"]
                downscaled_size = downscaled_sizes[str(anchor_path)]

                records.append(
                    {
                        "pair_id": f"{anchor_path.stem}__to__{other_path.stem}",
                        "group_id": group_id,
                        "anchor_image": anchor_path.name,
                        "other_image": other_path.name,
                        "anchor_original_width": anchor_meta["original_size"]["width"],
                        "anchor_original_height": anchor_meta["original_size"]["height"],
                        "anchor_1k_width": downscaled_size["width"],
                        "anchor_1k_height": downscaled_size["height"],
                        "other_original_width": other_meta["original_size"]["width"],
                        "other_original_height": other_meta["original_size"]["height"],
                        "cosine_original": cosine_original,
                        "cosine_1k": cosine_1k,
                        "delta_cosine": cosine_1k - cosine_original,
                        "score_original": score_original,
                        "score_1k": score_1k,
                        "delta_score": score_1k - score_original,
                    }
                )

    df = pd.DataFrame(records)
    df["_group_order"] = df["group_id"].astype(int)
    df = df.sort_values(by=["_group_order", "anchor_image", "other_image"]).drop(
        columns=["_group_order"]
    )
    df.to_csv(records_path, index=False, encoding="utf-8-sig")

    summary = summarize_results(df)
    make_summary_figure(df, summary, figure_path)
    summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_summary_markdown(
        summary=summary,
        output_path=summary_md_path,
        source_dir=input_dir,
        model_path=model_path,
        downscale_long_edge=args.downscale_long_edge,
    )

    print("\n输出文件:")
    print(f"- CSV: {records_path}")
    print(f"- Figure: {figure_path}")
    print(f"- Summary JSON: {summary_json_path}")
    print(f"- Summary Markdown: {summary_md_path}")
    print("\n总体统计:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
