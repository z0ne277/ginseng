\
\
\
\
\
\
\
\
\
\
\


import argparse
import os
from pathlib import Path

from PIL import Image

from UnsupervisedContrastiveDataset import UnsupervisedContrastiveDataset


def build_augment_helper(
    base_dir: Path,
    augment_strength: str = "medium",
    binarization_threshold: int = 128,
) -> UnsupervisedContrastiveDataset:
\
\
\
\

    csv_path = base_dir / "csv" / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"占位 CSV 未找到: {csv_path}")

    helper = UnsupervisedContrastiveDataset(
        csv_file=str(csv_path),
        transform=None,
        use_augment=True,
        use_binarization=True,
        binarization_threshold=binarization_threshold,
        augment_strength=augment_strength,
    )
    return helper


def generate_two_views_for_encoder(
    image_path: str,
    output_dir: str,
    augment_strength: str = "medium",
    binarization_threshold: int = 128,
) -> tuple[str, str]:
\
\
\
\
\
\
\

    base_dir = Path(__file__).resolve().parent
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"输入图像不存在: {img_path}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)


    aug_helper = build_augment_helper(
        base_dir=base_dir,
        augment_strength=augment_strength,
        binarization_threshold=binarization_threshold,
    )


    image = Image.open(img_path).convert("L")


    binary_img = aug_helper._binarize(image)


    view_q = aug_helper.augment_image(binary_img.copy())
    view_k = aug_helper.augment_image(binary_img.copy())


    stem = img_path.stem
    suffix = img_path.suffix or ".jpg"
    view_q_path = output_root / f"{stem}_encoder_q_aug{suffix}"
    view_k_path = output_root / f"{stem}_encoder_k_aug{suffix}"

    view_q.save(view_q_path)
    view_k.save(view_k_path)

    return str(view_q_path), str(view_k_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="单张人参图像的 hybrid_v3 风格数据增强演示��输入 1 张图，输出 2 张增强图（encoder_q / encoder_k）"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="输入人参图像路径（建议为 ginseng_extractor + 后处理后的单只人参图）",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        default="./demo_augment_outputs",
        help="增强结果输出目录（默认 ./demo_augment_outputs）",
    )
    parser.add_argument(
        "--augment-strength",
        choices=["light", "medium", "strong"],
        default="medium",
        help="增强强度（默认 medium，与训练默认配置一致）",
    )
    parser.add_argument(
        "--binarization-threshold",
        type=int,
        default=128,
        help="二值化阈值（目前 _binarize 使用自适应阈值，此参数预留用于兼容）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    view_q_path, view_k_path = generate_two_views_for_encoder(
        image_path=args.image,
        output_dir=args.output_dir,
        augment_strength=args.augment_strength,
        binarization_threshold=args.binarization_threshold,
    )

    print("=== Hybrid_v3 单张图数据增强演示 ===")
    print(f"输入图像: {os.path.abspath(args.image)}")
    print(f"增强强度: {args.augment_strength}")
    print(f"encoder_q 增强图: {os.path.abspath(view_q_path)}")
    print(f"encoder_k 增强图: {os.path.abspath(view_k_path)}")


if __name__ == "__main__":
    main()
