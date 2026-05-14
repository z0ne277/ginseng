"""
单张人参图像的数据增强演示脚本（hybrid_v3 风格）

用途：
- 输入一张已经经过 ginseng_extractor 处理的“单只人参”图片（建议来自 library_processed 或 library_binary）
- 使用 hybrid_v3 中 UnsupervisedContrastiveDataset 的增强逻辑
- 输出两张不同的数据增强图像，可用于 encoder_q / encoder_k 两路输入的示意

注意：
- 本脚本不改动任何现有代码，只复用 UnsupervisedContrastiveDataset 的 _binarize 和 augment_image。
- 默认会先做一次自适应二值化，再生成两份随机增强视图。
"""

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
    """
    构造一个仅用于复用增强逻辑的 UnsupervisedContrastiveDataset 实例。
    - 使用 hybrid_v3/csv/train.csv 作为占位 CSV（不会真正使用其中的数据行）
    - 不传 transform，只调用其 _binarize 和 augment_image
    """
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
    """
    输入一张已经是“单只人参”的图像（黑底或白底均可），输出两张增强后的图：
    - view_q：建议送入 encoder_q
    - view_k：建议送入 encoder_k

    返回值：
        (view_q_path, view_k_path)
    """
    base_dir = Path(__file__).resolve().parent
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"输入图像不存在: {img_path}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. 构造增强 helper（复用 hybrid_v3 的增强风格）
    aug_helper = build_augment_helper(
        base_dir=base_dir,
        augment_strength=augment_strength,
        binarization_threshold=binarization_threshold,
    )

    # 2. 读取原图并转为灰度（与 UnsupervisedContrastiveDataset 一致）
    image = Image.open(img_path).convert("L")

    # 3. 先做一次自适应二值化（对应“ginseng_extractor 后 + 二值化”的阶段）
    binary_img = aug_helper._binarize(image)

    # 4. 生成两份随机增强视图（与 __getitem__ 中生成 img1 / img2 的逻辑一致）
    view_q = aug_helper.augment_image(binary_img.copy())
    view_k = aug_helper.augment_image(binary_img.copy())

    # 5. 保存到指定目录，文件名基于原始文件名
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

