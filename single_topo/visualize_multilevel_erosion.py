from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "visualizations_erosion"


def resolve_image_path(image: str | None, image_dir: str | None, image_pattern: str) -> Path:
    if image:
        path = Path(image)
        if path.exists():
            return path
        raise FileNotFoundError(f"Image not found: {path}")

    if not image_dir:
        raise ValueError("Either --image or --image-dir must be provided.")

    matches = sorted(Path(image_dir).glob(image_pattern))
    if not matches:
        raise FileNotFoundError(
            f"No image matched pattern '{image_pattern}' under '{image_dir}'."
        )
    if len(matches) > 1:
        print(f"[info] matched {len(matches)} files, using: {matches[0]}")
    return matches[0]


def parse_box(text: str | None) -> Tuple[int, int, int, int] | None:
    if not text:
        return None
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid box '{text}'. Expected x0,y0,x1,y1.")
    x0, y0, x1, y1 = [int(float(part)) for part in parts]
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid box '{text}'. Require x1>x0 and y1>y0.")
    return x0, y0, x1, y1


def parse_boxes(texts: Iterable[str] | None) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    if not texts:
        return boxes
    for text in texts:
        box = parse_box(text)
        if box is not None:
            boxes.append(box)
    return boxes


def clamp_box(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0 = max(0, min(width, x0))
    y0 = max(0, min(height, y0))
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Box {box} is outside image size ({width}, {height}).")
    return x0, y0, x1, y1


def estimate_background_rgb(rgb_array: np.ndarray, border_ratio: float = 0.02) -> np.ndarray:
    height, width = rgb_array.shape[:2]
    border_width = max(1, int(min(height, width) * border_ratio))
    border_pixels = np.concatenate(
        [
            rgb_array[:border_width, :, :].reshape(-1, 3),
            rgb_array[-border_width:, :, :].reshape(-1, 3),
            rgb_array[:, :border_width, :].reshape(-1, 3),
            rgb_array[:, -border_width:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(border_pixels, axis=0).astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0 or np.count_nonzero(mask) == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label_idx] = 255
    return cleaned


def apply_ignore_boxes(
    mask: np.ndarray,
    ignore_boxes: List[Tuple[int, int, int, int]],
    offset_x: int = 0,
    offset_y: int = 0,
) -> np.ndarray:
    if not ignore_boxes:
        return mask

    height, width = mask.shape[:2]
    masked = mask.copy()
    for box in ignore_boxes:
        x0, y0, x1, y1 = box
        x0 -= offset_x
        x1 -= offset_x
        y0 -= offset_y
        y1 -= offset_y
        x0 = max(0, min(width, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(0, min(height, y1))
        if x1 > x0 and y1 > y0:
            masked[y0:y1, x0:x1] = 0
    return masked


def build_foreground_mask(
    rgb_array: np.ndarray,
    foreground_threshold: int,
    saturation_threshold: int,
    chroma_threshold: int,
    dark_threshold: int,
    value_ceiling: int,
    open_kernel_size: int,
    close_kernel_size: int,
    morph_iterations: int,
    min_area: int,
    ignore_boxes: List[Tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    background_rgb = estimate_background_rgb(rgb_array)
    diff = np.max(
        np.abs(rgb_array.astype(np.int16) - background_rgb.astype(np.int16)),
        axis=2,
    )

    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
    a_channel = lab[:, :, 1].astype(np.int16) - 128
    b_channel = lab[:, :, 2].astype(np.int16) - 128
    chroma = np.sqrt(a_channel * a_channel + b_channel * b_channel)

    color_signal = (saturation >= saturation_threshold) | (chroma >= chroma_threshold)
    dark_signal = value <= dark_threshold
    supported_by_background = diff >= foreground_threshold

    mask = np.where(
        (supported_by_background & (color_signal | dark_signal)) | dark_signal,
        255,
        0,
    ).astype(np.uint8)

    if ignore_boxes:
        mask = apply_ignore_boxes(mask, ignore_boxes)

    if open_kernel_size > 1 and morph_iterations > 0 and np.count_nonzero(mask) > 0:
        kernel_open = np.ones((open_kernel_size, open_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=morph_iterations)

    if close_kernel_size > 1 and morph_iterations > 0 and np.count_nonzero(mask) > 0:
        kernel_close = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=morph_iterations)

    if value_ceiling < 255:
        weak_background = np.where(value > value_ceiling, 0, 255).astype(np.uint8)
        weak_background = cv2.dilate(weak_background, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.bitwise_and(mask, weak_background)

    return remove_small_components(mask, min_area=min_area)


def apply_binary_erosion(mask: np.ndarray, iterations: int, kernel_size: int = 3) -> np.ndarray:
    if iterations <= 0:
        return mask.copy()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(mask, kernel, iterations=iterations)


def render_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 0, 255).astype(np.uint8)


def save_panel_image(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        Image.fromarray(image).save(output_path)
    else:
        Image.fromarray(image.astype(np.uint8)).save(output_path)


def crop_rgb_with_roi(
    rgb: np.ndarray,
    roi: Tuple[int, int, int, int] | None,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    height, width = rgb.shape[:2]
    if roi is None:
        return rgb, (0, 0, width, height)
    x0, y0, x1, y1 = clamp_box(roi, width=width, height=height)
    return rgb[y0:y1, x0:x1], (x0, y0, x1, y1)


def build_visuals(
    image_path: Path,
    output_dir: Path,
    roi: Tuple[int, int, int, int] | None,
    ignore_boxes: List[Tuple[int, int, int, int]],
    foreground_threshold: int,
    saturation_threshold: int,
    chroma_threshold: int,
    dark_threshold: int,
    value_ceiling: int,
    open_kernel_size: int,
    close_kernel_size: int,
    morph_iterations: int,
    min_area: int,
    num_levels: int,
) -> Dict[str, Path]:
    rgb = np.array(Image.open(image_path).convert("RGB"))
    rgb_crop, used_roi = crop_rgb_with_roi(rgb, roi)
    x0, y0, _, _ = used_roi

    mask_crop = build_foreground_mask(
        rgb_crop,
        foreground_threshold=foreground_threshold,
        saturation_threshold=saturation_threshold,
        chroma_threshold=chroma_threshold,
        dark_threshold=dark_threshold,
        value_ceiling=value_ceiling,
        open_kernel_size=open_kernel_size,
        close_kernel_size=close_kernel_size,
        morph_iterations=morph_iterations,
        min_area=min_area,
        ignore_boxes=ignore_boxes,
    )

    foreground_crop = np.full_like(rgb_crop, 255)
    foreground_crop[mask_crop > 0] = rgb_crop[mask_crop > 0]

    level_masks: List[np.ndarray] = []
    for level in range(num_levels):
        level_masks.append(apply_binary_erosion(mask_crop, iterations=level, kernel_size=3))

    stem = image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, Path] = {}
    roi_suffix = ""
    if roi is not None:
        roi_suffix = f"_roi_{roi[0]}_{roi[1]}_{roi[2]}_{roi[3]}"

    input_path = output_dir / f"{stem}{roi_suffix}_input_crop.png"
    foreground_path = output_dir / f"{stem}{roi_suffix}_foreground.png"
    mask_path = output_dir / f"{stem}{roi_suffix}_mask.png"
    save_panel_image(rgb_crop, input_path)
    save_panel_image(foreground_crop, foreground_path)
    save_panel_image(render_mask(mask_crop), mask_path)
    saved["input_crop"] = input_path
    saved["foreground"] = foreground_path
    saved["mask"] = mask_path

    for level, level_mask in enumerate(level_masks):
        level_path = output_dir / f"{stem}{roi_suffix}_erosion_level_{level}.png"
        save_panel_image(render_mask(level_mask), level_path)
        saved[f"erosion_level_{level}"] = level_path

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 12,
            "axes.unicode_minus": False,
        }
    )

    strip_panels = [("Input", rgb_crop)]
    strip_panels.extend(
        [(f"Erosion L{level}", render_mask(level_mask)) for level, level_mask in enumerate(level_masks)]
    )
    strip_fig, strip_axes = plt.subplots(1, len(strip_panels), figsize=(4.5 * len(strip_panels), 4.6), dpi=220)
    if len(strip_panels) == 1:
        strip_axes = [strip_axes]
    for ax, (title, panel) in zip(strip_axes, strip_panels):
        if panel.ndim == 2:
            ax.imshow(panel, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(panel)
        ax.set_title(title, pad=8)
        ax.axis("off")
    strip_fig.tight_layout(w_pad=0.8)
    strip_path = output_dir / f"{stem}{roi_suffix}_multilevel_erosion_strip.png"
    strip_fig.savefig(strip_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(strip_fig)
    saved["strip"] = strip_path

    overview_panels = [("Input", rgb_crop), ("Foreground", foreground_crop)]
    overview_panels.extend(
        [(f"Erosion L{level}", render_mask(level_mask)) for level, level_mask in enumerate(level_masks)]
    )
    overview_cols = min(3, len(overview_panels))
    overview_rows = int(np.ceil(len(overview_panels) / overview_cols))
    overview_fig, overview_axes = plt.subplots(
        overview_rows,
        overview_cols,
        figsize=(5.4 * overview_cols, 4.2 * overview_rows),
        dpi=220,
    )
    axes_array = np.array(overview_axes).reshape(-1)
    for ax, (title, panel) in zip(axes_array, overview_panels):
        if panel.ndim == 2:
            ax.imshow(panel, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(panel)
        ax.set_title(title, pad=8)
        ax.axis("off")
    for ax in axes_array[len(overview_panels):]:
        ax.axis("off")
    overview_fig.tight_layout(w_pad=0.8, h_pad=1.0)
    overview_path = output_dir / f"{stem}{roi_suffix}_multilevel_erosion_overview.png"
    overview_fig.savefig(overview_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(overview_fig)
    saved["overview"] = overview_path

    print(f"[info] used roi: {used_roi}")
    if ignore_boxes:
        print(f"[info] ignored boxes inside crop offset ({x0}, {y0}): {ignore_boxes}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the single-topo branch using multi-level erosion on one ginseng image."
    )
    parser.add_argument("--image", default=None, help="Direct image path.")
    parser.add_argument("--image-dir", default=None, help="Image directory for glob matching.")
    parser.add_argument("--image-pattern", default="*.jpg", help="Glob pattern under --image-dir.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--roi", default=None, help="Manual ROI as x0,y0,x1,y1.")
    parser.add_argument(
        "--ignore-box",
        action="append",
        default=[],
        help="Ignore region as x0,y0,x1,y1. Can be used multiple times.",
    )
    parser.add_argument("--foreground-threshold", type=int, default=12, help="Background difference threshold.")
    parser.add_argument("--saturation-threshold", type=int, default=15, help="HSV saturation threshold.")
    parser.add_argument("--chroma-threshold", type=int, default=10, help="LAB chroma threshold.")
    parser.add_argument("--dark-threshold", type=int, default=170, help="Dark pixel threshold in HSV value.")
    parser.add_argument(
        "--value-ceiling",
        type=int,
        default=245,
        help="Background value ceiling used to suppress bright cloth regions.",
    )
    parser.add_argument("--open-kernel-size", type=int, default=3, help="Morphological opening kernel size.")
    parser.add_argument("--close-kernel-size", type=int, default=3, help="Morphological closing kernel size.")
    parser.add_argument("--morph-iterations", type=int, default=1, help="Morphological iterations.")
    parser.add_argument("--min-area", type=int, default=30, help="Minimum connected-component area.")
    parser.add_argument("--num-levels", type=int, default=4, help="Number of erosion levels.")
    args = parser.parse_args()

    image_path = resolve_image_path(args.image, args.image_dir, args.image_pattern)
    output_dir = Path(args.output_dir)
    roi = parse_box(args.roi)
    ignore_boxes = parse_boxes(args.ignore_box)

    saved = build_visuals(
        image_path=image_path,
        output_dir=output_dir,
        roi=roi,
        ignore_boxes=ignore_boxes,
        foreground_threshold=args.foreground_threshold,
        saturation_threshold=args.saturation_threshold,
        chroma_threshold=args.chroma_threshold,
        dark_threshold=args.dark_threshold,
        value_ceiling=args.value_ceiling,
        open_kernel_size=args.open_kernel_size,
        close_kernel_size=args.close_kernel_size,
        morph_iterations=args.morph_iterations,
        min_area=args.min_area,
        num_levels=args.num_levels,
    )

    print(f"[done] image: {image_path}")
    for key, path in saved.items():
        print(f"[saved] {key}: {path}")


if __name__ == "__main__":
    main()
