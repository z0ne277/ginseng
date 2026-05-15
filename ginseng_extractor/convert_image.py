import os
import argparse
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def ensure_folder_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def _iter_images(input_root, suffixes, recursive):
    root = Path(input_root)
    if not root.exists():
        return []
    if recursive:
        return [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in suffixes]
    return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in suffixes]


def _estimate_background_rgb(rgb_array, border_ratio=0.02):
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


def _remove_small_components(mask, min_area):
    if min_area <= 0 or np.count_nonzero(mask) == 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label_idx in range(1, num_labels):
        if stats[label_idx, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_idx] = 255
    return cleaned if np.count_nonzero(cleaned) else mask


def _build_foreground_mask(
    image,
    binary_mode,
    threshold,
    foreground_threshold,
    close_kernel_size,
    close_iterations,
    min_area,
    alpha_threshold,
    dark_background_max,
):
    image_array = np.array(image)

    if image_array.ndim == 2:
        gray_array = image_array
        background_rgb = np.array([0, 0, 0], dtype=np.uint8)
        has_alpha = False
    else:
        has_alpha = image_array.shape[2] == 4
        rgb_array = image_array[:, :, :3]
        gray_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        background_rgb = _estimate_background_rgb(rgb_array)

    if binary_mode == "gray-threshold":
        mask = np.where(gray_array >= threshold, 255, 0).astype(np.uint8)
    else:
        use_alpha_mask = has_alpha
        use_foreground_mask = binary_mode == "foreground-mask"
        if binary_mode == "auto" and not use_alpha_mask:
            use_foreground_mask = int(background_rgb.max()) <= dark_background_max

        if use_alpha_mask:
            alpha_channel = image_array[:, :, 3]
            mask = np.where(alpha_channel >= alpha_threshold, 255, 0).astype(np.uint8)
        elif use_foreground_mask:
            diff = np.max(
                np.abs(rgb_array.astype(np.int16) - background_rgb.astype(np.int16)),
                axis=2,
            )
            mask = np.where(diff > foreground_threshold, 255, 0).astype(np.uint8)
        else:
            mask = np.where(gray_array >= threshold, 255, 0).astype(np.uint8)

    if close_kernel_size > 1 and close_iterations > 0 and np.count_nonzero(mask) > 0:
        kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)

    return gray_array, _remove_small_components(mask, min_area)


def convert_to_grayscale_and_binary(
    input_root,
    output_root_gray,
    output_root_binary,
    threshold=128,
    recursive=True,
    suffixes=IMAGE_SUFFIXES,
    binary_mode="auto",
    foreground_threshold=12,
    close_kernel_size=3,
    close_iterations=2,
    min_area=20,
    alpha_threshold=1,
    dark_background_max=8,
):
    input_root = Path(input_root)
    output_root_gray = Path(output_root_gray)
    output_root_binary = Path(output_root_binary)
    ensure_folder_exists(output_root_gray)
    ensure_folder_exists(output_root_binary)

    image_paths = _iter_images(input_root, suffixes, recursive)
    for img_path in image_paths:
        try:
            try:
                rel_path = img_path.relative_to(input_root)
            except ValueError:
                rel_path = Path(img_path.name)
            gray_out_path = output_root_gray / rel_path
            binary_out_path = output_root_binary / rel_path
            gray_out_path.parent.mkdir(parents=True, exist_ok=True)
            binary_out_path.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(img_path) as img:
                gray_array, mask = _build_foreground_mask(
                    img,
                    binary_mode=binary_mode,
                    threshold=threshold,
                    foreground_threshold=foreground_threshold,
                    close_kernel_size=close_kernel_size,
                    close_iterations=close_iterations,
                    min_area=min_area,
                    alpha_threshold=alpha_threshold,
                    dark_background_max=dark_background_max,
                )
                gray_array = gray_array.copy()
                gray_array[mask == 0] = 0
                gray_img = Image.fromarray(gray_array)
                gray_img.save(gray_out_path)
                print(f"Saved gray: {gray_out_path}")

                binary_img = Image.fromarray(mask)
                binary_img.save(binary_out_path)
                print(f"Saved binary: {binary_out_path}")
        except Exception as e:
            print(f"Failed to process: {img_path}, error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to grayscale and binary with folder preservation.")
    parser.add_argument("--input-root", default="./processed_result", help="Input image root.")
    parser.add_argument("--output-gray", default="./processed_result_gray", help="Gray output root.")
    parser.add_argument("--output-binary", default="./processed_result_binary", help="Binary output root.")
    parser.add_argument("--threshold", type=int, default=128, help="Binary threshold.")
    parser.add_argument(
        "--binary-mode",
        choices=("auto", "foreground-mask", "gray-threshold"),
        default="auto",
        help="auto: dark background/alpha uses foreground mask; gray-threshold keeps legacy intensity thresholding.",
    )
    parser.add_argument(
        "--foreground-threshold",
        type=int,
        default=12,
        help="Pixel distance to background when foreground-mask mode is used.",
    )
    parser.add_argument(
        "--close-kernel-size",
        type=int,
        default=3,
        help="Kernel size for binary closing to fill small texture gaps inside ginseng.",
    )
    parser.add_argument(
        "--close-iterations",
        type=int,
        default=2,
        help="Iterations for binary closing.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=20,
        help="Remove connected components smaller than this area.",
    )
    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=1,
        help="Foreground alpha threshold when input image has transparency.",
    )
    parser.add_argument(
        "--dark-background-max",
        type=int,
        default=8,
        help="When binary-mode=auto and border background max value is below this, use foreground mask extraction.",
    )
    parser.add_argument("--recursive", action="store_true", help="Scan subfolders recursively.")
    parser.add_argument("--suffixes", default=None, help="Comma-separated suffixes, e.g. .jpg,.png")
    args = parser.parse_args()

    suffixes = IMAGE_SUFFIXES
    if args.suffixes:
        suffixes = tuple(s.strip().lower() for s in args.suffixes.split(',') if s.strip())

    convert_to_grayscale_and_binary(
        args.input_root,
        args.output_gray,
        args.output_binary,
        threshold=args.threshold,
        recursive=args.recursive,
        suffixes=suffixes,
        binary_mode=args.binary_mode,
        foreground_threshold=args.foreground_threshold,
        close_kernel_size=args.close_kernel_size,
        close_iterations=args.close_iterations,
        min_area=args.min_area,
        alpha_threshold=args.alpha_threshold,
        dark_background_max=args.dark_background_max,
    )
