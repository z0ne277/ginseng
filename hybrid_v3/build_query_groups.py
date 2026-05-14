"""
Build query_groups.json for batch_retrieval.py.

Rule:
- same_root has subfolders (1,2,3...). Each subfolder contains images of the same ginseng.
- gallery_root is the retrieval library root. Image paths are gallery_root + filename.
- For each group, pick one image as query_image; others are same_ginsengs.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(folder: Path, suffixes) -> List[Path]:
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in suffixes]


def choose_query(images: List[Path], strategy: str, seed: int) -> Path:
    if strategy == "first":
        return images[0]
    if strategy == "random":
        rng = random.Random(seed)
        return rng.choice(images)
    return images[0]


def iter_queries(images: List[Path], strategy: str, seed: int, all_queries: bool) -> List[Path]:
    if all_queries:
        return images
    return [choose_query(images, strategy, seed)]


def main():
    parser = argparse.ArgumentParser(description="Build query_groups.json from same_ginseng folders.")
    parser.add_argument("--gallery-root", required=True, help="Retrieval gallery root folder.")
    parser.add_argument("--same-root", required=True, help="Root folder containing same_ginseng groups.")
    parser.add_argument("--output-json", required=True, help="Output query_groups.json path.")
    parser.add_argument("--suffixes", default=None, help="Comma-separated suffixes, e.g. .jpg,.png")
    parser.add_argument("--query-strategy", choices=["first", "random"], default="first")
    parser.add_argument("--all-queries", action="store_true", help="Use every image in group as query.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-images", type=int, default=2, help="Skip groups with fewer images.")
    args = parser.parse_args()

    suffixes = IMAGE_SUFFIXES
    if args.suffixes:
        suffixes = tuple(s.strip().lower() for s in args.suffixes.split(",") if s.strip())

    gallery_root = Path(args.gallery_root)
    same_root = Path(args.same_root)
    if not same_root.exists():
        raise FileNotFoundError(f"same_root not found: {same_root}")

    groups = []
    skipped = 0
    for group_dir in sorted([p for p in same_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        images = list_images(group_dir, suffixes)
        if len(images) < args.min_images:
            skipped += 1
            continue

        for query in iter_queries(images, args.query_strategy, args.seed, args.all_queries):
            same_images = [p for p in images if p.name != query.name]

            query_path = str(gallery_root / query.name)
            same_paths = [str(gallery_root / p.name) for p in same_images]

            groups.append({
                "name": group_dir.name,
                "query_image": query_path,
                "same_ginsengs": same_paths,
            })

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"query_groups": groups}, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_path} (groups={len(groups)}, skipped={skipped})")


if __name__ == "__main__":
    main()
