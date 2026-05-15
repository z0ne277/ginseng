\
\
\
\
\
\


import argparse
from pathlib import Path
from typing import Iterable, Set

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def list_image_basenames(folder: Path, suffixes: Iterable[str]) -> Set[str]:
    if not folder.exists():
        return set()
    names = set()
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in suffixes:
            names.add(p.name)
    return names


def should_remove(name: str, remove_prefixes: Iterable[str]) -> bool:
    lower = name.lower()
    return any(lower.startswith(p) for p in remove_prefixes)


def main():
    parser = argparse.ArgumentParser(description="Clean ginseng_extraction folders.")
    parser.add_argument("--extraction-root", required=True, help="Root of extracted images.")
    parser.add_argument("--same-root", required=True, help="Root of original same_ginseng folders.")
    parser.add_argument("--suffixes", default=None, help="Comma-separated suffixes, e.g. .jpg,.png")
    parser.add_argument(
        "--remove-prefixes",
        default="grounded_sam,mask_,raw_,temp_",
        help="Comma-separated prefixes to remove."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without deleting.")
    args = parser.parse_args()

    suffixes = IMAGE_SUFFIXES
    if args.suffixes:
        suffixes = tuple(s.strip().lower() for s in args.suffixes.split(',') if s.strip())

    remove_prefixes = [p.strip().lower() for p in args.remove_prefixes.split(',') if p.strip()]

    extraction_root = Path(args.extraction_root)
    same_root = Path(args.same_root)

    if not extraction_root.exists():
        print(f"Extraction root not found: {extraction_root}")
        return

    group_dirs = [p for p in extraction_root.iterdir() if p.is_dir()]
    if not group_dirs:
        print(f"No subfolders found in: {extraction_root}")
        return

    removed = 0
    kept = 0

    for group_dir in group_dirs:
        same_dir = same_root / group_dir.name
        keep_names = list_image_basenames(same_dir, suffixes)

        for p in group_dir.iterdir():
            if not p.is_file() or p.suffix.lower() not in suffixes:
                continue

            remove = False
            if should_remove(p.name, remove_prefixes):
                remove = True
            elif keep_names and p.name not in keep_names:
                remove = True

            if remove:
                removed += 1
                if args.dry_run:
                    print(f"[DRY] remove: {p}")
                else:
                    p.unlink(missing_ok=True)
            else:
                kept += 1

    print(f"Done. kept={kept}, removed={removed}")


if __name__ == "__main__":
    main()
