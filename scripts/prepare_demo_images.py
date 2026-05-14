from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compress catalog meal images for the interactive demo."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/catalog/three_component/train/catalog_manifest.csv"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/catalog_demo_images"),
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--quality", type=int, default=72)
    parser.add_argument("--format", choices=("webp", "jpeg"), default="webp")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with args.manifest.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit is not None:
        rows = rows[: args.limit]

    suffix = ".webp" if args.format == "webp" else ".jpg"
    written = 0
    skipped = 0
    missing = 0

    for row in tqdm(rows, desc="compressing catalog images"):
        catalog_id = row.get("catalog_id") or row.get("dish_id") or row.get("id")
        image_path = row.get("image_path") or row.get("image_paths")
        if not catalog_id or not image_path:
            skipped += 1
            continue

        first_image = image_path.split(";")[0].split("|")[0].strip()
        source = repo_root / first_image
        if not source.exists():
            missing += 1
            continue

        target = output_dir / f"{catalog_id}{suffix}"
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((args.size, args.size), Image.Resampling.LANCZOS)
            if args.format == "webp":
                image.save(target, "WEBP", quality=args.quality, method=6)
            else:
                image.save(target, "JPEG", quality=args.quality, optimize=True)
        written += 1

    print(
        f"wrote={written} skipped={skipped} missing={missing} "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
