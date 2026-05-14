from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download meal catalog source images from data/meal_catalog.csv."
    )
    parser.add_argument(
        "--recipe-file",
        type=Path,
        default=Path("data/meal_catalog.csv"),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/catalog/images"),
    )
    parser.add_argument("--image-width", type=int, default=1920)
    parser.add_argument("--image-height", type=int, default=1080)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.recipe_file)
    if args.limit is not None:
        df = df.head(args.limit)

    written = 0
    skipped = 0
    failed = 0
    for index, row in tqdm(df.iterrows(), total=len(df), desc="downloading catalog images"):
        image_url = str(row.get("image_url", "")).strip()
        target = args.image_root / f"catalog_{index}" / "rgb.png"
        if not image_url:
            skipped += 1
            continue
        if target.exists():
            skipped += 1
            continue
        try:
            download_resize_image(
                image_url=image_url,
                image_width=args.image_width,
                image_height=args.image_height,
                output_path=target,
                timeout=args.timeout,
            )
        except Exception as exc:  # noqa: BLE001 - report all failed source URLs
            failed += 1
            print(f"failed catalog_{index}: {exc}")
            continue
        written += 1

    print(
        f"wrote={written} skipped={skipped} failed={failed} "
        f"image_root={args.image_root}"
    )


def download_resize_image(
    image_url: str,
    image_width: int,
    image_height: int,
    output_path: Path,
    timeout: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(image_url, timeout=timeout)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content)).convert("RGB")
    scale = min(image_width / image.width, image_height / image.height)
    new_size = (int(image.width * scale), int(image.height * scale))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    padded = Image.new("RGB", (image_width, image_height), (0, 0, 0))
    x = (image_width - image.width) // 2
    y = (image_height - image.height) // 2
    padded.paste(image, (x, y))
    padded.save(output_path)


if __name__ == "__main__":
    main()
