from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from visionmealrl.constants import IMAGE_EXTENSIONS, TARGET_COLUMNS

LOGGER = logging.getLogger(__name__)


# a dish only ever has one DishData during processing
@dataclass(frozen=True)
class DishData:
    total_calories: float
    total_mass: float
    total_fat: float
    total_carb: float
    total_protein: float

    def as_list(self) -> list[float]:
        return [getattr(self, column) for column in TARGET_COLUMNS]


# a dish may have multiple images, dish_id and dish_data will be the same
@dataclass(frozen=True)
class ImageRecord:
    dish_id: str
    image_path: Path
    targets: DishData


def _resolve_existing(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Expected path does not exist: {path}")
    return path


def load_dish_targets(dataset_root: Path) -> dict[str, DishData]:
    metadata_dir = _resolve_existing(dataset_root / "metadata")
    csv_paths = [
        metadata_dir / "dish_metadata_cafe1.csv",
        metadata_dir / "dish_metadata_cafe2.csv",
    ]
    targets: dict[str, DishData] = {}

    for csv_path in csv_paths:
        if not csv_path.exists():
            continue

        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                # official is headerless, but support with header
                if row[0] == "dish_id":
                    continue
                if len(row) < 6:
                    raise ValueError(
                        f"Expected at least 6 columns in {csv_path}, got {len(row)}: {row}"
                    )
                dish_id = row[0]
                targets[dish_id] = DishData(
                    total_calories=float(row[1]),
                    total_mass=float(row[2]),
                    total_fat=float(row[3]),
                    total_carb=float(row[4]),
                    total_protein=float(row[5]),
                )

    if not targets:
        raise FileNotFoundError(
            f"No Nutrition5K dish metadata could be loaded from {metadata_dir}"
        )

    return targets


def discover_split_files(dataset_root: Path) -> dict[str, Path]:
    split_dir = _resolve_existing(dataset_root / "dish_ids" / "splits")
    split_paths = list(split_dir.glob("*.txt"))
    if not split_paths:
        raise FileNotFoundError(f"No split files found under {split_dir}")

    found: dict[str, Path] = {}
    for path in split_paths:
        name = path.stem.lower()
        if "train" in name and "train" not in found:
            found["train"] = path
        elif "test" in name and "test" not in found:
            found["test"] = path

    missing = {"train", "test"} - set(found)
    if missing:
        raise FileNotFoundError(
            f"Could not infer split files for {sorted(missing)} under {split_dir}"
        )

    return found


def load_split_ids(dataset_root: Path) -> dict[str, list[str]]:
    split_files = discover_split_files(dataset_root)
    split_ids: dict[str, list[str]] = {}
    for split, path in split_files.items():
        with path.open("r", encoding="utf-8") as handle:
            ids = [line.strip() for line in handle if line.strip()]
        split_ids[split] = ids
    return split_ids


def _list_overhead_rgb_images(dataset_root: Path, dish_id: str) -> list[Path]:
    dish_dir = dataset_root / "imagery" / "realsense_overhead" / dish_id
    if not dish_dir.exists():
        return []

    candidates: list[Path] = []
    for path in dish_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if "rgb" not in path.name.lower():
            continue
        candidates.append(path)
    return sorted(candidates)


def _list_side_angle_frame_images(dataset_root: Path, dish_id: str) -> list[Path]:
    dish_dir = dataset_root / "imagery" / "side_angles" / dish_id
    if not dish_dir.exists():
        return []

    candidates: list[Path] = []
    for path in dish_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        candidates.append(path)
    return sorted(candidates)


def list_images_for_dish(
    dataset_root: Path,
    dish_id: str,
    image_source: str,
) -> list[Path]:
    if image_source == "overhead_rgb":
        return _list_overhead_rgb_images(dataset_root, dish_id)
    if image_source == "side_angles_frames":
        return _list_side_angle_frame_images(dataset_root, dish_id)
    raise ValueError(f"Unsupported image source: {image_source}")


def build_image_records(
    dataset_root: Path,
    image_source: str,
) -> dict[str, list[ImageRecord]]:
    split_ids = load_split_ids(dataset_root)
    dish_targets = load_dish_targets(dataset_root)
    records_by_split: dict[str, list[ImageRecord]] = defaultdict(list)

    for split, dish_ids in split_ids.items():
        missing_images = 0
        missing_targets = 0
        for dish_id in dish_ids:
            targets = dish_targets.get(dish_id)
            if targets is None:
                missing_targets += 1
                continue

            image_paths = list_images_for_dish(dataset_root, dish_id, image_source)
            if not image_paths:
                missing_images += 1
                continue

            for image_path in image_paths:
                records_by_split[split].append(
                    ImageRecord(
                        dish_id=dish_id,
                        image_path=image_path,
                        targets=targets,
                    )
                )

        LOGGER.info(
            "Prepared %s split with %d image records across %d dishes (%d missing images, %d missing targets).",
            split,
            len(records_by_split[split]),
            len(set(dish_ids)),
            missing_images,
            missing_targets,
        )

    return dict(records_by_split)


def group_records_by_dish(records: Iterable[ImageRecord]) -> dict[str, list[ImageRecord]]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.dish_id].append(record)
    return dict(grouped)
