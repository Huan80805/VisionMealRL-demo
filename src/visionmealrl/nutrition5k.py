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


@dataclass(frozen=True)
class IngredientData:
    ingredient_id: str
    name: str
    mass: float


@dataclass(frozen=True)
class DishAnnotation:
    dish_id: str
    targets: DishData
    ingredients: tuple[IngredientData, ...]


@dataclass(frozen=True)
class DishRecord:
    dish_id: str
    image_paths: tuple[Path, ...]
    targets: DishData
    ingredients: tuple[IngredientData, ...]


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


def _metadata_csv_paths(dataset_root: Path) -> list[Path]:
    metadata_dir = _resolve_existing(dataset_root / "metadata")
    return [
        metadata_dir / "dish_metadata_cafe1.csv",
        metadata_dir / "dish_metadata_cafe2.csv",
    ]


def _parse_ingredient_columns(row: list[str]) -> tuple[IngredientData, ...]:
    ingredients: list[IngredientData] = []
    for start in range(6, len(row), 7):
        chunk = row[start : start + 7]
        if len(chunk) < 7:
            continue
        ingredient_id, name, mass = chunk[0], chunk[1], chunk[2]
        ingredients.append(
            IngredientData(
                ingredient_id=ingredient_id,
                name=name,
                mass=float(mass),
            )
        )
    return tuple(ingredients)


def load_dish_annotations(dataset_root: Path) -> dict[str, DishAnnotation]:
    annotations: dict[str, DishAnnotation] = {}
    csv_paths = _metadata_csv_paths(dataset_root)
    metadata_dir = _resolve_existing(dataset_root / "metadata")
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
                annotations[dish_id] = DishAnnotation(
                    dish_id=dish_id,
                    targets=DishData(
                        total_calories=float(row[1]),
                        total_mass=float(row[2]),
                        total_fat=float(row[3]),
                        total_carb=float(row[4]),
                        total_protein=float(row[5]),
                    ),
                    ingredients=_parse_ingredient_columns(row),
                )

    if not annotations:
        raise FileNotFoundError(
            f"No Nutrition5K dish metadata could be loaded from {metadata_dir}"
        )

    return annotations


def load_dish_targets(dataset_root: Path) -> dict[str, DishData]:
    annotations = load_dish_annotations(dataset_root)
    return {dish_id: annotation.targets for dish_id, annotation in annotations.items()}


def split_prefix_for_image_source(image_source: str) -> str:
    if image_source in {"overhead_rgb", "side_angles_frames"}:
        return "rgb"
    raise ValueError(f"Unsupported image source: {image_source}")


def discover_split_files(dataset_root: Path, image_source: str) -> dict[str, Path]:
    split_dir = _resolve_existing(dataset_root / "dish_ids" / "splits")
    prefix = split_prefix_for_image_source(image_source)
    split_files = {
        "train": split_dir / f"{prefix}_train_ids.txt",
        "test": split_dir / f"{prefix}_test_ids.txt",
    }
    for split, path in split_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Expected {split} split file for image source {image_source} at {path}"
            )
    return split_files


def load_split_ids(dataset_root: Path, image_source: str) -> dict[str, list[str]]:
    split_files = discover_split_files(dataset_root, image_source)
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
    dish_records_by_split = build_dish_records_by_split(dataset_root, image_source)
    return {
        split: image_records_from_dish_records(dish_records)
        for split, dish_records in dish_records_by_split.items()
    }


def image_records_from_dish_records(dish_records: Iterable[DishRecord]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for dish_record in dish_records:
        for image_path in dish_record.image_paths:
            records.append(
                ImageRecord(
                    dish_id=dish_record.dish_id,
                    image_path=image_path,
                    targets=dish_record.targets,
                )
            )
    return records


def _build_dish_records_for_ids(
    dish_ids: Iterable[str],
    annotations: dict[str, DishAnnotation],
    dataset_root: Path,
    image_source: str,
) -> tuple[list[DishRecord], int, int]:
    records: list[DishRecord] = []
    missing_images = 0
    missing_targets = 0

    for dish_id in dish_ids:
        annotation = annotations.get(dish_id)
        if annotation is None:
            missing_targets += 1
            continue

        image_paths = tuple(list_images_for_dish(dataset_root, dish_id, image_source))
        if not image_paths:
            missing_images += 1
            continue

        records.append(
            DishRecord(
                dish_id=dish_id,
                image_paths=image_paths,
                targets=annotation.targets,
                ingredients=annotation.ingredients,
            )
        )

    return records, missing_images, missing_targets


def build_dish_records(
    dataset_root: Path,
    dish_ids: Iterable[str],
    image_source: str,
) -> list[DishRecord]:
    annotations = load_dish_annotations(dataset_root)
    records, missing_images, missing_targets = _build_dish_records_for_ids(
        dish_ids=dish_ids,
        annotations=annotations,
        dataset_root=dataset_root,
        image_source=image_source,
    )

    LOGGER.info(
        "Prepared %d dish records (%d missing images, %d missing targets).",
        len(records),
        missing_images,
        missing_targets,
    )
    return records


def build_dish_records_by_split(
    dataset_root: Path,
    image_source: str,
) -> dict[str, list[DishRecord]]:
    split_ids = load_split_ids(dataset_root, image_source)
    annotations = load_dish_annotations(dataset_root)
    records_by_split: dict[str, list[DishRecord]] = {}

    for split, dish_ids in split_ids.items():
        split_records, missing_images, missing_targets = _build_dish_records_for_ids(
            dish_ids=dish_ids,
            annotations=annotations,
            dataset_root=dataset_root,
            image_source=image_source,
        )
        records_by_split[split] = split_records

        LOGGER.info(
            "Prepared %s split with %d dish records across %d dishes (%d missing images, %d missing targets).",
            split,
            len(split_records),
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
