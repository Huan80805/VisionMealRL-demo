from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence, TypedDict, Union

import numpy as np

from visionmealrl.constants import TARGET_COLUMNS

ManifestValue = Union[str, int, float]
NutritionMetadata = dict[str, ManifestValue]


class DishEmbeddingRecord(TypedDict):
    embedding: np.ndarray
    nutrition_metadata: NutritionMetadata


def parse_manifest_value(value: str) -> ManifestValue:
    if value == "":
        return value
    try:
        parsed_float = float(value)
    except ValueError:
        return value

    if parsed_float.is_integer():
        return int(parsed_float)
    return parsed_float


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_split_manifest(path: Path, split_name: str, dish_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "dish_id"])
        writer.writeheader()
        for dish_id in dish_ids:
            writer.writerow({"split": split_name, "dish_id": dish_id})


def load_dish_embedding_lookup(split_dir: Path | str) -> dict[str, DishEmbeddingRecord]:
    # Returns { dish_id: { embedding,  nutritions }}
    # embeddings and nutrition data are read from artifacts/embeddings/<model>/overhead_rgb/<train/test>
    # embeddings: read from dish_embeddings.npy
    # nutritions: read from dish_manifest.csv
    split_path = Path(split_dir)
    embeddings_path = split_path / "dish_embeddings.npy"
    manifest_path = split_path / "dish_manifest.csv"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Expected dish embeddings at {embeddings_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Expected dish manifest at {manifest_path}")

    embeddings = np.load(embeddings_path)
    manifest_rows = load_manifest_rows(manifest_path)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected dish_embeddings.npy to be a 2D array, got shape {embeddings.shape}"
        )
    if embeddings.shape[0] != len(manifest_rows):
        raise ValueError(
            "dish_embeddings.npy row count does not match dish_manifest.csv row count: "
            f"{embeddings.shape[0]} != {len(manifest_rows)}"
        )

    lookup: dict[str, DishEmbeddingRecord] = {}
    for row_index, (embedding, manifest_row) in enumerate(zip(embeddings, manifest_rows)):
        dish_id = manifest_row.get("dish_id", "")
        if not dish_id:
            raise ValueError(f"Missing dish_id in manifest row {row_index}")
        if dish_id in lookup:
            raise ValueError(f"Duplicate dish_id in manifest: {dish_id}")

        nutrition_metadata = {
            column: parse_manifest_value(manifest_row[column])
            for column in TARGET_COLUMNS
            if column in manifest_row and manifest_row[column] is not None
        }
        lookup[dish_id] = {
            "embedding": embedding.astype(np.float32, copy=False),
            "nutrition_metadata": nutrition_metadata,
        }

    return lookup
