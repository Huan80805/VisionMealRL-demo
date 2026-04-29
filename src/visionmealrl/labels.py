from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from visionmealrl.nutrition5k import DishAnnotation


@dataclass(frozen=True)
class IngredientLabelConfig:
    labels: list[str]
    top_k: int
    min_mass: float
    min_fraction: float


def filtered_ingredient_names(
    annotation: DishAnnotation,
    min_mass: float,
    min_fraction: float,
) -> set[str]:
    dish_mass = max(annotation.targets.total_mass, 1e-6)
    names = set()
    for ingredient in annotation.ingredients:
        if ingredient.mass < min_mass:
            continue
        if ingredient.mass < min_fraction * dish_mass:
            continue
        names.add(ingredient.name)
    return names


def build_ingredient_vocabulary(
    dish_ids: Sequence[str],
    annotations: dict[str, DishAnnotation],
    top_k: int,
    min_mass: float,
    min_fraction: float,
) -> list[str]:
    counter: dict[str, int] = {}
    for dish_id in dish_ids:
        annotation = annotations.get(dish_id)
        if annotation is None:
            continue
        for name in filtered_ingredient_names(annotation, min_mass=min_mass, min_fraction=min_fraction):
            counter[name] = counter.get(name, 0) + 1

    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [name for name, _count in ranked[:top_k]]


def encode_multi_hot_ingredients(
    dish_ids: Sequence[str],
    annotations: dict[str, DishAnnotation],
    labels: Sequence[str],
    min_mass: float,
    min_fraction: float,
) -> np.ndarray:
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    targets = np.zeros((len(dish_ids), len(labels)), dtype=np.float32)

    for row_idx, dish_id in enumerate(dish_ids):
        annotation = annotations.get(dish_id)
        if annotation is None:
            continue
        for name in filtered_ingredient_names(annotation, min_mass=min_mass, min_fraction=min_fraction):
            label_idx = label_to_index.get(name)
            if label_idx is not None:
                targets[row_idx, label_idx] = 1.0

    return targets
