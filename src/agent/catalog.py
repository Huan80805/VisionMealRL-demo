from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class MealTemplate:
    """A single meal template in the catalog."""

    name: str
    calories: float
    protein: float      # grams
    carbs: float        # grams
    fat: float          # grams
    embedding: np.ndarray  # latent visual embedding from CV module
    catalog_id: str = ""
    style: str = ""
    image_path: str = ""

    @property
    def nutrition(self) -> np.ndarray:
        """Returns [calories, protein, carbs, fat]."""
        return np.array([self.calories, self.protein, self.carbs, self.fat])


class MealCatalog:
    """Catalog of K meal templates with vectorised lookup matrices.

    Primary constructor takes a list of MealTemplate.
    Use ``load_dummy`` for test/dev catalogs and ``load_from_artifact``
    for the real catalog manifest/embedding pair.

    The pre-stacked ``embeddings_matrix`` and ``nutrition_matrix`` exist
    so baselines can replace per-action Python loops with a single matmul
    once the action space scales to 1000+ entries.
    """

    def __init__(self, meals: Sequence[MealTemplate]):
        if not meals:
            raise ValueError("MealCatalog requires at least one meal template")

        self.meals: list[MealTemplate] = list(meals)
        self.num_meals: int = len(self.meals)

        embeddings = np.stack([m.embedding for m in self.meals], axis=0).astype(np.float32)
        nutrition = np.stack([m.nutrition for m in self.meals], axis=0).astype(np.float32)

        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")

        self.embeddings_matrix: np.ndarray = embeddings   # (N, emb_dim)
        self.nutrition_matrix: np.ndarray = nutrition     # (N, 4)
        self.embedding_dim: int = embeddings.shape[1]

    # ------------------------------------------------------------------
    # Per-item accessors (kept for compatibility with existing call sites)
    # ------------------------------------------------------------------

    def get_nutrition(self, meal_idx: int, portion: float) -> np.ndarray:
        """Returns scaled [cal, protein, carbs, fat] for a meal at given portion."""
        return self.nutrition_matrix[meal_idx] * portion

    def get_embedding(self, meal_idx: int) -> np.ndarray:
        return self.embeddings_matrix[meal_idx]

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def load_dummy(
        cls,
        num_meals: int = 1000,
        embedding_dim: int = 512,
        seed: int = 42,
    ) -> "MealCatalog":
        """Synthesise a catalog with the same shape as the future real
        artifact, so downstream code (env, baselines, training, eval)
        can be developed and unit-tested today.

        Each entry is a (random unit-vector embedding, plausible nutrition
        macros) pair. ``num_meals`` defaults to 1000 to match the planned
        RecipeNLG action space.
        """
        rng = np.random.RandomState(seed)
        cal_ranges = [(200, 400), (300, 600), (400, 800)]
        meals: list[MealTemplate] = []
        for i in range(num_meals):
            lo, hi = cal_ranges[i % 3]
            cal = float(rng.uniform(lo, hi))
            protein = float(rng.uniform(5, 50))
            carbs = float(rng.uniform(10, 80))
            fat = float(rng.uniform(3, 40))
            emb = rng.randn(embedding_dim).astype(np.float32)
            emb /= np.linalg.norm(emb) + 1e-8
            meals.append(MealTemplate(
                name=f"dummy_meal_{i:04d}",
                calories=cal, protein=protein, carbs=carbs, fat=fat,
                embedding=emb,
            ))
        return cls(meals)

    @classmethod
    def load_from_artifact(
        cls,
        manifest_path: Path,
        embeddings_path: Path,
    ) -> "MealCatalog":
        """Load the real action catalog from a manifest and embedding array.

        Supported manifest columns:
          - name: ``recipe_name``, ``dish_name``, or ``name``
          - nutrition: either ``calories/protein/carbs/fat`` or
            ``total_calories/total_protein/total_carb/total_fat``
          - optional metadata: ``catalog_id``, ``style``, ``image_path``

        ``embeddings_path`` must be a 2D ``.npy`` array with one row per
        manifest row. Rows are L2-normalised on load for cosine reward and
        preference calculations.
        """
        manifest_path = Path(manifest_path)
        embeddings_path = Path(embeddings_path)

        rows = _load_manifest_csv(manifest_path)
        embeddings = np.load(embeddings_path).astype(np.float32)
        if embeddings.ndim != 2:
            raise ValueError(
                f"catalog embeddings must be 2D, got shape {embeddings.shape}"
            )
        if embeddings.shape[0] != len(rows):
            raise ValueError(
                "catalog embedding rows must match manifest rows: "
                f"{embeddings.shape[0]} != {len(rows)}"
            )

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms <= 0.0):
            raise ValueError("catalog embeddings contain zero-vector rows")
        embeddings = embeddings / norms

        meals: list[MealTemplate] = []
        for i, row in enumerate(rows):
            meal_id = _first_present(row, ("catalog_id", "dish_id", "id"), default=f"meal_{i}")
            name = _first_present(
                row,
                ("recipe_name", "dish_name", "name"),
                default=meal_id,
            )
            meals.append(MealTemplate(
                name=name,
                calories=_float_field(row, ("calories", "total_calories"), i),
                protein=_float_field(row, ("protein", "total_protein"), i),
                carbs=_float_field(row, ("carbs", "carb", "total_carb", "total_carbs"), i),
                fat=_float_field(row, ("fat", "total_fat"), i),
                embedding=embeddings[i],
                catalog_id=meal_id,
                style=_first_present(row, ("style", "cuisine"), default=""),
                image_path=_first_present(row, ("image_path", "image_paths"), default=""),
            ))
        return cls(meals)


def _load_manifest_csv(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() != ".csv":
        raise ValueError(f"catalog manifest must be CSV for now, got {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"catalog manifest has no rows: {path}")
    return rows


def _first_present(
    row: dict[str, str],
    columns: Sequence[str],
    default: str = "",
) -> str:
    for col in columns:
        value = row.get(col)
        if value not in (None, ""):
            return value
    return default


def _float_field(row: dict[str, str], columns: Sequence[str], row_idx: int) -> float:
    for col in columns:
        value = row.get(col)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError as exc:
                raise ValueError(
                    f"catalog row {row_idx} column {col} is not numeric: {value!r}"
                ) from exc
    raise ValueError(
        f"catalog row {row_idx} missing required numeric field; "
        f"expected one of {tuple(columns)}"
    )
