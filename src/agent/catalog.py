from __future__ import annotations

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

    @property
    def nutrition(self) -> np.ndarray:
        """Returns [calories, protein, carbs, fat]."""
        return np.array([self.calories, self.protein, self.carbs, self.fat])


class MealCatalog:
    """Catalog of K meal templates with vectorised lookup matrices.

    Primary constructor takes a list of MealTemplate.
    Use ``load_dummy`` for test/dev catalogs.
    TODO: fix ``load_from_artifact`` once the catalog artifact provided.

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
        """Real loader — TODO: wires to catalog artifact.

        Expected format:
          - manifest_path: CSV/JSON with one row per dish, columns
              ``dish_name, calories, protein, carbs, fat`` (optionally
              ``image_paths, mass``).
          - embeddings_path: float32 .npy of shape ``(N, embedding_dim)``,
              L2-normalised CLIP embeddings, rows aligned with the manifest.

        Once the format is frozen, parse both into a list of MealTemplate
        and call ``cls(meals)``. Until then, use ``load_dummy``.
        """
        raise NotImplementedError(
            f"load_from_artifact(manifest_path={manifest_path}, "
            f"embeddings_path={embeddings_path}) is awaiting the catalog format;"
            f"use MealCatalog.load_dummy(...) until then."
        )
