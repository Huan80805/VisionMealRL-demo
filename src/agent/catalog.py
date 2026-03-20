from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    """Manages K meal templates.

    In the real system these come from clustering Food-101 categories and
    averaging their Nutrition5k annotations via the extract-embeddings pipeline.
    For now, synthetic meals are generated.
    """

    def __init__(self, num_meals: int = 30, embedding_dim: int = 512, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.embedding_dim = embedding_dim
        self.meals = self._generate_synthetic_meals(num_meals)
        self.num_meals = len(self.meals)

    def _generate_synthetic_meals(self, n: int) -> list[MealTemplate]:
        """Generate synthetic meals spanning breakfast/lunch/dinner calorie ranges.

        TODO: Replace with real CLIP embeddings from the `extract-embeddings` pipeline.
        Once dish_embeddings.npy and dish_manifest.csv are available, call
        MealCatalog.load_from_file() instead of this method.
        """
        meals = []
        cal_ranges = [(200, 400), (300, 600), (400, 800)]
        for i in range(n):
            lo, hi = cal_ranges[i % 3]
            cal = self.rng.uniform(lo, hi)
            protein = self.rng.uniform(5, 50)
            carbs = self.rng.uniform(10, 80)
            fat = self.rng.uniform(3, 40)
            emb = self.rng.randn(self.embedding_dim).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)  # unit normalize
            meals.append(MealTemplate(
                name=f"meal_{i:03d}",
                calories=cal, protein=protein, carbs=carbs, fat=fat,
                embedding=emb,
            ))
        return meals

    def get_nutrition(self, meal_idx: int, portion: float) -> np.ndarray:
        """Returns scaled [cal, protein, carbs, fat] for a meal at given portion."""
        return self.meals[meal_idx].nutrition * portion

    def get_embedding(self, meal_idx: int) -> np.ndarray:
        return self.meals[meal_idx].embedding

    @classmethod
    def load_from_file(cls, npy_path: Path, manifest_csv: Path) -> "MealCatalog":
        """Load dish embeddings and nutrition data from extract-embeddings output.

        TODO: Load dish_embeddings.npy + dish_manifest.csv produced by
        visionmealrl extract-embeddings.  Expected formats:
          - npy_path:      float32 array of shape (N, embedding_dim),
                           L2-normalized CLIP embeddings (one row per dish).
          - manifest_csv:  CSV with columns dish_id, total_calories, total_mass,
                           total_fat, total_carb, total_protein
                           (see embedding.aggregate_dish_embeddings for schema).
        """
        raise NotImplementedError
