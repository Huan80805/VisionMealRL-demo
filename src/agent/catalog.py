from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

COMPONENT_NAMES = ("ingredient", "cuisine", "name")
DEFAULT_SLOT_COUNT = 3
SNACK_SLOT_PENALTY = -0.3
SLOT_MISMATCH_PENALTY = -1.0


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"embedding component must be 1D, got shape {arr.shape}")
    if arr.size == 0:
        return arr
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def _split_dims(total_dim: int) -> tuple[int, int, int]:
    if total_dim < 3:
        return total_dim, 0, 0
    ingredient_dim = max(1, total_dim // 2)
    cuisine_dim = max(1, (total_dim - ingredient_dim) // 2)
    name_dim = total_dim - ingredient_dim - cuisine_dim
    return ingredient_dim, cuisine_dim, name_dim


@dataclass
class MealTemplate:
    """A single meal template in the catalog."""

    name: str
    calories: float
    protein: float      # grams
    carbs: float        # grams
    fat: float          # grams
    embedding: np.ndarray | None = None
    catalog_id: str = ""
    style: str = ""
    image_path: str = ""
    meal_type: str = ""
    dish_type: str = ""
    ingredient_embedding: np.ndarray | None = None
    cuisine_embedding: np.ndarray | None = None
    name_embedding: np.ndarray | None = None
    valid_slots: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.ingredient_embedding is None:
            if self.embedding is None:
                raise ValueError("MealTemplate requires embedding components or embedding")
            self.ingredient_embedding = _normalize_vector(self.embedding)
        else:
            self.ingredient_embedding = _normalize_vector(self.ingredient_embedding)

        self.cuisine_embedding = _normalize_vector(
            np.array([], dtype=np.float32)
            if self.cuisine_embedding is None
            else self.cuisine_embedding
        )
        self.name_embedding = _normalize_vector(
            np.array([], dtype=np.float32)
            if self.name_embedding is None
            else self.name_embedding
        )
        self.embedding = self.concatenated_embedding
        if not self.valid_slots:
            self.valid_slots = valid_slots_from_meal_type(self.meal_type)

    @property
    def nutrition(self) -> np.ndarray:
        """Returns [calories, protein, carbs, fat]."""
        return np.array([self.calories, self.protein, self.carbs, self.fat])

    @property
    def components(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.ingredient_embedding,
            self.cuisine_embedding,
            self.name_embedding,
        )

    @property
    def concatenated_embedding(self) -> np.ndarray:
        return np.concatenate(self.components).astype(np.float32)


class MealCatalog:
    """Catalog of K meal templates with vectorised lookup matrices.

    Primary constructor takes a list of MealTemplate.
    Use ``load_dummy`` for test/dev catalogs and ``load_from_artifact``
    for the real catalog manifest/embedding artifacts.

    The pre-stacked ``embeddings_matrix`` and ``nutrition_matrix`` exist
    so baselines can replace per-action Python loops with a single matmul
    once the action space scales to 1000+ entries.
    """

    def __init__(self, meals: Sequence[MealTemplate]):
        if not meals:
            raise ValueError("MealCatalog requires at least one meal template")

        self.meals: list[MealTemplate] = list(meals)
        self.num_meals: int = len(self.meals)

        self.ingredient_matrix = _stack_components(self.meals, "ingredient_embedding")
        self.cuisine_matrix = _stack_components(self.meals, "cuisine_embedding")
        self.name_matrix = _stack_components(self.meals, "name_embedding")
        embeddings = np.concatenate(
            [self.ingredient_matrix, self.cuisine_matrix, self.name_matrix],
            axis=1,
        ).astype(np.float32)
        nutrition = np.stack([m.nutrition for m in self.meals], axis=0).astype(np.float32)

        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")

        self.embeddings_matrix: np.ndarray = embeddings   # (N, emb_dim)
        self.nutrition_matrix: np.ndarray = nutrition     # (N, 4)
        self.embedding_dim: int = embeddings.shape[1]
        self.component_dims = {
            "ingredient": self.ingredient_matrix.shape[1],
            "cuisine": self.cuisine_matrix.shape[1],
            "name": self.name_matrix.shape[1],
        }
        start = 0
        self.component_slices: dict[str, slice] = {}
        for name in COMPONENT_NAMES:
            end = start + self.component_dims[name]
            self.component_slices[name] = slice(start, end)
            start = end

    # ------------------------------------------------------------------
    # Per-item accessors
    # ------------------------------------------------------------------

    def get_nutrition(self, meal_idx: int, portion: float) -> np.ndarray:
        """Returns scaled [cal, protein, carbs, fat] for a meal at given portion."""
        return self.nutrition_matrix[meal_idx] * portion

    def get_embedding(self, meal_idx: int) -> np.ndarray:
        return self.embeddings_matrix[meal_idx]

    def get_component(self, meal_idx: int, component: str) -> np.ndarray:
        if component == "ingredient":
            return self.ingredient_matrix[meal_idx]
        if component == "cuisine":
            return self.cuisine_matrix[meal_idx]
        if component == "name":
            return self.name_matrix[meal_idx]
        raise KeyError(f"unknown component {component!r}")

    def get_components(self, meal_idx: int) -> dict[str, np.ndarray]:
        return {name: self.get_component(meal_idx, name) for name in COMPONENT_NAMES}

    def split_embedding(self, embedding: np.ndarray) -> dict[str, np.ndarray]:
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.shape != (self.embedding_dim,):
            raise ValueError(
                f"embedding shape {emb.shape}; expected ({self.embedding_dim},)"
            )
        return {
            name: emb[self.component_slices[name]]
            for name in COMPONENT_NAMES
        }

    def slot_score(self, meal_idx: int, slot: int) -> float:
        meal = self.meals[meal_idx]
        if slot in meal.valid_slots:
            return 0.0
        if _contains_token(meal.meal_type, "snack"):
            return SNACK_SLOT_PENALTY
        return SLOT_MISMATCH_PENALTY

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
        ingredient_dim, cuisine_dim, name_dim = _split_dims(embedding_dim)
        meals: list[MealTemplate] = []
        for i in range(num_meals):
            lo, hi = cal_ranges[i % 3]
            cal = float(rng.uniform(lo, hi))
            protein = float(rng.uniform(5, 50))
            carbs = float(rng.uniform(10, 80))
            fat = float(rng.uniform(3, 40))
            ingredient_emb = _normalize_vector(rng.randn(ingredient_dim).astype(np.float32))
            cuisine_emb = _normalize_vector(rng.randn(cuisine_dim).astype(np.float32))
            name_emb = _normalize_vector(rng.randn(name_dim).astype(np.float32))
            meal_type = ("breakfast", "lunch/dinner", "snack")[i % 3]
            meals.append(MealTemplate(
                name=f"dummy_meal_{i:04d}",
                calories=cal, protein=protein, carbs=carbs, fat=fat,
                ingredient_embedding=ingredient_emb,
                cuisine_embedding=cuisine_emb,
                name_embedding=name_emb,
                meal_type=meal_type,
                dish_type="main course",
            ))
        return cls(meals)

    @classmethod
    def load_from_artifact(
        cls,
        manifest_path: Path,
        ingredient_embeddings_path: Path | None = None,
        cuisine_embeddings_path: Path | None = None,
        name_embeddings_path: Path | None = None,
    ) -> "MealCatalog":
        """Load the real action catalog from manifest and representation arrays.

        Supported manifest columns:
          - name: ``recipe_name``, ``dish_name``, or ``name``
          - nutrition: either ``calories/protein/carbs/fat`` or
            ``total_calories/total_protein/total_carb/total_fat``
          - optional metadata: ``catalog_id``, ``style``, ``image_path``,
            ``meal_type``, ``dish_type``

        Required artifacts are ``ingredient_embeddings.npy``,
        ``cuisine_embeddings.npy``, and ``name_embeddings.npy``.
        """
        manifest_path = Path(manifest_path)

        rows = _load_manifest_csv(manifest_path)
        artifact_dir = manifest_path.parent
        ingredient_path = (
            Path(ingredient_embeddings_path)
            if ingredient_embeddings_path is not None
            else artifact_dir / "ingredient_embeddings.npy"
        )
        cuisine_path = (
            Path(cuisine_embeddings_path)
            if cuisine_embeddings_path is not None
            else artifact_dir / "cuisine_embeddings.npy"
        )
        name_path = (
            Path(name_embeddings_path)
            if name_embeddings_path is not None
            else artifact_dir / "name_embeddings.npy"
        )

        missing = [
            str(path)
            for path in (ingredient_path, cuisine_path, name_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "three-component catalog is incomplete; missing "
                + ", ".join(missing)
            )

        ingredient_embeddings = _load_component_matrix(ingredient_path, len(rows))
        cuisine_embeddings = _load_component_matrix(cuisine_path, len(rows))
        name_embeddings = _load_component_matrix(name_path, len(rows))

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
                ingredient_embedding=ingredient_embeddings[i],
                cuisine_embedding=cuisine_embeddings[i],
                name_embedding=name_embeddings[i],
                catalog_id=meal_id,
                style=_first_present(row, ("style", "cuisine"), default=""),
                image_path=_first_present(row, ("image_path", "image_paths"), default=""),
                meal_type=_first_present(row, ("meal_type",), default=""),
                dish_type=_first_present(row, ("dish_type",), default=""),
            ))
        return cls(meals)


def _stack_components(meals: Sequence[MealTemplate], attr: str) -> np.ndarray:
    matrix = np.stack([getattr(m, attr) for m in meals], axis=0).astype(np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{attr} must stack to 2D, got shape {matrix.shape}")
    return matrix


def _load_component_matrix(path: Path, expected_rows: int) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{path} must be 2D, got shape {arr.shape}")
    if arr.shape[0] != expected_rows:
        raise ValueError(
            f"{path} rows must match manifest rows: {arr.shape[0]} != {expected_rows}"
        )
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError(f"{path} contains zero-vector rows")
    return (arr / norms).astype(np.float32)


def valid_slots_from_meal_type(meal_type: str) -> tuple[int, ...]:
    if _contains_token(meal_type, "breakfast"):
        return (0,)
    if _contains_token(meal_type, "lunch/dinner"):
        return (1, 2)
    if _contains_token(meal_type, "lunch") or _contains_token(meal_type, "dinner"):
        return (1, 2)
    if _contains_token(meal_type, "snack"):
        return ()
    return tuple(range(DEFAULT_SLOT_COUNT))


def _contains_token(value: str, token: str) -> bool:
    return token in _parse_listish(value)


def _parse_listish(value: str) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, str):
        return tuple(str(v).strip().lower() for v in value)
    text = value.strip()
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = text
    if isinstance(parsed, str):
        return (parsed.strip().lower(),)
    if isinstance(parsed, (list, tuple, set)):
        return tuple(str(v).strip().lower() for v in parsed)
    return (str(parsed).strip().lower(),)


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
