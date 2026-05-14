from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from agent.catalog import COMPONENT_NAMES

if TYPE_CHECKING:
    from agent.config import AgentConfig
    from agent.catalog import MealTemplate

PREFERENCE_COMPONENT_WEIGHTS = {
    "ingredient": 0.40,
    "cuisine": 0.35,
    "name": 0.25,
}
DEFAULT_PREFERENCE_NOISE = 0.05


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"preference component must be 1D, got shape {arr.shape}")
    if arr.size == 0:
        return arr
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32)


class SimulatedUser:
    """Simulates a user with daily/weekly nutrition targets and a preference embedding.

    The preference embedding drives meal recommendation quality (PA metric).
    In the real system it is derived from the user's historical meal choices
    or from a list of dietary-style-matched templates produced by the catalog filter.
    Preference is stored as ingredient, cuisine, and recipe-name components,
    with ``preference_embedding`` exposed as the concatenated model input.
    """

    def __init__(
        self,
        daily_cal: float = 2000.0,
        daily_protein: float = 80.0,
        daily_carbs: float = 250.0,
        daily_fat: float = 65.0,
        embedding_dim: int = 512,
        preference_embedding: Optional[np.ndarray] = None,
        ingredient_preference: Optional[np.ndarray] = None,
        cuisine_preference: Optional[np.ndarray] = None,
        name_preference: Optional[np.ndarray] = None,
        preference_noise: float = DEFAULT_PREFERENCE_NOISE,
        seed: int = 123,
    ):
        self.rng = np.random.RandomState(seed)
        self.daily_target = np.array(
            [daily_cal, daily_protein, daily_carbs, daily_fat], dtype=np.float64
        )
        self.weekly_target = self.daily_target * 7.0
        self.embedding_dim = embedding_dim
        self.preference_noise = preference_noise

        component_preferences = {
            "ingredient": ingredient_preference,
            "cuisine": cuisine_preference,
            "name": name_preference,
        }
        has_component_preferences = any(
            value is not None for value in component_preferences.values()
        )

        if has_component_preferences:
            self.ingredient_preference = _normalize(
                np.array([], dtype=np.float32)
                if ingredient_preference is None
                else ingredient_preference
            )
            self.cuisine_preference = _normalize(
                np.array([], dtype=np.float32)
                if cuisine_preference is None
                else cuisine_preference
            )
            self.name_preference = _normalize(
                np.array([], dtype=np.float32)
                if name_preference is None
                else name_preference
            )
            pref = self.combined_preference_embedding
            if pref.shape[0] != embedding_dim:
                raise ValueError(
                    f"component preference dim {pref.shape[0]} "
                    f"!= embedding_dim {embedding_dim}"
                )
        elif preference_embedding is None:
            # TODO: remove this once we have a real preference signal from meal history or template matching
            pref = self.rng.randn(embedding_dim).astype(np.float32)
            pref = pref / (np.linalg.norm(pref) + 1e-8)
            self.ingredient_preference = pref
            self.cuisine_preference = np.array([], dtype=np.float32)
            self.name_preference = np.array([], dtype=np.float32)
        else:
            pref = np.asarray(preference_embedding, dtype=np.float32)
            if pref.ndim != 1:
                raise ValueError(
                    f"preference_embedding must be 1D, got shape {pref.shape}"
                )
            if pref.shape[0] != embedding_dim:
                raise ValueError(
                    f"preference_embedding dim {pref.shape[0]} "
                    f"!= embedding_dim {embedding_dim}"
                )
            pref = _normalize(pref)
            self.ingredient_preference = pref
            self.cuisine_preference = np.array([], dtype=np.float32)
            self.name_preference = np.array([], dtype=np.float32)

        self.preference_embedding = self.combined_preference_embedding

    @property
    def preference_components(self) -> dict[str, np.ndarray]:
        return {
            "ingredient": self.ingredient_preference,
            "cuisine": self.cuisine_preference,
            "name": self.name_preference,
        }

    @property
    def combined_preference_embedding(self) -> np.ndarray:
        combined = np.concatenate([
            self.ingredient_preference,
            self.cuisine_preference,
            self.name_preference,
        ]).astype(np.float32)
        return _normalize(combined)

    def set_preference_components(
        self,
        ingredient: np.ndarray,
        cuisine: np.ndarray,
        name: np.ndarray,
    ) -> None:
        self.ingredient_preference = _normalize(ingredient)
        self.cuisine_preference = _normalize(cuisine)
        self.name_preference = _normalize(name)
        self.preference_embedding = self.combined_preference_embedding
        self.embedding_dim = int(self.preference_embedding.shape[0])

    def preference_score(self, meal_embedding: np.ndarray) -> float:
        """Cosine similarity between user preference and meal embedding, with noise."""
        sim = float(np.dot(self.preference_embedding, meal_embedding))
        return sim + float(self.rng.normal(0, self.preference_noise))

    def preference_component_scores(
        self,
        meal_components: dict[str, np.ndarray],
        weights: dict[str, float] | None = None,
        add_noise: bool = True,
    ) -> tuple[float, dict[str, float]]:
        weights = weights or PREFERENCE_COMPONENT_WEIGHTS
        scores: dict[str, float] = {}
        weighted = 0.0
        active_weight = 0.0
        for component in COMPONENT_NAMES:
            user_vec = self.preference_components[component]
            meal_vec = meal_components[component]
            if user_vec.size == 0 or meal_vec.size == 0:
                scores[component] = 0.0
                continue
            score = float(np.dot(user_vec, meal_vec))
            scores[component] = score
            weight = float(weights.get(component, 0.0))
            weighted += weight * score
            active_weight += weight
        if active_weight > 0:
            weighted /= active_weight
        if add_noise:
            weighted += float(self.rng.normal(0, self.preference_noise))
        return float(weighted), scores

    @classmethod
    def from_config(cls, cfg: "AgentConfig", seed_offset: int = 1) -> "SimulatedUser":
        """Construct a SimulatedUser from an AgentConfig with a random preference."""
        return cls(
            daily_cal=cfg.daily_cal,
            daily_protein=cfg.daily_protein,
            daily_carbs=cfg.daily_carbs,
            daily_fat=cfg.daily_fat,
            embedding_dim=cfg.embedding_dim,
            seed=cfg.seed + seed_offset,
        )

    @classmethod
    def from_meal_history(
        cls,
        meal_embeddings: np.ndarray,
        daily_cal: float = 2000.0,
        daily_protein: float = 80.0,
        daily_carbs: float = 250.0,
        daily_fat: float = 65.0,
        preference_noise: float = DEFAULT_PREFERENCE_NOISE,
        seed: int = 123,
    ) -> "SimulatedUser":
        """Build a user whose preference is the L2-normalised mean of the
        rows of ``meal_embeddings``.

        Each row is a meal representation of one of the user's past meals
        (typically already L2-normalised by the extraction pipeline).
        """
        meal_embeddings = np.asarray(meal_embeddings, dtype=np.float32)
        if meal_embeddings.ndim != 2:
            raise ValueError(
                f"meal_embeddings must be 2D, got shape {meal_embeddings.shape}"
            )
        if meal_embeddings.shape[0] == 0:
            raise ValueError("meal_embeddings must contain at least one row")

        pref = meal_embeddings.mean(axis=0)
        pref = pref / (np.linalg.norm(pref) + 1e-8)
        return cls(
            daily_cal=daily_cal,
            daily_protein=daily_protein,
            daily_carbs=daily_carbs,
            daily_fat=daily_fat,
            embedding_dim=int(meal_embeddings.shape[1]),
            preference_embedding=pref,
            preference_noise=preference_noise,
            seed=seed,
        )

    @classmethod
    def from_templates(
        cls,
        templates: Sequence["MealTemplate"],
        daily_cal: float = 2000.0,
        daily_protein: float = 80.0,
        daily_carbs: float = 250.0,
        daily_fat: float = 65.0,
        preference_noise: float = DEFAULT_PREFERENCE_NOISE,
        seed: int = 123,
    ) -> "SimulatedUser":
        """
        Build a user whose preference is the L2-normalised mean of the embeddings of ``templates``
        """
        if len(templates) == 0:
            raise ValueError("templates must contain at least one MealTemplate")
        component_means = {}
        for component in COMPONENT_NAMES:
            rows = np.stack(
                [getattr(t, f"{component}_embedding") for t in templates],
                axis=0,
            )
            component_means[component] = rows.mean(axis=0)

        return cls(
            daily_cal=daily_cal,
            daily_protein=daily_protein,
            daily_carbs=daily_carbs,
            daily_fat=daily_fat,
            embedding_dim=int(sum(v.shape[0] for v in component_means.values())),
            ingredient_preference=component_means["ingredient"],
            cuisine_preference=component_means["cuisine"],
            name_preference=component_means["name"],
            preference_noise=preference_noise,
            seed=seed,
        )
