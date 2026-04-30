from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from agent.config import AgentConfig
    from agent.catalog import MealTemplate


class SimulatedUser:
    """Simulates a user with daily/weekly nutrition targets and a preference embedding.

    The preference embedding drives meal recommendation quality (PA metric).
    In the real system it is derived from the user's historical meal choices
    or from a list of dietary-style-matched templates produced by the catalog filter
    """

    def __init__(
        self,
        daily_cal: float = 2000.0,
        daily_protein: float = 80.0,
        daily_carbs: float = 250.0,
        daily_fat: float = 65.0,
        embedding_dim: int = 512,
        preference_embedding: Optional[np.ndarray] = None,
        preference_noise: float = 0.2,
        seed: int = 123,
    ):
        self.rng = np.random.RandomState(seed)
        self.daily_target = np.array(
            [daily_cal, daily_protein, daily_carbs, daily_fat], dtype=np.float64
        )
        self.weekly_target = self.daily_target * 7.0
        self.embedding_dim = embedding_dim
        self.preference_noise = preference_noise

        if preference_embedding is None:
            # TODO: remove this once we have a real preference signal from meal history or template matching
            pref = self.rng.randn(embedding_dim).astype(np.float32)
            pref = pref / (np.linalg.norm(pref) + 1e-8)
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

        self.preference_embedding = pref

    def preference_score(self, meal_embedding: np.ndarray) -> float:
        """Cosine similarity between user preference and meal embedding, with noise."""
        sim = float(np.dot(self.preference_embedding, meal_embedding))
        return sim + float(self.rng.normal(0, self.preference_noise))

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
        preference_noise: float = 0.2,
        seed: int = 123,
    ) -> "SimulatedUser":
        """Build a user whose preference is the L2-normalised mean of the
        rows of ``meal_embeddings``.

        Each row is a CLIP embedding of one of the user's past meals
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
        preference_noise: float = 0.2,
        seed: int = 123,
    ) -> "SimulatedUser":
        """
        Build a user whose preference is the L2-normalised mean of the embeddings of ``templates``
        """
        if len(templates) == 0:
            raise ValueError("templates must contain at least one MealTemplate")
        embeddings = np.stack([t.embedding for t in templates], axis=0)
        return cls.from_meal_history(
            meal_embeddings=embeddings,
            daily_cal=daily_cal,
            daily_protein=daily_protein,
            daily_carbs=daily_carbs,
            daily_fat=daily_fat,
            preference_noise=preference_noise,
            seed=seed,
        )
