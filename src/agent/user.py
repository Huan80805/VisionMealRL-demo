from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agent.config import AgentConfig


class SimulatedUser:
    """Simulates a user with daily/weekly nutrition targets and a preference embedding.

    The preference embedding drives meal recommendation quality (PA metric).
    In the real system it is derived from the user's historical meal choices
    passed through the CV module's CLIP encoder.
    """

    def __init__(
        self,
        daily_cal: float = 2000.0,
        daily_protein: float = 80.0,
        daily_carbs: float = 250.0,
        daily_fat: float = 65.0,
        embedding_dim: int = 512,
        preference_noise: float = 0.2,
        seed: int = 123,
    ):
        self.rng = np.random.RandomState(seed)
        self.daily_target = np.array([daily_cal, daily_protein, daily_carbs, daily_fat])
        self.weekly_target = self.daily_target * 7.0
        self.embedding_dim = embedding_dim
        self.preference_noise = preference_noise

        # TODO: Initialize from user's historical meal choices.
        # Real version: mean-pool + L2-normalize CLIP embeddings of user's meal history
        # (see SimulatedUser.from_meal_history).
        pref = self.rng.randn(embedding_dim).astype(np.float32)
        self.preference_embedding = pref / (np.linalg.norm(pref) + 1e-8)

    def preference_score(self, meal_embedding: np.ndarray) -> float:
        """Cosine similarity between user preference and meal embedding, with noise."""
        sim = np.dot(self.preference_embedding, meal_embedding)
        return float(sim + self.rng.normal(0, self.preference_noise))

    @classmethod
    def from_config(cls, cfg: "AgentConfig", seed_offset: int = 1) -> "SimulatedUser":
        """Construct a SimulatedUser from an AgentConfig."""
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
        """Initialize user preference from historical CLIP meal embeddings.

        TODO: Mean-pool + L2-normalize the rows of meal_embeddings to form
        preference_embedding.  Each row should already be L2-normalized (unit
        vector) as produced by the extract-embeddings pipeline.
        """
        raise NotImplementedError
