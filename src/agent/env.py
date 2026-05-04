from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from agent.catalog import MealCatalog
from agent.user import SimulatedUser

if TYPE_CHECKING:
    from agent.config import AgentConfig

EpisodeResampler = Callable[[SimulatedUser, np.random.Generator], None]


def estimate_observed_meal_from_photo(
    photo_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Placeholder for the future CV photo-feedback API.

    The CV/integration layer should replace this with a call that returns:
      - nutrition: shape (4,), [calories, protein, carbs, fat]
      - embedding: shape (embedding_dim,), in the same CLIP space as the catalog

    The agent environment can already consume those arrays through
    ``MealPlanningEnv.step(..., observed_nutrition=..., observed_embedding=...)``.
    """
    raise NotImplementedError(
        "Photo-to-observed-meal inference is owned by the CV integration path. "
        f"Expected to process photo_path={photo_path!s} and return "
        "(nutrition[4], embedding[embedding_dim])."
    )


class MealPlanningEnv(gym.Env):
    """Gymnasium environment for meal planning over a multi-day horizon.

    Episode = num_days × meals_per_day steps.
    At each step the agent picks (meal_template, portion_level).

    Observation layout (obs_dim = 13 + meals_per_day + 2 × embedding_dim):
      [0:4]                         daily_deficit  (normalized by daily_target)
      [4:8]                         episode_deficit (normalized by episode target)
      [8:12]                        daily_target   (scaled)
      [12:13]                       remaining_steps_fraction
      [13 : 13+mpd]                 time slot one-hot  (mpd = meals_per_day)
      [13+mpd : 13+mpd+emb]         mean recent meal embedding
      [13+mpd+emb : 13+mpd+2*emb]  user preference embedding

    Action: integer in [0, num_meals × len(portion_levels)).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        catalog: MealCatalog,
        user: SimulatedUser,
        num_days: int = 1,
        meals_per_day: int = 3,
        history_len: int = 6,
        portion_levels: tuple[float, ...] = (0.75, 1.0, 1.25),
        w_health: float = 1.0,
        w_diversity: float = 0.3,
        w_preference: float = 0.2,
        w_boundary: float = 0.5,
        bootstrap_pool: Optional[tuple[np.ndarray, np.ndarray]] = None,
        episode_resampler: Optional[EpisodeResampler] = None,
    ):
        super().__init__()
        self.catalog = catalog
        self.user = user
        # optional callable that mutates self.user at each reset
        # (random targets / fresh preference for training; no-op for eval).
        self.episode_resampler = episode_resampler
        self.num_days = num_days
        self.meals_per_day = meals_per_day
        self.horizon = num_days * meals_per_day
        self.history_len = history_len
        self.portion_levels = np.array(portion_levels)
        self.num_portions = len(portion_levels)
        self.emb_dim = catalog.embedding_dim

        self.w_health = w_health
        self.w_diversity = w_diversity
        self.w_preference = w_preference
        self.w_boundary = w_boundary
        self.target_scale = np.array(
            [3200.0, 200.0, 450.0, 100.0],
            dtype=np.float32,
        )

        self.num_actions = catalog.num_meals * self.num_portions
        self.action_space = spaces.Discrete(self.num_actions)

        obs_dim = (
            4               # daily deficit
            + 4             # episode deficit
            + 4             # daily target
            + 1             # remaining steps fraction
            + meals_per_day # time slot one-hot
            + self.emb_dim  # mean-pooled recent embeddings
            + self.emb_dim  # user preference
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # User meal-history pool sampled at reset to (a) warm up the
        # diversity context window with non-zero recent embeddings and
        # (b) pre-consume part of the weekly nutrition budget — i.e.
        # simulate a user starting the planning episode mid-week with
        # some meals already eaten. Two row-aligned arrays:
        #   embeddings: (N, emb_dim)
        #   nutrition:  (N, 4) in [cal, protein, carbs, fat]
        #
        # This pool is prior user history and should come from the
        # Nutrition5K-derived history artifact, not the recipe catalog.
        # The catalog is only the action space for future recommendations.
        # If omitted, the episode starts with no pre-consumed history.
        if bootstrap_pool is None:
            self.bootstrap_embeddings = np.empty((0, self.emb_dim), dtype=np.float32)
            self.bootstrap_nutrition = np.empty((0, 4), dtype=np.float32)
        else:
            emb_pool, nut_pool = bootstrap_pool
            emb_pool = np.asarray(emb_pool, dtype=np.float32)
            nut_pool = np.asarray(nut_pool, dtype=np.float32)
            if emb_pool.ndim != 2 or emb_pool.shape[1] != self.emb_dim:
                raise ValueError(
                    f"bootstrap embeddings shape {emb_pool.shape} incompatible "
                    f"with embedding_dim {self.emb_dim}"
                )
            if nut_pool.ndim != 2 or nut_pool.shape[1] != 4:
                raise ValueError(
                    f"bootstrap nutrition shape {nut_pool.shape}; expected (N, 4)"
                )
            if emb_pool.shape[0] != nut_pool.shape[0]:
                raise ValueError(
                    f"bootstrap embedding rows ({emb_pool.shape[0]}) "
                    f"!= nutrition rows ({nut_pool.shape[0]})"
                )
            self.bootstrap_embeddings = emb_pool
            self.bootstrap_nutrition = nut_pool

        # Internal state (initialized in reset)
        self._step_count = 0
        self._daily_deficit = np.zeros(4)
        self._weekly_deficit = np.zeros(4)
        self._episode_deficit = np.zeros(4)
        self._recent_embeddings: list[np.ndarray] = []

    @classmethod
    def from_config(
        cls,
        cfg: "AgentConfig",
        catalog: MealCatalog,
        user: SimulatedUser,
        bootstrap_pool: Optional[tuple[np.ndarray, np.ndarray]] = None,
        episode_resampler: Optional[EpisodeResampler] = None,
    ) -> "MealPlanningEnv":
        """Construct environment from an AgentConfig."""
        return cls(
            catalog=catalog,
            user=user,
            num_days=cfg.num_days,
            meals_per_day=cfg.meals_per_day,
            history_len=cfg.history_len,
            portion_levels=cfg.portion_levels,
            w_health=cfg.w_health,
            w_diversity=cfg.w_diversity,
            w_preference=cfg.w_preference,
            w_boundary=cfg.w_boundary,
            bootstrap_pool=bootstrap_pool,
            episode_resampler=episode_resampler,
        )

    def set_user(self, user: SimulatedUser) -> None:
        """Replace the env's user (eval harness uses this to cycle through
        the held-out pool). Call before ``reset()`` — this method does
        not handle mid-episode swaps.
        """
        if user.embedding_dim != self.emb_dim:
            raise ValueError(
                f"user.embedding_dim {user.embedding_dim} "
                f"!= env.emb_dim {self.emb_dim}"
            )
        self.user = user

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_action(self, action: int) -> tuple[int, float]:
        """Decode flat action index into (meal_idx, portion_multiplier)."""
        meal_idx = action // self.num_portions
        portion_idx = action % self.num_portions
        return meal_idx, self.portion_levels[portion_idx]

    def _get_meal_slot(self) -> int:
        """Current meal slot within the day (0=breakfast, 1=lunch, …)."""
        return self._step_count % self.meals_per_day

    def _get_day(self) -> int:
        """Current day index (0-based)."""
        return self._step_count // self.meals_per_day

    def _recent_mean_embedding(self) -> np.ndarray:
        """Mean-pooled recent meal embedding, normalised for cosine use."""
        if not self._recent_embeddings:
            return np.zeros(self.emb_dim, dtype=np.float32)
        recent_mean = np.mean(self._recent_embeddings[-self.history_len:], axis=0)
        norm = np.linalg.norm(recent_mean)
        if norm <= 1e-8:
            return np.zeros(self.emb_dim, dtype=np.float32)
        return (recent_mean / norm).astype(np.float32)

    def _coerce_observed_meal(
        self,
        observed_nutrition: Optional[np.ndarray],
        observed_embedding: Optional[np.ndarray],
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Validate optional CV-observed meal arrays.

        Observed nutrition and embedding must be supplied together because
        they represent the meal the user actually ate after a recommendation.
        """
        if observed_nutrition is None and observed_embedding is None:
            return None
        if observed_nutrition is None or observed_embedding is None:
            raise ValueError(
                "observed_nutrition and observed_embedding must be provided together"
            )

        nutrition = np.asarray(observed_nutrition, dtype=np.float32)
        if nutrition.shape != (4,):
            raise ValueError(
                f"observed_nutrition shape {nutrition.shape}; expected (4,)"
            )

        embedding = np.asarray(observed_embedding, dtype=np.float32)
        if embedding.shape != (self.emb_dim,):
            raise ValueError(
                f"observed_embedding shape {embedding.shape}; "
                f"expected ({self.emb_dim},)"
            )
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-8:
            raise ValueError("observed_embedding must be non-zero")
        embedding = embedding / norm

        return nutrition, embedding.astype(np.float32)

    def _build_obs(self) -> np.ndarray:
        daily_norm = self._daily_deficit / (self.user.daily_target + 1e-8)
        episode_target = self.user.daily_target * self.num_days
        episode_norm = self._episode_deficit / (episode_target + 1e-8)
        target_norm = self.user.daily_target / self.target_scale
        remaining_steps = (self.horizon - self._step_count) / max(self.horizon, 1)
        remaining_steps = np.array([remaining_steps], dtype=np.float32)

        slot_onehot = np.zeros(self.meals_per_day, dtype=np.float32)
        slot_onehot[self._get_meal_slot()] = 1.0

        recent_mean = self._recent_mean_embedding()

        return np.concatenate([
            daily_norm.astype(np.float32),
            episode_norm.astype(np.float32),
            target_norm.astype(np.float32),
            remaining_steps,
            slot_onehot,
            recent_mean.astype(np.float32),
            self.user.preference_embedding,
        ])

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # per-episode user mutation (random targets, fresh preference).
        # Runs BEFORE the deficit state is seeded so the new targets propagate.
        if self.episode_resampler is not None:
            self.episode_resampler(self.user, self.np_random)

        self._step_count = 0
        self._daily_deficit = self.user.daily_target.copy()
        self._weekly_deficit = self.user.weekly_target.copy()
        self._episode_deficit = self.user.daily_target.copy() * self.num_days

        # Bootstrap "user has been eating already this week": sample
        # ``history_len`` rows from the pool, use their embeddings to
        # warm up the diversity context (so the first few steps see a
        # non-zero recent-mean), and subtract their nutrition from the
        # weekly deficit (clipped to >= 0). Daily deficit is left fresh
        # — today is a new day. Uses the env's seeded ``np_random`` so
        # the same env-seed reproduces the same bootstrap.
        pool_size = len(self.bootstrap_embeddings)
        if pool_size > 0 and self.history_len > 0:
            indices = self.np_random.integers(
                low=0, high=pool_size, size=self.history_len
            )
            self._recent_embeddings = [
                self.bootstrap_embeddings[int(i)].copy() for i in indices
            ]
            consumed = self.bootstrap_nutrition[indices].sum(axis=0)
            self._weekly_deficit = np.maximum(
                self._weekly_deficit - consumed, 0.0
            )
        else:
            self._recent_embeddings = []

        return self._build_obs(), {}

    def step(
        self,
        action: int,
        observed_nutrition: Optional[np.ndarray] = None,
        observed_embedding: Optional[np.ndarray] = None,
    ):
        """Take one catalog-indexed meal-planning step.

        The supported training/evaluation path decodes ``action`` to a
        catalog meal and portion, then indexes the catalog's pre-generated
        embedding and nutrition metadata. For a future real-photo demo,
        CV-observed nutrition and embedding can be supplied together to
        update deficits and recent meal history with the meal the user
        actually ate.

        Args:
            action: flat action index in [0, num_actions).
            observed_nutrition: optional CV estimate [cal, protein, carbs, fat].
            observed_embedding: optional CV meal embedding in catalog CLIP space.
        """

        # Capture day/slot BEFORE incrementing step count so info dict
        # correctly reflects the step that was just taken.
        current_day = self._get_day()
        current_slot = self._get_meal_slot()

        meal_idx, portion = self._decode_action(action)
        catalog_nutrition = self.catalog.get_nutrition(meal_idx, portion)
        catalog_embedding = self.catalog.get_embedding(meal_idx)

        observed_meal = self._coerce_observed_meal(
            observed_nutrition, observed_embedding
        )
        if observed_meal is None:
            nutrition = catalog_nutrition
            embedding = catalog_embedding
            used_observed_meal = False
        else:
            nutrition, embedding = observed_meal
            used_observed_meal = True

        # 1. Health: how much daily deficit was reduced. Use per-component
        # normalized deficits so calories do not swamp protein/carbs/fat.
        old_daily = np.abs(
            self._daily_deficit / (self.user.daily_target + 1e-8)
        ).mean()
        new_daily_deficit = self._daily_deficit - nutrition
        new_daily_abs = np.abs(
            new_daily_deficit / (self.user.daily_target + 1e-8)
        ).mean()
        delta_health = old_daily - new_daily_abs

        # 2. Diversity: dissimilarity from recent meals
        if self._recent_embeddings:
            recent_mean = self._recent_mean_embedding()
            diversity = 1.0 - float(np.dot(embedding, recent_mean))
        else:
            diversity = 1.0

        # 3. Preference alignment (with stochastic noise for training)
        pref_score = self.user.preference_score(embedding)

        # 4. Boundary bonus at end of each day
        boundary_bonus = 0.0
        is_last_meal_of_day = (current_slot == self.meals_per_day - 1)
        if is_last_meal_of_day:
            remaining = np.abs(new_daily_deficit).sum()
            target_sum = self.user.daily_target.sum()
            nga_threshold = 0.10 * target_sum
            boundary_bonus = float(
                np.clip(1.0 - remaining / (nga_threshold + 1e-8), -1.0, 1.0)
            )

        weighted_health = self.w_health * delta_health
        weighted_diversity = self.w_diversity * diversity
        weighted_preference = self.w_preference * pref_score
        weighted_boundary = self.w_boundary * boundary_bonus
        terminal_bonus = 0.0
        weighted_terminal = 0.0

        reward = (
            weighted_health
            + weighted_diversity
            + weighted_preference
            + weighted_boundary
        )

        # --- Update state ---
        self._daily_deficit = new_daily_deficit
        if is_last_meal_of_day:
            self._daily_deficit = self.user.daily_target.copy()

        self._weekly_deficit -= nutrition
        self._episode_deficit -= nutrition
        self._recent_embeddings.append(embedding.copy())

        self._step_count += 1
        terminated = (self._step_count >= self.horizon)
        truncated = False

        if terminated:
            episode_remaining = np.abs(self._episode_deficit).sum()
            episode_target_sum = (self.user.daily_target * self.num_days).sum()
            terminal_bonus = max(
                0, 1.0 - episode_remaining / (episode_target_sum + 1e-8)
            )
            weighted_terminal = self.w_boundary * terminal_bonus
            reward += weighted_terminal

        return self._build_obs(), reward, terminated, truncated, {
            "meal_idx": meal_idx,
            "portion": portion,
            "nutrition": nutrition,
            "embedding": embedding.copy(),
            "used_observed_meal": used_observed_meal,
            "day": current_day,
            "slot": current_slot,
            "reward_terms": {
                "delta_health": float(delta_health),
                "diversity": float(diversity),
                "preference": float(pref_score),
                "boundary": float(boundary_bonus),
                "terminal": float(terminal_bonus),
                "weighted_health": float(weighted_health),
                "weighted_diversity": float(weighted_diversity),
                "weighted_preference": float(weighted_preference),
                "weighted_boundary": float(weighted_boundary),
                "weighted_terminal": float(weighted_terminal),
                "total": float(reward),
            },
        }
