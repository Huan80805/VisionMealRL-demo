from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from agent.catalog import MealCatalog
from agent.user import SimulatedUser

if TYPE_CHECKING:
    from agent.config import AgentConfig


class MealPlanningEnv(gym.Env):
    """Gymnasium environment for meal planning over a multi-day horizon.

    Episode = num_days × meals_per_day steps.
    At each step the agent picks (meal_template, portion_level).

    Observation layout (obs_dim = 8 + meals_per_day + 2 × embedding_dim):
      [0:4]                         daily_deficit  (normalized by daily_target)
      [4:8]                         weekly_deficit (normalized by weekly_target)
      [8 : 8+mpd]                   time slot one-hot  (mpd = meals_per_day)
      [8+mpd : 8+mpd+emb]           mean recent meal embedding
      [8+mpd+emb : 8+mpd+2*emb]    user preference embedding

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
    ):
        super().__init__()
        self.catalog = catalog
        self.user = user
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

        self.num_actions = catalog.num_meals * self.num_portions
        self.action_space = spaces.Discrete(self.num_actions)

        obs_dim = (
            4               # daily deficit
            + 4             # weekly deficit
            + meals_per_day # time slot one-hot
            + self.emb_dim  # mean-pooled recent embeddings
            + self.emb_dim  # user preference
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Internal state (initialized in reset)
        self._step_count = 0
        self._daily_deficit = np.zeros(4)
        self._weekly_deficit = np.zeros(4)
        self._recent_embeddings: list[np.ndarray] = []

    @classmethod
    def from_config(
        cls,
        cfg: "AgentConfig",
        catalog: MealCatalog,
        user: SimulatedUser,
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
        )

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

    def _build_obs(self) -> np.ndarray:
        daily_norm = self._daily_deficit / (self.user.daily_target + 1e-8)
        weekly_norm = self._weekly_deficit / (self.user.weekly_target + 1e-8)

        slot_onehot = np.zeros(self.meals_per_day, dtype=np.float32)
        slot_onehot[self._get_meal_slot()] = 1.0

        if self._recent_embeddings:
            recent_mean = np.mean(self._recent_embeddings[-self.history_len:], axis=0)
        else:
            recent_mean = np.zeros(self.emb_dim, dtype=np.float32)

        return np.concatenate([
            daily_norm.astype(np.float32),
            weekly_norm.astype(np.float32),
            slot_onehot,
            recent_mean.astype(np.float32),
            self.user.preference_embedding,
        ])

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._step_count = 0
        self._daily_deficit = self.user.daily_target.copy()
        self._weekly_deficit = self.user.weekly_target.copy()
        self._recent_embeddings = []
        return self._build_obs(), {}

    def step(self, action: int, observed_nutrition: Optional[np.ndarray] = None):
        """Take one meal-planning step.

        Args:
            action: flat action index in [0, num_actions).
            observed_nutrition: TODO hook for the CV module.  When real
                photo-based inference is available, pass the observed
                [cal, protein, carbs, fat] array here to override the
                catalog lookup.  Must be None until the hook is wired up.
        """
        # TODO: CV hook — when observed_nutrition is provided (from vision
        # module photo inference), use it instead of catalog.get_nutrition().
        assert observed_nutrition is None, "CV hook not yet implemented"

        # Capture day/slot BEFORE incrementing step count so info dict
        # correctly reflects the step that was just taken.
        current_day = self._get_day()
        current_slot = self._get_meal_slot()

        meal_idx, portion = self._decode_action(action)
        nutrition = self.catalog.get_nutrition(meal_idx, portion)
        embedding = self.catalog.get_embedding(meal_idx)

        # 1. Health: how much daily deficit was reduced (normalized)
        old_daily = np.abs(self._daily_deficit).sum()
        new_daily_deficit = self._daily_deficit - nutrition
        new_daily_abs = np.abs(new_daily_deficit).sum()
        delta_health = (old_daily - new_daily_abs) / (self.user.daily_target.sum() + 1e-8)

        # 2. Diversity: dissimilarity from recent meals
        if self._recent_embeddings:
            recent_stack = np.array(self._recent_embeddings[-self.history_len:])
            sims = recent_stack @ embedding  # cosine sims (embeddings are unit-normed)
            diversity = 1.0 - np.mean(sims)
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
            boundary_bonus = max(0, 1.0 - remaining / target_sum)

        reward = (
            self.w_health * delta_health
            + self.w_diversity * diversity
            + self.w_preference * pref_score
            + self.w_boundary * boundary_bonus
        )

        # --- Update state ---
        self._daily_deficit = new_daily_deficit
        if is_last_meal_of_day:
            self._daily_deficit = self.user.daily_target.copy()

        self._weekly_deficit -= nutrition
        self._recent_embeddings.append(embedding.copy())

        self._step_count += 1
        terminated = (self._step_count >= self.horizon)
        truncated = False

        # TODO: extend weekly bonus for >7-day horizons (provide bonus at end
        # of each 7-day window, not just episode end).
        if terminated and self.num_days > 1:
            weekly_remaining = np.abs(self._weekly_deficit).sum()
            weekly_target_sum = self.user.weekly_target.sum()
            reward += self.w_boundary * max(0, 1.0 - weekly_remaining / weekly_target_sum)

        return self._build_obs(), reward, terminated, truncated, {
            "meal_idx": meal_idx,
            "portion": portion,
            "nutrition": nutrition,
            "day": current_day,
            "slot": current_slot,
        }
