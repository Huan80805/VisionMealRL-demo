from __future__ import annotations

import numpy as np

from agent.config import AgentConfig
from agent.env import MealPlanningEnv


class RandomPolicy:
    """Uniform-random baseline matching SB3's ``predict`` interface."""

    def __init__(self, env: MealPlanningEnv, seed: int = 0):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic: bool = True):
        """Return one random discrete action."""
        del obs, deterministic
        return int(self.rng.integers(0, self.env.num_actions)), None


class HealthGreedy:
    """Myopic health-only baseline.

    Always picks the meal + portion that minimises remaining daily deficit
    (L1 norm).  Ignores diversity, preference, and future consequences.

    Copied from GreedyBaseline in the original agent.py.
    """

    def __init__(self, env: MealPlanningEnv):
        self.env = env

    def predict(self, obs, deterministic: bool = True):
        """Match SB3's predict(obs, deterministic) interface."""
        best_action = 0
        best_score = -np.inf

        for action in range(self.env.num_actions):
            meal_idx = action // self.env.num_portions
            portion_idx = action % self.env.num_portions
            portion = self.env.portion_levels[portion_idx]
            nutrition = self.env.catalog.get_nutrition(meal_idx, portion)

            new_deficit = self.env._daily_deficit - nutrition
            score = -np.abs(new_deficit).sum()

            if score > best_score:
                best_score = score
                best_action = action

        return best_action, None


class MultiObjectiveGreedy:
    """One-step lookahead baseline using the full reward formula.

    Scores each candidate action by the same weighted combination of health,
    diversity, and preference used by MealPlanningEnv.step(), but looks only
    one step ahead (no planning).  Preference is deterministic (no noise).
    """

    def __init__(self, env: MealPlanningEnv, cfg: AgentConfig):
        self.env = env
        self.cfg = cfg

        mpd = cfg.meals_per_day
        emb = cfg.embedding_dim

        # Obs slice boundaries — must match MealPlanningEnv._build_obs
        self._s_daily = slice(0, 4)
        self._s_episode = slice(4, 8)
        self._s_target = slice(8, 12)
        self._s_remaining = slice(12, 13)
        self._s_time = slice(13, 13 + mpd)
        self._s_meal_emb = slice(13 + mpd, 13 + mpd + emb)
        self._s_user_pref = slice(13 + mpd + emb, 13 + mpd + 2 * emb)

    def _parse_obs(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract (daily_deficit, recent_mean_emb, user_pref) from observation."""
        daily_norm = obs[self._s_daily]
        daily_deficit = daily_norm * (self.env.user.daily_target + 1e-8)
        recent_emb = obs[self._s_meal_emb]
        user_pref = obs[self._s_user_pref]
        return daily_deficit, recent_emb, user_pref

    def predict(self, obs, deterministic: bool = True):
        """Match SB3's predict(obs, deterministic) interface."""
        daily_deficit, recent_emb, user_pref = self._parse_obs(obs)

        best_action = 0
        best_score = -np.inf

        for action in range(self.env.num_actions):
            meal_idx = action // self.env.num_portions
            portion_idx = action % self.env.num_portions
            portion = self.env.portion_levels[portion_idx]
            nutrition = self.env.catalog.get_nutrition(meal_idx, portion)
            embedding = self.env.catalog.get_embedding(meal_idx)

            # Health component
            old_daily = np.abs(
                daily_deficit / (self.env.user.daily_target + 1e-8)
            ).mean()
            new_daily = daily_deficit - nutrition
            delta_health = old_daily - np.abs(
                new_daily / (self.env.user.daily_target + 1e-8)
            ).mean()

            # Diversity component
            diversity = 1.0 - float(np.dot(recent_emb, embedding))

            # Preference component (no noise — deterministic baseline)
            pref_score = float(np.dot(user_pref, embedding))

            score = (
                self.cfg.w_health * delta_health
                + self.cfg.w_diversity * diversity
                + self.cfg.w_preference * pref_score
            )

            if score > best_score:
                best_score = score
                best_action = action

        return best_action, None
