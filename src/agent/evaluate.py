"""Evaluation harness for the meal planning agent.

Five metrics reported per policy (mean ± std over N episodes):

  NGA  — Nutritional Goal Adherence:
          % of days where L1(daily_deficit) < 10 % of daily_target_sum.

  DDS  — Dietary Diversity Score:
          1 − mean pairwise cosine similarity of meal embeddings in the episode.

  PA   — Preference Alignment:
          mean cosine similarity between chosen meal embeddings and user_pref
          (deterministic, no noise) per step.

  WDC  — Weekly Diet Completeness:
          1 − L1(weekly_deficit_final) / weekly_target_sum.

  Return — sum of step rewards over the episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agent.config import AgentConfig
from agent.env import MealPlanningEnv


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Raw data collected during a single rollout."""
    total_return: float
    daily_deficits: list          # one np.ndarray per day (daily_target - daily_nutrition)
    meal_embeddings: list         # one np.ndarray per step
    pref_scores: list             # float per step: cos_sim(meal_emb, user_pref)
    weekly_deficit_final: np.ndarray  # env._weekly_deficit at episode end


@dataclass
class AggregatedMetrics:
    """Mean ± std across N episodes for all five metrics."""
    nga_mean: float
    nga_std: float
    dds_mean: float
    dds_std: float
    pa_mean: float
    pa_std: float
    wdc_mean: float
    wdc_std: float
    ret_mean: float
    ret_std: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pairwise_cosine_sim(embeddings: list[np.ndarray]) -> float:
    """Mean pairwise cosine similarity of a list of (approximately) unit vectors."""
    if len(embeddings) < 2:
        return 0.0
    mat = np.stack(embeddings, axis=0)  # (N, emb_dim)
    gram = mat @ mat.T                  # (N, N)
    n = len(embeddings)
    off_diag = 1.0 - np.eye(n)
    return float((gram * off_diag).sum() / (off_diag.sum() + 1e-8))


def _run_episode(
    policy,
    env: MealPlanningEnv,
    cfg: AgentConfig,
    seed: int,
) -> EpisodeResult:
    """Execute one full episode and collect raw metric data."""
    obs, _ = env.reset(seed=seed)
    total_return = 0.0
    meal_embeddings: list[np.ndarray] = []
    pref_scores: list[float] = []
    daily_deficits: list[np.ndarray] = []

    daily_nutrition = np.zeros(4, dtype=np.float32)

    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_return += reward
        done = terminated or truncated

        meal_idx = info["meal_idx"]
        nutrition = info["nutrition"]
        slot = info["slot"]

        emb = env.catalog.get_embedding(meal_idx)
        meal_embeddings.append(emb.copy())

        # Preference score without noise for reproducible evaluation
        pref_scores.append(float(np.dot(env.user.preference_embedding, emb)))

        daily_nutrition += nutrition
        if slot == env.meals_per_day - 1:
            # End of a day — record the daily deficit and reset accumulator
            daily_deficits.append(env.user.daily_target - daily_nutrition)
            daily_nutrition = np.zeros(4, dtype=np.float32)

    return EpisodeResult(
        total_return=total_return,
        daily_deficits=daily_deficits,
        meal_embeddings=meal_embeddings,
        pref_scores=pref_scores,
        weekly_deficit_final=env._weekly_deficit.copy(),
    )


def _aggregate(
    results: list[EpisodeResult],
    env: MealPlanningEnv,
) -> AggregatedMetrics:
    daily_target_sum = env.user.daily_target.sum()
    weekly_target_sum = env.user.weekly_target.sum()

    nga_vals, dds_vals, pa_vals, wdc_vals, ret_vals = [], [], [], [], []

    for r in results:
        # NGA: fraction of days within 10 % of daily target
        if r.daily_deficits:
            nga = float(np.mean([
                np.abs(d).sum() < 0.10 * daily_target_sum
                for d in r.daily_deficits
            ]))
        else:
            nga = 0.0
        nga_vals.append(nga)

        # DDS: 1 − mean pairwise cosine similarity
        dds_vals.append(1.0 - _pairwise_cosine_sim(r.meal_embeddings))

        # PA: mean cosine similarity to user preference
        pa_vals.append(float(np.mean(r.pref_scores)) if r.pref_scores else 0.0)

        # WDC: 1 − L1(weekly_deficit_final) / weekly_target_sum
        wdc = 1.0 - np.abs(r.weekly_deficit_final).sum() / (weekly_target_sum + 1e-8)
        wdc_vals.append(float(wdc))

        ret_vals.append(r.total_return)

    def _ms(vals: list[float]) -> tuple[float, float]:
        return float(np.mean(vals)), float(np.std(vals))

    return AggregatedMetrics(
        nga_mean=_ms(nga_vals)[0], nga_std=_ms(nga_vals)[1],
        dds_mean=_ms(dds_vals)[0], dds_std=_ms(dds_vals)[1],
        pa_mean=_ms(pa_vals)[0],  pa_std=_ms(pa_vals)[1],
        wdc_mean=_ms(wdc_vals)[0], wdc_std=_ms(wdc_vals)[1],
        ret_mean=_ms(ret_vals)[0], ret_std=_ms(ret_vals)[1],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_policy(
    policy,
    env: MealPlanningEnv,
    cfg: AgentConfig,
    n_episodes: int = 50,
) -> AggregatedMetrics:
    """Evaluate a policy over N independent episodes."""
    results = [_run_episode(policy, env, cfg, seed=i) for i in range(n_episodes)]
    return _aggregate(results, env)


def compare_policies(
    policies: dict[str, Any],
    env: MealPlanningEnv,
    cfg: AgentConfig,
    n_episodes: int = 50,
) -> dict[str, AggregatedMetrics]:
    """Evaluate multiple policies and return a dict of AggregatedMetrics."""
    return {
        name: evaluate_policy(policy, env, cfg, n_episodes)
        for name, policy in policies.items()
    }


def print_comparison_table(results: dict[str, AggregatedMetrics]) -> None:
    """Print a formatted comparison table to stdout."""
    col_w = 18
    header = (
        f"{'Policy':<20}"
        f"{'NGA':>{col_w}}"
        f"{'DDS':>{col_w}}"
        f"{'PA':>{col_w}}"
        f"{'WDC':>{col_w}}"
        f"{'Return':>{col_w}}"
    )
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        def fmt(mean, std):
            return f"{mean:.3f}±{std:.3f}"
        print(
            f"{name:<20}"
            f"{fmt(m.nga_mean, m.nga_std):>{col_w}}"
            f"{fmt(m.dds_mean, m.dds_std):>{col_w}}"
            f"{fmt(m.pa_mean,  m.pa_std):>{col_w}}"
            f"{fmt(m.wdc_mean, m.wdc_std):>{col_w}}"
            f"{fmt(m.ret_mean, m.ret_std):>{col_w}}"
        )
