"""Evaluation harness for the meal planning agent.

Five metrics reported per policy (mean ± std over N episodes):

  NGA  — Nutritional Goal Adherence:
          % of days where L1(daily_deficit) < 10 % of daily_target_sum.

  DDS  — Dietary Diversity Score:
          1 − mean pairwise cosine similarity of meal embeddings in the episode.

  PA   — Preference Alignment:
          mean cosine similarity between chosen meal embeddings and user_pref
          (deterministic, no noise) per step.

  EDC  — Episodic Deficit Closure:
          1 − L1(episode_deficit_final) / L1(daily_target × num_days).

  Return — sum of step rewards over the episode.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from agent.baseline import HealthGreedy, MultiObjectiveGreedy, RandomPolicy
from agent.catalog import MealCatalog
from agent.config import AgentConfig
from agent.env import MealPlanningEnv
from agent.profiles import (
    EVAL_STYLES,
    TRAIN_STYLES,
    EvalUserSpec,
    build_eval_pool,
    make_dummy_style_template_lists,
)
from agent.user import SimulatedUser


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
    daily_target: np.ndarray      # user target used for this episode
    episode_target: np.ndarray    # daily_target × env.num_days
    episode_deficit_final: np.ndarray  # env._episode_deficit at episode end
    metadata: dict = field(default_factory=dict)
    # When the episode is driven by an EvalUserSpec, metadata carries
    # {"persona", "style", "seed"} so per-cell breakdowns can be built
    # downstream.


@dataclass
class AggregatedMetrics:
    """Mean ± std across N episodes for all five metrics."""
    nga_mean: float
    nga_std: float
    dds_mean: float
    dds_std: float
    pa_mean: float
    pa_std: float
    edc_mean: float
    edc_std: float
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

        nutrition = info["nutrition"]
        slot = info["slot"]

        emb = info["embedding"]
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
        daily_target=env.user.daily_target.copy(),
        episode_target=env.user.daily_target.copy() * env.num_days,
        episode_deficit_final=env._episode_deficit.copy(),
    )


def _aggregate(
    results: list[EpisodeResult],
    env: MealPlanningEnv,
) -> AggregatedMetrics:
    nga_vals, dds_vals, pa_vals, edc_vals, ret_vals = [], [], [], [], []

    for r in results:
        daily_target_sum = r.daily_target.sum()

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

        # EDC: 1 − L1(episode_deficit_final) / L1(episode_target)
        edc = 1.0 - np.abs(r.episode_deficit_final).sum() / (
            np.abs(r.episode_target).sum() + 1e-8
        )
        edc_vals.append(float(edc))

        ret_vals.append(r.total_return)

    def _ms(vals: list[float]) -> tuple[float, float]:
        return float(np.mean(vals)), float(np.std(vals))

    return AggregatedMetrics(
        nga_mean=_ms(nga_vals)[0], nga_std=_ms(nga_vals)[1],
        dds_mean=_ms(dds_vals)[0], dds_std=_ms(dds_vals)[1],
        pa_mean=_ms(pa_vals)[0],  pa_std=_ms(pa_vals)[1],
        edc_mean=_ms(edc_vals)[0], edc_std=_ms(edc_vals)[1],
        ret_mean=_ms(ret_vals)[0], ret_std=_ms(ret_vals)[1],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_per_episode(
    policy,
    env: MealPlanningEnv,
    cfg: AgentConfig,
    n_episodes: int = 50,
    user_iter: Optional[Iterable[EvalUserSpec]] = None,
) -> list[EpisodeResult]:
    """Run N rollouts and return the raw per-episode data.

    When ``user_iter`` is provided, one episode is run per spec: the
    held-out user is swapped onto the env via ``set_user`` and the spec's
    seed drives ``env.reset``. ``n_episodes`` is ignored in this mode.
    Otherwise the env keeps its existing user and seeds are ``range(n_episodes)``.
    """
    if user_iter is None:
        return [_run_episode(policy, env, cfg, seed=i) for i in range(n_episodes)]

    results: list[EpisodeResult] = []
    for spec in user_iter:
        env.set_user(spec.user)
        result = _run_episode(policy, env, cfg, seed=spec.seed)
        result.metadata = {
            "persona": spec.persona,
            "style": spec.style,
            "seed": spec.seed,
        }
        results.append(result)
    return results


def evaluate_policy(
    policy,
    env: MealPlanningEnv,
    cfg: AgentConfig,
    n_episodes: int = 50,
    user_iter: Optional[Iterable[EvalUserSpec]] = None,
) -> AggregatedMetrics:
    """Evaluate a policy and return aggregated metrics.

    See ``evaluate_per_episode`` for the semantics of ``user_iter``.
    """
    results = evaluate_per_episode(policy, env, cfg, n_episodes, user_iter)
    return _aggregate(results, env)


def compare_policies(
    policies: dict[str, Any],
    env: MealPlanningEnv,
    cfg: AgentConfig,
    n_episodes: int = 50,
    user_iter: Optional[Sequence[EvalUserSpec]] = None,
) -> dict[str, AggregatedMetrics]:
    """Evaluate multiple policies and return a dict of AggregatedMetrics.

    When ``user_iter`` is supplied it must be a *sequence* (re-iterable)
    so each policy sees the same held-out users.
    """
    return {
        name: evaluate_policy(policy, env, cfg, n_episodes, user_iter)
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
        f"{'EDC':>{col_w}}"
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
            f"{fmt(m.edc_mean, m.edc_std):>{col_w}}"
            f"{fmt(m.ret_mean, m.ret_std):>{col_w}}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

POLICY_ALIASES = {
    "dqn": "dqn",
    "policy": "dqn",
    "health": "health",
    "health_greedy": "health",
    "multi": "multi",
    "multi_greedy": "multi",
    "multi_objective": "multi",
    "random": "random",
}


def metrics_to_rows(
    results: dict[str, AggregatedMetrics],
    cfg: AgentConfig,
    run_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for policy, metrics in results.items():
        row = {
            "run_dir": str(run_dir),
            "policy": policy,
            "num_days": cfg.num_days,
            "meals_per_day": cfg.meals_per_day,
            "horizon": cfg.horizon,
            "seed": cfg.seed,
            "total_timesteps": cfg.total_timesteps,
            "nga_mean": metrics.nga_mean,
            "nga_std": metrics.nga_std,
            "dds_mean": metrics.dds_mean,
            "dds_std": metrics.dds_std,
            "pa_mean": metrics.pa_mean,
            "pa_std": metrics.pa_std,
            "edc_mean": metrics.edc_mean,
            "edc_std": metrics.edc_std,
            "return_mean": metrics.ret_mean,
            "return_std": metrics.ret_std,
        }
        rows.append(row)
    return rows


def write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no metrics rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent.evaluate",
        description="Evaluate a trained DQN run against baseline policies.",
    )
    parser.add_argument(
        "--run_dir", type=Path, required=True,
        help="Training run directory containing config.json and dqn_model.zip.",
    )
    parser.add_argument(
        "--policies", nargs="+",
        default=["dqn", "health", "multi", "random"],
        help="Policy tags to evaluate. Supported: dqn, health, multi, random. "
             "Aliases: policy, health_greedy, multi_greedy.",
    )
    parser.add_argument(
        "--output_csv", type=Path, default=None,
        help="Where to write aggregate metrics. Defaults to "
             "<run_dir>/eval_metrics.csv.",
    )
    parser.add_argument(
        "--n_eval_seeds", type=int, default=2,
        help="Number of seeds per persona/style cell. Default 2 gives 50 "
             "held-out episodes.",
    )
    parser.add_argument(
        "--eval_seed", type=int, default=None,
        help="Seed for constructing the held-out eval pool. Defaults to the "
             "training config seed.",
    )
    parser.add_argument(
        "--random_seed", type=int, default=0,
        help="Seed for RandomPolicy.",
    )
    parser.add_argument(
        "--catalog_artifact", type=Path, default=None,
        help="Optional real catalog manifest. Currently waits on "
             "MealCatalog.load_from_artifact.",
    )
    parser.add_argument(
        "--catalog_embeddings", type=Path, default=None,
        help="Catalog embedding .npy required with --catalog_artifact.",
    )
    return parser


def _normalise_policy_tags(tags: Sequence[str]) -> list[str]:
    normalised: list[str] = []
    for raw in tags:
        tag = raw.lower().strip()
        if tag not in POLICY_ALIASES:
            raise SystemExit(
                f"unknown policy tag {raw!r}; supported: "
                f"{sorted(set(POLICY_ALIASES))}"
            )
        resolved = POLICY_ALIASES[tag]
        if resolved not in normalised:
            normalised.append(resolved)
    return normalised


def _load_eval_catalog(args: argparse.Namespace, cfg: AgentConfig) -> MealCatalog:
    if args.catalog_artifact is not None:
        if args.catalog_embeddings is None:
            raise SystemExit("--catalog_artifact requires --catalog_embeddings")
        return MealCatalog.load_from_artifact(
            manifest_path=args.catalog_artifact,
            embeddings_path=args.catalog_embeddings,
        )

    return MealCatalog.load_dummy(
        num_meals=cfg.num_meals,
        embedding_dim=cfg.embedding_dim,
        seed=cfg.seed,
    )


def _build_policy_map(
    tags: Sequence[str],
    run_dir: Path,
    env: MealPlanningEnv,
    cfg: AgentConfig,
    random_seed: int,
) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    for tag in tags:
        if tag == "dqn":
            from stable_baselines3 import DQN

            model_path = run_dir / "dqn_model.zip"
            if not model_path.exists():
                raise FileNotFoundError(f"missing trained model: {model_path}")
            policies["DQN"] = DQN.load(str(model_path), env=env)
        elif tag == "health":
            policies["HealthGreedy"] = HealthGreedy(env)
        elif tag == "multi":
            policies["MultiObjectiveGreedy"] = MultiObjectiveGreedy(env, cfg)
        elif tag == "random":
            policies["Random"] = RandomPolicy(env, seed=random_seed)
        else:
            raise AssertionError(f"unhandled policy tag: {tag}")
    return policies


def main() -> None:
    args = build_parser().parse_args()
    cfg = AgentConfig.from_json(args.run_dir / "config.json")
    catalog = _load_eval_catalog(args, cfg)

    if catalog.num_meals != cfg.num_meals:
        raise ValueError(
            f"catalog has {catalog.num_meals} meals but config expects "
            f"{cfg.num_meals}"
        )
    if catalog.embedding_dim != cfg.embedding_dim:
        raise ValueError(
            f"catalog embedding_dim {catalog.embedding_dim} != "
            f"config embedding_dim {cfg.embedding_dim}"
        )

    style_lists = make_dummy_style_template_lists(
        catalog,
        style_names=tuple(TRAIN_STYLES) + tuple(EVAL_STYLES),
        per_style=max(
            1, catalog.num_meals // (len(TRAIN_STYLES) + len(EVAL_STYLES))
        ),
        seed=cfg.seed,
    )
    eval_pool = build_eval_pool(
        style_lists,
        n_seeds=args.n_eval_seeds,
        seed=cfg.seed if args.eval_seed is None else args.eval_seed,
    )

    user = SimulatedUser.from_templates(catalog.meals[:10], seed=cfg.seed)
    env = MealPlanningEnv.from_config(cfg, catalog, user)
    tags = _normalise_policy_tags(args.policies)
    policies = _build_policy_map(tags, args.run_dir, env, cfg, args.random_seed)

    results = compare_policies(policies, env, cfg, user_iter=eval_pool)
    print_comparison_table(results)

    output_csv = args.output_csv or (args.run_dir / "eval_metrics.csv")
    rows = metrics_to_rows(results, cfg, args.run_dir)
    write_metrics_csv(output_csv, rows)
    print(f"\nMetrics saved -> {output_csv}")


if __name__ == "__main__":
    main()
