"""CLI entrypoint for training the DQN meal planning agent.

Usage examples:

  # Default: dummy catalog + dummy style filter, randomised targets and
  # preference per episode.
  python -m agent.train --num_days 3 --total_timesteps 1000000 --seed 42 \\
      --output_dir runs/exp1

  # Pin a specific dietary persona (no target randomisation).
  python -m agent.train --num_days 7 --total_timesteps 1000000 \\
      --dietary_profile high_protein_lifter --output_dir runs/exp2

  # Three-component real catalog:
  python -m agent.train --catalog_dir artifacts/catalog/three_component/train \\
      --output_dir runs/exp3

Outputs saved to output_dir/:
  dqn_model.zip   — trained SB3 DQN (load with DQN.load)
  config.json     — full AgentConfig for reproducibility
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from agent.config import AgentConfig
from agent.catalog import MealCatalog
from agent.user import SimulatedUser
from agent.env import MealPlanningEnv
from agent.model import make_dqn
from agent.profiles import (
    NUTRITION_PERSONAS,
    TRAIN_STYLES,
    EVAL_STYLES,
    apply_persona,
    make_training_resampler,
    make_style_template_lists,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent.train",
        description="Train DQN agent on MealPlanningEnv.",
    )

    parser.add_argument(
        "--num_days", type=int, default=3,
        help="Planning horizon in days.  Set to 1, 3, 7 for Exp 2 horizon sweep.",
    )
    parser.add_argument("--meals_per_day", type=int, default=3)
    parser.add_argument(
        "--embedding_dim", type=int, default=512,
        help="Concatenated representation dimension for dummy catalogs; use 32 for smoke tests. "
             "Overridden when --catalog_dir is supplied.",
    )
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--learning_rate", type=float, default=AgentConfig.learning_rate,
        help="DQN optimizer learning rate. Default follows AgentConfig.",
    )
    parser.add_argument(
        "--gamma", type=float, default=AgentConfig.gamma,
        help="DQN discount factor. Lower values emphasize day-level rewards.",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Directory to write dqn_model.zip and config.json.",
    )
    parser.add_argument("--w_health",     type=float, default=1.0)
    parser.add_argument("--w_diversity",  type=float, default=0.3)
    parser.add_argument("--w_preference", type=float, default=0.2)
    parser.add_argument("--w_slot",       type=float, default=AgentConfig.w_slot)
    parser.add_argument("--w_boundary",   type=float, default=0.5)
    parser.add_argument(
        "--tb_log_dir", type=Path, default=None,
        help="Optional TensorBoard log directory.",
    )
    parser.add_argument(
        "--tb_log_name", type=str, default="dqn",
        help="TensorBoard run name used when --tb_log_dir is set.",
    )

    # ----- catalog / style / user-history wiring -----
    parser.add_argument(
        "--catalog_dir", type=Path, default=None,
        help="Directory containing catalog_manifest.csv plus ingredient, cuisine, "
             "and name embedding matrices. When omitted, a dummy catalog is used.",
    )
    parser.add_argument(
        "--dietary_profile", type=str, default=None,
        choices=sorted(NUTRITION_PERSONAS.keys()),
        help="Pin nutrition targets to a named persona (disables target "
             "randomisation). Useful for ablations and demos.",
    )
    parser.add_argument(
        "--randomize_targets", action=argparse.BooleanOptionalAction, default=True,
        help="Resample nutrition targets per episode from TARGET_RANGES. "
             "Forced off when --dietary_profile is set.",
    )
    parser.add_argument(
        "--randomize_preference", action=argparse.BooleanOptionalAction, default=True,
        help="Resample preference embedding per episode from the active "
             "training style pool.",
    )
    parser.add_argument(
        "--catalog_history_bootstrap",
        action=argparse.BooleanOptionalAction,
        default=AgentConfig.catalog_history_bootstrap,
        help="Seed recent-meal diversity history from catalog meals sampled "
             "around the current user preference. Does not alter nutrition deficits.",
    )
    parser.add_argument(
        "--catalog_history_preferred_fraction",
        type=float,
        default=AgentConfig.catalog_history_preferred_fraction,
        help="Fraction of catalog-history bootstrap rows sampled from the "
             "top preference-aligned catalog meals.",
    )
    parser.add_argument(
        "--bootstrap_pool_npy", type=Path, default=None,
        help="Optional .npy of shape (N, embedding_dim) for external "
             "diversity-history embeddings in the same catalog representation space. "
             "When supplied, this overrides catalog-history bootstrap.",
    )

    return parser


def _load_bootstrap_pool(
    path: Path, embedding_dim: int
) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != embedding_dim:
        raise ValueError(
            f"--bootstrap_pool_npy expected shape (N, {embedding_dim}), "
            f"got {arr.shape}"
        )
    emb = arr.astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError(f"{path} contains zero-vector rows")
    return emb / norms


class RewardTermLogger(BaseCallback):
    """Log raw and weighted reward terms emitted by MealPlanningEnv."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        term_values: dict[str, list[float]] = {}
        for info in infos:
            terms = info.get("reward_terms")
            if not terms:
                continue
            for key, value in terms.items():
                term_values.setdefault(key, []).append(float(value))

        for key, values in term_values.items():
            self.logger.record(f"reward_terms/{key}", float(np.mean(values)))
        return True


def main() -> None:
    args = build_parser().parse_args()
    if args.tb_log_dir is not None:
        try:
            import tensorboard  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "--tb_log_dir requires the tensorboard package. Install project "
                "dependencies with `uv sync` or `pip install -e .`."
            ) from exc

    # ----- catalog -----
    if args.catalog_dir is not None:
        catalog = MealCatalog.load_from_artifact(
            manifest_path=args.catalog_dir / "catalog_manifest.csv",
        )
    else:
        # Use the requested embedding_dim for the dummy catalog so smoke
        # tests can stay small.
        catalog = MealCatalog.load_dummy(
            num_meals=1000,
            embedding_dim=args.embedding_dim,
            seed=args.seed,
        )

    # ----- config (re-derived from the loaded catalog) -----
    cfg = AgentConfig(
        num_meals=catalog.num_meals,
        embedding_dim=catalog.embedding_dim,
        num_days=args.num_days,
        meals_per_day=args.meals_per_day,
        catalog_history_bootstrap=args.catalog_history_bootstrap,
        catalog_history_preferred_fraction=args.catalog_history_preferred_fraction,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        seed=args.seed,
        w_health=args.w_health,
        w_diversity=args.w_diversity,
        w_preference=args.w_preference,
        w_slot=args.w_slot,
        w_boundary=args.w_boundary,
    )
    cfg.validate()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(args.output_dir / "config.json")

    # ----- style template lists (train + eval, disjoint) -----
    style_lists = make_style_template_lists(
        catalog,
        style_names=tuple(TRAIN_STYLES) + tuple(EVAL_STYLES),
        per_style=max(1, catalog.num_meals // (len(TRAIN_STYLES) + len(EVAL_STYLES))),
        seed=args.seed,
    )

    # ----- training-style template pools (drive preference resampling) -----
    train_style_pools = {style: list(style_lists[style]) for style in TRAIN_STYLES}
    train_pool: list = []
    for s in TRAIN_STYLES:
        train_pool.extend(train_style_pools[s])
    if not train_pool:
        raise SystemExit("training style pool is empty; check style_lists")

    # ----- initial user (preference seeded from train pool) -----
    user = SimulatedUser.from_templates(train_pool, seed=cfg.seed + 1)

    # ----- persona pin (optional) -----
    randomize_targets = args.randomize_targets
    if args.dietary_profile is not None:
        apply_persona(user, args.dietary_profile)
        randomize_targets = False

    # ----- per-episode resampler -----
    # user identity/profile generator
    resampler = make_training_resampler(
        train_style_pools,
        randomize_targets=randomize_targets,
        randomize_preference=args.randomize_preference,
        style_names=TRAIN_STYLES,
    )

    # ----- diversity-history bootstrap -----
    if args.bootstrap_pool_npy is not None:
        bootstrap_pool = _load_bootstrap_pool(
            args.bootstrap_pool_npy, cfg.embedding_dim
        )
        bootstrap_source = f"external embedding npy ({len(bootstrap_pool)} rows)"
    else:
        bootstrap_pool = None
        bootstrap_source = (
            "catalog preference history"
            if cfg.catalog_history_bootstrap
            else "none"
        )

    env = MealPlanningEnv.from_config(
        cfg, catalog, user,
        bootstrap_pool=bootstrap_pool,
        episode_resampler=resampler,
    )

    print(f"Training DQN for {cfg.total_timesteps:,} steps")
    print(f"  Horizon       : {cfg.num_days} days × {cfg.meals_per_day} meals = {cfg.horizon} steps")
    print(f"  Actions       : {cfg.num_actions}  ({cfg.num_meals} meals × {len(cfg.portion_levels)} portions)")
    print(f"  Obs dim       : {cfg.obs_dim}  (embedding_dim={cfg.embedding_dim})")
    print(f"  Train pool    : {len(train_pool)} templates over {len(TRAIN_STYLES)} styles")
    print(f"  Persona       : {args.dietary_profile or '<randomised>'}")
    print(f"  Randomise tgt : {randomize_targets}")
    print(f"  Randomise pref: {args.randomize_preference}")
    print(f"  History boot  : {bootstrap_source}")
    print("  Q-network     : action scoring")
    print(f"  TensorBoard   : {args.tb_log_dir or 'none'}")
    print(f"  Output        : {args.output_dir}")
    print()

    model = make_dqn(
        env,
        cfg,
        catalog=catalog,
        tensorboard_log=str(args.tb_log_dir) if args.tb_log_dir else None,
    )
    callback = RewardTermLogger() if args.tb_log_dir else None
    model.learn(
        total_timesteps=cfg.total_timesteps,
        progress_bar=True,
        callback=callback,
        tb_log_name=args.tb_log_name,
    )

    save_path = args.output_dir / "dqn_model"
    model.save(str(save_path))
    print(f"\nModel saved  → {save_path}.zip")
    print(f"Config saved → {args.output_dir / 'config.json'}")


if __name__ == "__main__":
    main()
