"""CLI entrypoint for training the DQN meal planning agent.

Usage examples:

  # Experiment 1: 3-day baseline
  python -m agent.train --num_days 3 --total_timesteps 200000 --seed 42 \\
      --output_dir runs/exp1

  # Experiment 2: horizon scaling
  python -m agent.train --num_days 7 --total_timesteps 500000 \\
      --output_dir runs/exp2_7days

Outputs saved to output_dir/:
  dqn_model.zip   — trained SB3 DQN (load with DQN.load)
  config.json     — full AgentConfig for reproducibility
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agent.config import AgentConfig
from agent.catalog import MealCatalog
from agent.user import SimulatedUser
from agent.env import MealPlanningEnv
from agent.model import make_dqn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent.train",
        description="Train DQN agent on MealPlanningEnv.",
    )

    # TODO: Add --catalog_npy / --catalog_csv to load real Nutrition5k dish
    #       embeddings from the extract-embeddings pipeline (MealCatalog.load_from_file).
    # TODO: Add --user_history_npy to initialize preference embedding from a user's
    #       historical meal CLIP embeddings (SimulatedUser.from_meal_history).

    parser.add_argument(
        "--num_days", type=int, default=3,
        help="Planning horizon in days.  Set to 1, 3, 7 for Exp 2 horizon sweep.",
    )
    parser.add_argument("--meals_per_day", type=int, default=3)
    parser.add_argument(
        "--embedding_dim", type=int, default=512,
        help="Embedding dimension.  512 for CLIP ViT-B/32; use 32 for smoke tests.",
    )
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Directory to write dqn_model.zip and config.json.",
    )
    parser.add_argument("--w_health",     type=float, default=1.0)
    parser.add_argument("--w_diversity",  type=float, default=0.3)
    parser.add_argument("--w_preference", type=float, default=0.2)
    parser.add_argument("--w_boundary",   type=float, default=0.5)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    cfg = AgentConfig(
        num_days=args.num_days,
        meals_per_day=args.meals_per_day,
        embedding_dim=args.embedding_dim,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        w_health=args.w_health,
        w_diversity=args.w_diversity,
        w_preference=args.w_preference,
        w_boundary=args.w_boundary,
    )
    cfg.validate()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(args.output_dir / "config.json")

    catalog = MealCatalog(
        num_meals=cfg.num_meals,
        embedding_dim=cfg.embedding_dim,
        seed=cfg.seed,
    )
    user = SimulatedUser.from_config(cfg)
    env = MealPlanningEnv.from_config(cfg, catalog, user)

    print(f"Training DQN for {cfg.total_timesteps:,} steps")
    print(f"  Horizon   : {cfg.num_days} days × {cfg.meals_per_day} meals = {cfg.horizon} steps")
    print(f"  Actions   : {cfg.num_actions}  ({cfg.num_meals} meals × {len(cfg.portion_levels)} portions)")
    print(f"  Obs dim   : {cfg.obs_dim}  (embedding_dim={cfg.embedding_dim})")
    print(f"  Output    : {args.output_dir}")
    print()

    model = make_dqn(env, cfg)
    model.learn(total_timesteps=cfg.total_timesteps)

    save_path = args.output_dir / "dqn_model"
    model.save(str(save_path))
    print(f"\nModel saved  → {save_path}.zip")
    print(f"Config saved → {args.output_dir / 'config.json'}")


if __name__ == "__main__":
    main()
