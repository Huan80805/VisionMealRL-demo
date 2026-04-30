"""CLI entrypoint for training the DQN meal planning agent.

Usage examples:

  # Default: dummy catalog + dummy style filter, randomised targets and
  # preference per episode.
  python -m agent.train --num_days 3 --total_timesteps 200000 --seed 42 \\
      --output_dir runs/exp1

  # Pin a specific dietary persona (no target randomisation).
  python -m agent.train --num_days 7 --total_timesteps 500000 \\
      --dietary_profile high_protein_lifter --output_dir runs/exp2

  # Real catalog + real style filter (once those artifacts ship):
  python -m agent.train --catalog_artifact data/catalog.csv \\
      --catalog_embeddings data/embeddings.npy \\
      --styles_artifact data/styles.json \\
      --output_dir runs/exp3

Outputs saved to output_dir/:
  dqn_model.zip   — trained SB3 DQN (load with DQN.load)
  config.json     — full AgentConfig for reproducibility
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

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
    make_dummy_style_template_lists,
)
from visionmealrl import load_dish_embedding_lookup


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
        help="Embedding dimension.  512 for CLIP ViT-B/32; use 32 for smoke tests. "
             "Overridden when --catalog_artifact is supplied.",
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

    # ----- catalog / style / user-history wiring -----
    # TODO: wire to the real catalog manifest + embeddings file once the
    # format is frozen. Until then --catalog_artifact must remain unset
    # and the dummy catalog is used.
    parser.add_argument(
        "--catalog_artifact", type=Path, default=None,
        help="Path to catalog manifest (CSV/JSON). When omitted, falls back "
             "to MealCatalog.load_dummy.",
    )
    parser.add_argument(
        "--catalog_embeddings", type=Path, default=None,
        help="Path to catalog embeddings .npy. Required when "
             "--catalog_artifact is set.",
    )
    # TODO: wire to the real dietary-style filter artifact once the
    # format is frozen. Expected to load a dict[str, list[MealTemplate]]
    # keyed by style name. Until then a dummy disjoint partition is used.
    parser.add_argument(
        "--styles_artifact", type=Path, default=None,
        help="Path to dietary-style filter artifact. When omitted, "
             "make_dummy_style_template_lists partitions the catalog instead.",
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
    history_group = parser.add_mutually_exclusive_group()
    history_group.add_argument(
        "--bootstrap_pool_npy", type=Path, default=None,
        help="Optional .npy of shape (N, embedding_dim+4) for the user "
        "meal-history bootstrap pool. First embedding_dim columns = "
             "embeddings, last 4 = [cal, protein, carbs, fat]. This should "
             "come from Nutrition5K-derived past-meal history. When omitted, "
             "episodes start with no pre-consumed history.",
    )
    history_group.add_argument(
        "--nutrition5k-data", dest="nutrition5k_data", type=Path, default=None,
        help="Nutrition5K embedding split directory containing "
             "dish_embeddings.npy and dish_manifest.csv. Used only for "
             "user-history bootstrap, not for the action catalog.",
    )
    parser.add_argument(
        "--nutrition5k-subsample-size", type=int, default=None,
        help="Optional number of Nutrition5K dishes to keep in the bootstrap "
             "pool after loading --nutrition5k-data. Defaults to all rows.",
    )

    return parser


def _load_bootstrap_pool(
    path: Path, embedding_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != embedding_dim + 4:
        raise ValueError(
            f"--bootstrap_pool_npy expected shape (N, {embedding_dim + 4}), "
            f"got {arr.shape}"
        )
    embeddings = arr[:, :embedding_dim].astype(np.float32)
    nutrition = arr[:, embedding_dim:embedding_dim + 4].astype(np.float32)
    return embeddings, nutrition


def _metadata_float(
    metadata: dict[str, object], field: str, dish_id: str
) -> float:
    if field not in metadata:
        raise ValueError(f"Nutrition5K dish {dish_id} missing {field}")
    try:
        return float(metadata[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Nutrition5K dish {dish_id} has non-numeric {field}: "
            f"{metadata[field]!r}"
        ) from exc


def _load_nutrition5k_history_pool(
    split_dir: Path,
    embedding_dim: int,
    subsample_size: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = load_dish_embedding_lookup(split_dir)
    if not lookup:
        raise ValueError(f"No Nutrition5K dishes found in {split_dir}")

    embeddings: list[np.ndarray] = []
    nutrition_rows: list[list[float]] = []
    for dish_id in sorted(lookup):
        record = lookup[dish_id]
        embedding = np.asarray(record["embedding"], dtype=np.float32)
        if embedding.shape != (embedding_dim,):
            raise ValueError(
                f"Nutrition5K dish {dish_id} embedding shape {embedding.shape}; "
                f"expected ({embedding_dim},)"
            )

        metadata = record["nutrition_metadata"]
        nutrition_rows.append([
            _metadata_float(metadata, "total_calories", dish_id),
            _metadata_float(metadata, "total_protein", dish_id),
            _metadata_float(metadata, "total_carb", dish_id),
            _metadata_float(metadata, "total_fat", dish_id),
        ])
        embeddings.append(embedding)

    emb = np.stack(embeddings, axis=0).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError(f"Nutrition5K embeddings in {split_dir} contain zero rows")
    emb = emb / norms
    nut = np.asarray(nutrition_rows, dtype=np.float32)

    if subsample_size is not None:
        if subsample_size <= 0:
            raise ValueError("--nutrition5k-subsample-size must be positive")
        if subsample_size < emb.shape[0]:
            rng = np.random.default_rng(seed)
            indices = rng.choice(emb.shape[0], size=subsample_size, replace=False)
            emb = emb[indices]
            nut = nut[indices]

    return emb, nut


def main() -> None:
    args = build_parser().parse_args()

    # ----- catalog -----
    if args.catalog_artifact is not None:
        if args.catalog_embeddings is None:
            raise SystemExit(
                "--catalog_artifact requires --catalog_embeddings"
            )
        catalog = MealCatalog.load_from_artifact(
            manifest_path=args.catalog_artifact,
            embeddings_path=args.catalog_embeddings,
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

    # ----- style template lists (train + eval, disjoint) -----
    if args.styles_artifact is not None:
        # TODO: load the real dict[str, list[MealTemplate]] from disk.
        raise NotImplementedError(
            f"--styles_artifact={args.styles_artifact} loader is awaiting "
            f"the artifact format; omit the flag to use the dummy partition."
        )
    style_lists = make_dummy_style_template_lists(
        catalog,
        style_names=tuple(TRAIN_STYLES) + tuple(EVAL_STYLES),
        per_style=max(1, catalog.num_meals // (len(TRAIN_STYLES) + len(EVAL_STYLES))),
        seed=args.seed,
    )

    # ----- training-style template pool (drives preference resampling) -----
    train_pool: list = []
    for s in TRAIN_STYLES:
        train_pool.extend(style_lists[s])
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
        train_pool,
        randomize_targets=randomize_targets,
        randomize_preference=args.randomize_preference,
    )

    # ----- bootstrap pool (optional Nutrition5K/history override) -----
    # past meal history initializer
    if args.bootstrap_pool_npy is not None:
        bootstrap_pool = _load_bootstrap_pool(
            args.bootstrap_pool_npy, cfg.embedding_dim
        )
        bootstrap_source = f"packed npy ({len(bootstrap_pool[0])} rows)"
    elif args.nutrition5k_data is not None:
        bootstrap_pool = _load_nutrition5k_history_pool(
            args.nutrition5k_data,
            cfg.embedding_dim,
            subsample_size=args.nutrition5k_subsample_size,
            seed=args.seed,
        )
        bootstrap_source = (
            f"Nutrition5K split ({len(bootstrap_pool[0])} rows)"
        )
    else:
        bootstrap_pool = None
        bootstrap_source = "none"

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
    print(f"  Bootstrap pool: {bootstrap_source}")
    print(f"  Output        : {args.output_dir}")
    print()

    model = make_dqn(env, cfg)
    model.learn(total_timesteps=cfg.total_timesteps, progress_bar=True)

    save_path = args.output_dir / "dqn_model"
    model.save(str(save_path))
    print(f"\nModel saved  → {save_path}.zip")
    print(f"Config saved → {args.output_dir / 'config.json'}")


if __name__ == "__main__":
    main()
