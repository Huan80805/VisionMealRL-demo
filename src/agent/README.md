# Meal Planning Agent

This package contains the DQN meal-planning agent. The agent chooses a
meal template and portion size at each environment step, then receives a
reward based on nutrition deficit closure, diversity, preference
alignment, and meal-boundary bonuses.

Use the project DL environment unless you intentionally installed the
package somewhere else:

```bash
~/.venv/dl/bin/python -m agent.train --help
```

## What Works Today

- Dummy-catalog training runs end to end.
- Optional Nutrition5K-derived recent-meal history can be loaded with
  `--nutrition5k-data`.
- Evaluation is available as both a CLI and Python API in
  `agent.evaluate`.
- Real action-catalog and dietary-style artifacts are still integration
  TODOs.

The Nutrition5K data is **not** the action catalog. It is only used to
bootstrap the user's recent meal history. The recommendation action
space remains `MealCatalog`.

## Training

Minimal smoke run:

```bash
~/.venv/dl/bin/python -m agent.train \
  --num_days 3 \
  --total_timesteps 10000 \
  --output_dir runs/agent_smoke
```

Longer dummy-catalog run:

```bash
~/.venv/dl/bin/python -m agent.train \
  --num_days 3 \
  --total_timesteps 200000 \
  --seed 42 \
  --output_dir runs/agent_exp1
```

Run with Nutrition5K recent-history bootstrap:

```bash
~/.venv/dl/bin/python -m agent.train \
  --num_days 3 \
  --total_timesteps 200000 \
  --output_dir runs/agent_n5k_history \
  --nutrition5k-data artifacts/embeddings/open_clip_ViT-B-32_laion2b_s34b_b79k/overhead_rgb/train \
  --nutrition5k-subsample-size 1000
```

`--nutrition5k-data` should point to a split directory containing
`dish_embeddings.npy` and `dish_manifest.csv`, as produced by the
top-level `visionmealrl extract-embeddings` command.

Training writes:

- `dqn_model.zip`: Stable-Baselines3 DQN model.
- `config.json`: serialized `AgentConfig` used for the run.

## Evaluation

Evaluate a run from the CLI:

```bash
~/.venv/dl/bin/python -m agent.evaluate \
  --run_dir runs/agent_exp1 \
  --policies dqn health multi random \
  --output_csv runs/agent_exp1/eval_metrics.csv
```

Policy tags:

- `dqn`: trained DQN loaded from `<run_dir>/dqn_model.zip`.
- `health`: health-only greedy baseline.
- `multi`: one-step multi-objective greedy baseline.
- `random`: uniform random baseline.

The CLI prints a comparison table and writes aggregate rows to CSV. Use
the Python API when you need custom env/policy construction:

```python
from stable_baselines3 import DQN

from agent.baseline import HealthGreedy, MultiObjectiveGreedy
from agent.catalog import MealCatalog
from agent.config import AgentConfig
from agent.env import MealPlanningEnv
from agent.evaluate import compare_policies, print_comparison_table
from agent.profiles import (
    EVAL_STYLES,
    TRAIN_STYLES,
    build_eval_pool,
    make_dummy_style_template_lists,
)
from agent.user import SimulatedUser

run_dir = "runs/agent_exp1"
cfg = AgentConfig.from_json(f"{run_dir}/config.json")

# Until the real catalog loader lands, recreate the dummy catalog with
# the same shape. For final experiments, load the same real catalog used
# during training.
catalog = MealCatalog.load_dummy(
    num_meals=cfg.num_meals,
    embedding_dim=cfg.embedding_dim,
    seed=cfg.seed,
)

style_lists = make_dummy_style_template_lists(
    catalog,
    style_names=tuple(TRAIN_STYLES) + tuple(EVAL_STYLES),
    per_style=max(1, catalog.num_meals // (len(TRAIN_STYLES) + len(EVAL_STYLES))),
    seed=cfg.seed,
)
eval_pool = build_eval_pool(style_lists, n_seeds=2, seed=cfg.seed)

user = SimulatedUser.from_templates(catalog.meals[:10], seed=cfg.seed)
env = MealPlanningEnv.from_config(cfg, catalog, user)

dqn = DQN.load(f"{run_dir}/dqn_model.zip", env=env)
policies = {
    "DQN": dqn,
    "HealthGreedy": HealthGreedy(env),
    "MultiObjectiveGreedy": MultiObjectiveGreedy(env, cfg),
}

results = compare_policies(policies, env, cfg, user_iter=eval_pool)
print_comparison_table(results)
```

The held-out evaluation pool is a grid of:

```text
5 nutrition personas x 5 held-out dietary styles x 2 seeds = 50 episodes
```

## Reading Metrics

`print_comparison_table(...)` reports mean +/- std across episodes:

- `NGA`: Nutritional Goal Adherence. Fraction of days where the final
  daily nutrition deficit is within 10% of the daily target sum. Higher
  is better.
- `DDS`: Dietary Diversity Score. `1 - mean_pairwise_cosine_similarity`
  across meal embeddings chosen during the episode. Higher means less
  repeated embedding content.
- `PA`: Preference Alignment. Mean cosine similarity between selected
  meal embeddings and the user's preference embedding. Higher is better.
- `EDC`: Episodic Deficit Closure.
  `1 - L1(final_episode_deficit) / L1(daily_target x num_days)`.
  Higher is better; `1.0` means the episode exactly met the aggregate
  target.
- `Return`: sum of environment rewards. Useful for optimization
  debugging, but the report should emphasize NGA/DDS/PA/EDC.

During training, the environment's step reward uses:

```text
r_t = w_health * delta_health
    + w_diversity * diversity
    + w_preference * preference
    + w_boundary * boundary_bonus
```

The diversity reward is `1 - cos(selected_embedding,
mean_recent_embedding)`, where `mean_recent_embedding` is the normalized
mean over the sliding `history_len` recent-meal window.

## Data Roles

There are three different data sources. Keep them separate.

### Action Catalog

The action catalog defines what the policy can recommend. Every action
decodes to:

```text
meal_idx = action // num_portions
portion_idx = action % num_portions
```

The environment then indexes:

```python
catalog.get_embedding(meal_idx)
catalog.get_nutrition(meal_idx, portion)
```

Current status: `MealCatalog.load_dummy(...)` works;
`MealCatalog.load_from_artifact(...)` is still a stub.

Desired real format:

- Manifest with one row per recommendable meal template.
- Row-aligned embedding `.npy` array with shape `(N, embedding_dim)`.
- Required nutrition columns:
  `calories`, `protein`, `carbs`, `fat`.
- Required identity/display fields:
  stable meal id and human-readable meal name.
- Optional fields:
  source image paths, recipe text, mass, cuisine/style tags.

### Dietary-Style Lists

Training and evaluation need style-filtered template lists:

```python
dict[str, list[MealTemplate]]
```

Expected style keys:

- Training: `japanese`, `mediterranean`, `vegan`, `indian`, `american`
- Evaluation: `mexican`, `korean`, `italian`, `middle_eastern`,
  `caribbean`

Current status: `make_dummy_style_template_lists(...)` creates a dummy
disjoint partition. The real `--styles_artifact` loader is still a TODO.

### Nutrition5K Recent History

Nutrition5K is used to initialize what the user recently ate, not to
define the action space.

`--nutrition5k-data` expects a split directory with:

```text
dish_embeddings.npy
dish_manifest.csv
```

The loader reads these manifest columns:

- `total_calories`
- `total_protein`
- `total_carb`
- `total_fat`

It converts them into the agent's `[calories, protein, carbs, fat]`
order and normalizes each embedding row. At `env.reset()`, the env
samples `history_len` rows from this pool, appends their embeddings to
recent history, and subtracts their nutrition from the weekly deficit.

## Integration Contracts (TODOs)

Needed from the meal catalogue:

- Real action-catalog manifest and embedding file.
- Confirm exact field names for meal id, meal name, nutrition columns,
  and embedding row alignment.
- Real dietary-style artifact mapping style names to catalog meal ids or
  directly to `MealTemplate`-compatible rows.

Needed from the Nutrition5K/CV:

- Keep producing README-compatible Nutrition5K embedding split dirs:
  `dish_embeddings.npy` + `dish_manifest.csv`.
- Ensure manifest contains `total_calories`, `total_protein`,
  `total_carb`, and `total_fat`.
- For future demo/photo feedback, provide an API equivalent to:

```python
nutrition, embedding = estimate_observed_meal_from_photo(photo_path)
```

where:

- `nutrition.shape == (4,)` in `[calories, protein, carbs, fat]` order.
- `embedding.shape == (embedding_dim,)`.
- The embedding is in the same CLIP space as the action catalog.

## Sweep Script

Use `scripts/agent_sweep.sh` to train and evaluate multiple horizons:

```bash
TOTAL_TIMESTEPS=200000 \
HORIZONS="1 3 7 21" \
SEEDS="42" \
scripts/agent_sweep.sh
```

The script writes one run directory per horizon/seed under
`runs/agent_sweep` by default, then combines each run's
`eval_metrics.csv` into `runs/agent_sweep/sweep_metrics.csv`.

Useful overrides:

```bash
OUTPUT_ROOT=runs/agent_sweep_demo
POLICIES="dqn health multi random"
N_EVAL_SEEDS=2
NUTRITION5K_DATA=artifacts/embeddings/open_clip_ViT-B-32_laion2b_s34b_b79k/overhead_rgb/train
NUTRITION5K_SUBSAMPLE_SIZE=1000
```

## Agent-side TODOs

- Implement `MealCatalog.load_from_artifact(...)`.
- Implement the real `--styles_artifact` loader.
- Add real-catalog support to `agent.evaluate` once the catalog loader
  is implemented.

## Tests

The current agent tests are scenario-level scripts:

```bash
~/.venv/dl/bin/python tests/test_agent_section2.py
~/.venv/dl/bin/python tests/test_agent_section2_4.py
~/.venv/dl/bin/python tests/test_agent_section2_5_6.py
```

They can also be run with `pytest` if `pytest` is installed in the
environment.
