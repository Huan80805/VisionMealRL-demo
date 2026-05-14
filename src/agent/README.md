# Meal Planning Agent

The agent chooses a meal template and portion size at each environment step.
Rewards combine nutrition deficit closure, meal diversity, preference
alignment, slot validity, and meal-boundary bonuses.

Use the project DL environment from the repository root:

```bash
PYTHONPATH=src ~/.venv/dl/bin/python -m agent.train --help
```

## Current Representation

The current training path uses explicit three-component meal representations:

```text
ingredient_vector + cuisine_vector + recipe_name_vector
```

The active catalog artifacts live at:

```text
artifacts/catalog/three_component/train
```

Required files:

- `catalog_manifest.csv`: recipe metadata, meal type, dish type, image path, and per-serving nutrition.
- `ingredient_embeddings.npy`: ingredient weight-share / TF-IDF style vectors.
- `cuisine_embeddings.npy`: cuisine phrase embeddings.
- `name_embeddings.npy`: recipe-name prompt embeddings.
- `metadata.json`: component dimensions, ingredient vocabulary, templates, and source file.

The agent loads the three component matrices directly.

Build or refresh the active catalog artifacts with:

```bash
PYTHONPATH=src ~/.venv/dl/bin/python scripts/build_agent_catalog_artifacts.py \
  --recipe-file data/meal_catalog.csv \
  --output-dir artifacts/catalog/three_component/train
```

## What Works Today

- Dummy-catalog training runs end to end.
- Real action-catalog loading works with `--catalog_dir`.
- Training preference resampling samples one training style first, then samples
  templates within that style, so users have sharper style-specific
  preferences.
- Diversity history is bootstrapped from the active catalog by default, in the
  same ingredient/cuisine/name representation used by the reward.
- Evaluation is available as both a CLI and Python API in `agent.evaluate`.
- Catalog style metadata is used for train/eval style splits.

Recent-history bootstrap is used only to warm up diversity calculations. It
does not subtract prior meals from nutrition deficits.

## Training

Dummy-catalog run:

```bash
PYTHONPATH=src ~/.venv/dl/bin/python -m agent.train \
  --num_days 3 \
  --total_timesteps 200000 \
  --seed 42 \
  --output_dir runs/agent_exp1
```

Run with the real action catalog:

```bash
PYTHONPATH=src ~/.venv/dl/bin/python -m agent.train \
  --num_days 3 \
  --total_timesteps 200000 \
  --output_dir runs/agent_catalog \
  --catalog_dir artifacts/catalog/three_component/train
```

Disable catalog-derived recent-history bootstrap for an ablation:

```bash
PYTHONPATH=src ~/.venv/dl/bin/python -m agent.train \
  --num_days 3 \
  --total_timesteps 200000 \
  --output_dir runs/agent_no_history \
  --catalog_dir artifacts/catalog/three_component/train \
  --no-catalog_history_bootstrap
```

Training writes:

- `dqn_model.zip`: Stable-Baselines3 DQN model.
- `config.json`: serialized `AgentConfig` used for the run.

Training uses SB3 DQN with an action-scoring Q-network. The Q-network scores
each action from fixed catalog action features:

```text
[meal_representation, scaled_nutrition, portion, valid_slot_flags]
```

`meal_representation` is the concatenated ingredient, cuisine, and recipe-name
representation. The default DQN learning rate is `1e-4`; override it with
`--learning_rate` when running ablations.

TensorBoard reward-term logging:

```bash
PYTHONPATH=src ~/.venv/dl/bin/python -m agent.train \
  --num_days 3 \
  --total_timesteps 200000 \
  --output_dir runs/agent_catalog_logged \
  --catalog_dir artifacts/catalog/three_component/train \
  --tb_log_dir runs/tensorboard \
  --tb_log_name agent_catalog_logged
```

## Reward

During training, the environment's step reward uses:

```text
r_t = w_health * delta_health
    + w_diversity * diversity_score
    + w_preference * preference_score
    + w_slot * slot_score
    + w_boundary * boundary_bonus
    + terminal_terms
```

Preference is the weighted component similarity between the selected meal and
the simulated user's preference profile:

```text
preference_score =
    0.40 * cos(user_ingredient_pref, meal_ingredient)
  + 0.35 * cos(user_cuisine_pref, meal_cuisine)
  + 0.25 * cos(user_name_pref, meal_name)
  + preference_noise
```

The default simulated preference noise is `0.05`.

Diversity is ingredient-only distance from the recent-meal history. Cuisine
and recipe-name components are intentionally excluded so diversity does not
reward leaving the user's preferred cuisine/style:

```text
diversity_score =
    1.00 * (1 - cos(meal_ingredient, recent_ingredient_mean))
```

Slot handling is penalty-only; actions are not hard-masked:

```text
slot_score =  0.0  if meal matches slot
slot_score = -0.3  if snack is used in a main slot
slot_score = -1.0  otherwise
```

Slot mapping:

```text
slot 0 -> breakfast
slot 1 -> lunch/dinner
slot 2 -> lunch/dinner
```

Reward logs include component terms:

```text
preference_ingredient, preference_cuisine, preference_name,
diversity_ingredient, diversity_cuisine, diversity_name,
slot_score, weighted_slot
```

## Evaluation

Evaluate a run from the CLI:

```bash
PYTHONPATH=src ~/.venv/dl/bin/python -m agent.evaluate \
  --run_dir runs/agent_exp1 \
  --policies dqn health multi random \
  --output_csv runs/agent_exp1/eval_metrics.csv \
  --catalog_dir artifacts/catalog/three_component/train
```

Policy tags:

- `dqn`: trained DQN loaded from `<run_dir>/dqn_model.zip`.
- `health`: health-only greedy baseline.
- `multi`: one-step multi-objective greedy baseline using the same component reward terms.
- `random`: uniform random baseline.

`print_comparison_table(...)` reports mean +/- std across episodes:

- `NGA`: Nutritional Goal Adherence. Fraction of days where the final daily
  nutrition deficit is within 10% of the daily target sum.
- `DDS`: Dietary Diversity Score. `1 - mean_pairwise_component_similarity`
  across selected meals using the ingredient diversity component.
- `PA`: Preference Alignment. Mean weighted ingredient/cuisine/name preference
  score between selected meals and the user's preference profile, without
  training-time preference noise.
- `EDC`: Episodic Deficit Closure.
  `1 - L1(final_episode_deficit) / L1(daily_target x num_days)`.
- `Return`: sum of environment rewards. Useful for optimization debugging, but
  reports should emphasize NGA/DDS/PA/EDC.

The held-out evaluation pool is a grid of:

```text
5 nutrition personas x 8 held-out dietary styles x 2 seeds = 80 episodes
```

## Observations

The DQN observation includes:

- Normalized daily deficit.
- Normalized episode deficit.
- Scaled absolute daily target.
- Remaining-steps fraction.
- Current meal-slot one-hot.
- Recent-history representation using concatenated component vectors.
- User preference representation using concatenated component vectors.

The absolute target matters because two users can have the same normalized
deficit but need different catalog nutrition. The episode-deficit feature is
normalized by `daily_target * num_days`, so it stays meaningful across horizon
settings.

## Style Splits

Training and evaluation use style-filtered template lists:

```python
dict[str, list[MealTemplate]]
```

Default training styles:

- `american`, `asian`, `mediterranean`, `central europe`
- `nordic`, `chinese`, `indian`, `japanese`

Default evaluation styles:

- `french`, `italian`, `mexican`, `south american`
- `eastern europe`, `british`, `middle eastern`, `south east asian`

`make_style_template_lists(...)` uses real catalog style metadata when present
and falls back to a dummy partition only for synthetic catalogs.

## Sweep Script

Use `scripts/agent_sweep.sh` to train and evaluate multiple horizons:

```bash
TOTAL_TIMESTEPS=200000 \
HORIZONS="1 3 7 21" \
SEEDS="42" \
scripts/agent_sweep.sh
```

The script writes one run directory per horizon/seed under
`runs/agent_sweep` by default, then combines each run's `eval_metrics.csv` into
`runs/agent_sweep/sweep_metrics.csv`.

Useful overrides:

```bash
CATALOG_DIR=artifacts/catalog/three_component/train
OUTPUT_ROOT=runs/agent_sweep_demo
POLICIES="dqn health multi random"
TOTAL_TIMESTEPS=1000000
W_HEALTH=3.0
W_DIVERSITY=0.1
W_PREFERENCE=0.1
W_SLOT=0.25
W_BOUNDARY=1.0
GAMMA=0.95
CATALOG_HISTORY_BOOTSTRAP=1
CATALOG_HISTORY_PREFERRED_FRACTION=0.7
BOOTSTRAP_POOL_NPY=/path/to/optional_bootstrap_pool.npy
TB_LOG_ROOT=runs/agent_sweep_demo/tensorboard
N_EVAL_SEEDS=2
```

`BOOTSTRAP_POOL_NPY` is optional. If provided, it must contain vectors with the
same total representation dimension as the active catalog.
