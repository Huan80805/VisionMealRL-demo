# Meal Planning Agent

The agent chooses a meal template and portion size at each environment step, then receives a reward based on nutrition deficit closure, diversity, preference
alignment, and meal-boundary bonuses.

Use the project DL environment unless you intentionally installed the package somewhere else:

```bash
~/.venv/dl/bin/python -m agent.train --help
```

## What Works Today

- Dummy-catalog training runs end to end.
- Real action-catalog loading works with `--catalog_dir`.
- Optional Nutrition5K-derived recent-meal history can be loaded with
  `--nutrition5k-data`.
- Evaluation is available as both a CLI and Python API in
  `agent.evaluate`.
- Catalog style metadata is used for train/eval style splits.

The Nutrition5K data is **not** the action catalog. It is only used to
bootstrap the user's recent meal history. The recommendation action
space remains `MealCatalog`.

## Training

Dummy-catalog run:

```bash
python -m agent.train \
  --num_days 3 \
  --total_timesteps 200000 \
  --seed 42 \
  --output_dir runs/agent_exp1
```

Run with the real action catalog:

```bash
python -m agent.train \
  --num_days 3 \
  --total_timesteps 200000 \
  --output_dir runs/agent_catalog \
  --catalog_dir artifacts/catalog/open_clip_ViT-B-32_laion2b_s34b_b79k_overhead_rgb/finetuned_embeddings/train
```

Run with Nutrition5K recent-history bootstrap:

```bash
python -m agent.train \
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

Training always uses SB3 DQN with the action-scoring Q-network. The Q-network scores each action from fixed catalog action features `[meal_embedding, scaled_nutrition, portion]`

The default DQN learning rate is `1e-4`. Override it with
`--learning_rate` when running ablations.

TensorBoard reward-term logging:
```bash
python -m agent.train \
  --num_days 3 \
  --total_timesteps 200000 \
  --output_dir runs/agent_catalog_logged \
  --catalog_dir artifacts/catalog/open_clip_ViT-B-32_laion2b_s34b_b79k_overhead_rgb/finetuned_embeddings/train \
  --tb_log_dir runs/tensorboard \
  --tb_log_name agent_catalog_logged
```

## Evaluation

Evaluate a run from the CLI:

```bash
python -m agent.evaluate \
  --run_dir runs/agent_exp1 \
  --policies dqn health multi random \
  --output_csv runs/agent_exp1/eval_metrics.csv \
  --catalog_dir artifacts/catalog/open_clip_ViT-B-32_laion2b_s34b_b79k_overhead_rgb/finetuned_embeddings/train
```

Policy tags:

- `dqn`: trained DQN loaded from `<run_dir>/dqn_model.zip`.
- `health`: health-only greedy baseline.
- `multi`: one-step multi-objective greedy baseline.
- `random`: uniform random baseline.

The CLI prints a comparison table and writes aggregate rows to CSV. The held-out evaluation pool is a grid of:

```text
5 nutrition personas x 8 held-out dietary styles x 2 seeds = 80 episodes
```

## Reading Metrics

`print_comparison_table(...)` reports mean +/- std across episodes:

- `NGA`: Nutritional Goal Adherence. Fraction of days where the final daily nutrition deficit is within 10% of the daily target sum. Higher is better.
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
mean_recent_embedding)`, where `mean_recent_embedding` is the normalized mean over the sliding `history_len` recent-meal window.

The DQN observation includes normalized daily deficit, normalized
episode deficit, scaled absolute daily target, and a remaining-steps
fraction. The absolute target matters because two users can have the
same normalized deficit but need very different catalog nutrition. The
episode-deficit feature is normalized by `daily_target * num_days`, so it
stays meaningful for all horizon settings. The remaining-steps feature
keeps the finite-horizon MDP Markov for long episodes.

## Additional data notes

### Dietary-Style Lists

Training and evaluation need style-filtered template lists:

```python
dict[str, list[MealTemplate]]
```

Verified style keys:

- Training: `american`, `asian`, `mediterranean`, `central europe`,
  `nordic`, `chinese`, `indian`, `japanese`
- Evaluation: `french`, `italian`, `mexican`, `south american`,
  `eastern europe`, `british`, `middle eastern`, `south east asian`

The shipped catalog has 53-56 templates for each of these styles. The training split has 442 templates and the evaluation split has 443 templates. `make_style_template_lists(...)` uses real catalog style metadata when present and falls back to a dummy partition only for synthetic catalogs.

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
TOTAL_TIMESTEPS=1000000
W_HEALTH=3.0
W_DIVERSITY=0.1
W_PREFERENCE=0.1
W_BOUNDARY=1.0
GAMMA=0.95
TB_LOG_ROOT=runs/agent_sweep_demo/tensorboard
N_EVAL_SEEDS=2
NUTRITION5K_DATA=artifacts/embeddings/open_clip_ViT-B-32_laion2b_s34b_b79k/overhead_rgb/train
NUTRITION5K_SUBSAMPLE_SIZE=1000
```