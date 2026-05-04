# Agent — Plan & TODO List

## §1. State of the agent

| Component | File | Status | Notes |
|---|---|---|---|
| Hyperparam config | `config.py` | &#9745; | `obs_dim` derived; serializable |
| Meal catalog | `catalog.py` | &#9745; | `load_dummy(num_meals=1000)` works; `load_from_artifact` loads `catalog_manifest.csv` + `catalog_embeddings.npy`; vectorised `embeddings_matrix` / `nutrition_matrix` exposed |
| Simulated user | `user.py` | &#9745; | `from_templates` + `from_meal_history` implemented and consistent |
| Profiles | `profiles.py` | &#9745; | `TARGET_RANGES`, 5 `NUTRITION_PERSONAS`, verified 8/8 `TRAIN_STYLES`/`EVAL_STYLES` partition, `make_training_resampler`, `build_eval_pool` (5×8×2 = 80), catalog style metadata loader |
| Env (rollout) | `env.py` | &#9745; | Nutrition5K/history pool sampled at reset; omitted pool means no pre-episode history; selected catalog/observed meals are appended during planning; observation now includes daily target, episode deficit, and remaining steps; NGA-aligned boundary reward + EDC terminal bonus implemented |
| Modular encoder + DQN | `model.py` | &#9745; | SB3 DQN algorithm with mandatory action-scoring Q head; state encoder features_dim = 304; spec calls for 112 |
| Baselines | `baseline.py` | &#9745; | HealthGreedy + MultiObjectiveGreedy + RandomPolicy implemented and aligned with env interfaces |
| Training CLI | `train.py` | &#9745; | `--tb_log_dir / --catalog_dir / --catalog_artifact / --catalog_embeddings / --styles_artifact / --dietary_profile / --randomize_(targets\|preference) / --bootstrap_pool_npy / --nutrition5k-data` wired; bootstrap pool is explicit Nutrition5K/history input |
| Evaluation harness | `evaluate.py` | &#9745;| Python API plus `python -m agent.evaluate`; supports policy tags `dqn`, `health`, `multi`, `random`; reports EDC, not WDC |
| Horizon-sweep driver | `scripts/agent_sweep.sh` | &#9745; | loops over horizons/seeds, trains action-scoring DQN, evaluates selected policy tags, and combines aggregate CSV rows |
| Tests | `tests/test_agent_section2*.py` | &#9745; | 7 end-to-end script tests cover the §2/§3/§4 plumbing; pytest is optional and may not be installed in the active env |

---

## §2. RL training-data construction (finished)

The agent's episode generator: a combination of a meal catalog, a user profile
(nutrition targets + preference embedding), bootstrapped recent meals,
the rollout environment, and a held-out evaluation pool. The detailed algorithmic behavior lives in the code docstrings/tests; this section is only the high-level map.

### Implementation map

| Area | Main symbols | File(s) | What to look for |
|---|---|---|---|
| Catalog contract | `MealTemplate`, `MealCatalog`, `load_dummy`, `load_from_artifact` | `catalog.py` | Stores recipe nutrition + CLIP embedding rows; exposes `embeddings_matrix` and `nutrition_matrix`; dummy catalog runs at 1,000-meal scale; real artifact loader handles the shipped catalog manifest/embedding pair. |
| User preference | `SimulatedUser`, `from_templates`, `from_meal_history`, `preference_score` | `user.py` | Builds preference embeddings by mean-pooling meal/template embeddings and L2-normalising; random-init path exists only as a fallback. |
| Episode env | `MealPlanningEnv`, `reset`, `step`, `set_user`, `estimate_observed_meal_from_photo` | `env.py` | Builds the MDP observation/action space, bootstraps Nutrition5K-derived recent meal history at reset, indexes catalog nutrition/embedding in `step`, optionally accepts observed CV nutrition/embedding for demos, and lets evaluation swap users before reset. |
| Training/eval profiles | `TARGET_RANGES`, `NUTRITION_PERSONAS`, `TRAIN_STYLES`, `EVAL_STYLES`, `make_training_resampler`, `build_eval_pool`, `make_style_template_lists` | `profiles.py` | Samples per-episode training targets/preferences and builds the 80-user held-out grid (`5 personas × 8 eval styles × 2 seeds`). |
| CLI wiring | `build_parser`, `_load_bootstrap_pool`, `_load_nutrition5k_history_pool`, `RewardTermLogger`, `main` | `train.py` | Re-derives config from the loaded catalog, wires target/preference randomisation, persona pins, explicit Nutrition5K bootstrap pools, TensorBoard reward-term logging, and catalog style metadata. |
| Evaluation | `EpisodeResult`, `evaluate_per_episode`, `evaluate_policy`, `compare_policies`, `main` | `evaluate.py` | Runs policies over either repeated seeds or a fixed `EvalUserSpec` sequence; CLI loads a run and writes aggregate metrics CSV. |
| Tests | `test_agent_section2*.py` | `tests/` | Scenario-style end-to-end tests covering catalog/user/env/profile/train/eval plumbing; currently 7 passing under `~/.venv/dl/bin/python`. |

### Data/artifact interfaces

The real catalog artifact is the action space: one row per candidate recipe/dish with `dish_name`, one or more source images or source text, pre-generated CLIP embedding, and nutrition metadata (`calories`, `protein`, `carbs`, `fat`, optional `mass`).
The embedding array must be row-aligned with the manifest because
`env.step(action)` decodes the action to `(meal_idx, portion)` and
then indexes `catalog.get_embedding(meal_idx)` and
`catalog.get_nutrition(meal_idx, portion)`. The shipped artifact at
`artifacts/catalog/.../finetuned_embeddings/train` is now supported via
`--catalog_dir`, which expects `catalog_manifest.csv` and
`catalog_embeddings.npy`.

The catalog style metadata now provides the style template lists.
Training uses the union of `TRAIN_STYLES`; evaluation uses only
`EVAL_STYLES`. The verified styles all exist in the shipped catalog with
53-56 templates each: train split = 442 templates, eval split = 443
templates. Dummy catalogs still use `make_dummy_style_template_lists(...)`
as a fallback.

The bootstrap pool is the user's pre-episode past meals and should
come from Nutrition5K-derived embeddings/nutrition, not the recipe
catalog. `train.py` can load it directly from a Nutrition5K embedding
split with `--nutrition5k-data`, where the directory contains
`dish_embeddings.npy` and `dish_manifest.csv` as documented in the
README. It maps `total_calories`, `total_protein`, `total_carb`, and
`total_fat` into the agent's `[calories, protein, carbs, fat]` order.
For prepacked experiments, `--bootstrap_pool_npy` still accepts
`(N, embedding_dim + 4)`: first `embedding_dim` columns are
recent-meal embeddings, last 4 are `[calories, protein, carbs, fat]`.
When omitted, the environment starts with no pre-consumed history.

### Verified status

Run with the project DL environment:

```bash
~/.venv/dl/bin/python tests/test_agent_section2.py
~/.venv/dl/bin/python tests/test_agent_section2_4.py
~/.venv/dl/bin/python tests/test_agent_section2_5_6.py
```

Current result: `3 + 2 + 2 = 7` end-to-end tests pass. These runners do not
require `pytest`; `~/.venv/dl/bin/python -m pytest tests/` only works
if `pytest` is installed in that environment.


## §3. Catalog-indexed rollout and optional CV feedback (agent side done)

For normal training/evaluation, each environment step is catalog-indexed:

1. The policy chooses a flat discrete action.
2. `env._decode_action(action)` maps it to `(meal_idx, portion)`.
3. `env.step(...)` reads the pre-generated catalog embedding with
   `catalog.get_embedding(meal_idx)`.
4. It reads nutrition metadata with
   `catalog.get_nutrition(meal_idx, portion)`.
5. The selected catalog embedding is appended to `_recent_embeddings`,
   so future steps see the recommended meals in the history context.

*The agent now does not run CV inference inside the SB3 rollout loop.*

The env-side demo hook is now wired:

- Simulation-only training/eval can keep calling `env.step(action)`.
- A future real-photo demo can call
  `env.step(action, observed_nutrition=..., observed_embedding=...)`.
  Those observed values replace the catalog values for deficit updates
  and recent-meal history.
- `estimate_observed_meal_from_photo(photo_path)` is a placeholder
  function that currently raises `NotImplementedError`. The TODO is to implement that function or replace it with an equivalent API returning
  `(nutrition[4], embedding[embedding_dim])` in the same CLIP space as
  the catalog.

---

## §4. Reward / state implementation

The report reward formula is still the implementation target:
`r_t = w₁Δhealth + w₂Δdiv + w₃Δpref + w₄b_boundary`.

Implemented pieces:

- Health: `old_normalized_deficit − new_normalized_deficit`, where
  each deficit is `mean(abs(deficit / daily_target))`. This keeps
  calories from dominating protein/carbs/fat and still penalises
  undershoot and overshoot.
- Diversity: `1 − cos(selected_embedding, mean_recent_embedding)`.
  The recent mean is a normalised centroid over the sliding
  `history_len` window, not every meal ever seen. `_build_obs` exposes
  that normalised recent mean, and
  `MultiObjectiveGreedy` consumes that same representation.
- Preference: cosine similarity, with user noise during rollout and
  deterministic dot products in evaluation.
- Per-day boundary bonus: fires at each last meal of day and is
  directly aligned with NGA:
  `clip(1 − abs(daily_deficit).sum() / (0.10 * daily_target.sum()), -1, 1)`.
  Passing the NGA threshold gives positive reward, the exact threshold
  gives zero, and missing the threshold gives negative reward.
- Episode terminal bonus: tracks `_episode_deficit = daily_target ×
  num_days − selected_nutrition_sum` and rewards closure at episode end.
- Evaluation metric: `EDC = 1 − |_episode_deficit|₁ /
  |daily_target × num_days|₁`. `WDC` has been removed because it does
  not match the milestone metric or non-7-day horizon sweep.

End-to-end test coverage:

- `test_end_to_end_history_bootstrap_observed_feedback_and_rollout`
  covers Nutrition5K bootstrap history, observed-meal feedback, and
  catalog-indexed rollout in one scenario.
- `test_end_to_end_edc_uses_episode_target_not_weekly_target`
  verifies EDC uses the episode target; the old weekly denominator
  would fail this case.

### State
`1040-dim` (13 + 3 + 2*512) mapped by `ModularEncoder` to `FEATURES_DIM = 304` (32+16+128+128):

| Encoder input | Raw dim | Encoded dim |
|---|---:|---:|
| Daily + episode deficits + daily target + remaining steps | 13 | 32 |
| Meal slot one-hot | `meals_per_day` (default 3) | 16 |
| Recent meal embedding | `embedding_dim` (default 512) | 128 |
| User preference embedding | `embedding_dim` (default 512) | 128 |

Targets are required because normalized deficits alone make different target personas indistinguishable at the start of each day; remaining steps keep long finite-horizon episodes
Markov.

### DQN changes from the original independent-head setup

The learning algorithm is still Stable-Baselines3 DQN: no Double DQN target, no distributional Q, no actor-critic method. The old DQN architecture with independent action-id output heads has been removed. `make_dqn(...)` now builds `ActionScoringDQNPolicy`, so every train/eval run scores catalog actions from their fixed meal features.

Implemented DQN-facing changes:

- **Observation now includes absolute target information.** The original
  state used normalized deficits only, so different personas looked
  identical at the start of a day. We added scaled `daily_target` so DQN
  can distinguish, for example, a 1600 kcal target from a 3000 kcal
  target.
- **Observation uses episode deficit, not fixed weekly deficit.** The old
  state exposed `weekly_deficit / weekly_target`, where `weekly_target`
  is always 7 days. This is inconsistent for horizon sweeps like
  `num_days=21`. The current state exposes
  `_episode_deficit / (daily_target * num_days)`, which matches the
  actual episode horizon.
- **Observation includes `remaining_steps_fraction`.** Long finite
  horizons are not Markov if the agent cannot tell whether it is early
  or late in the episode. This scalar gives DQN time-to-go information,
  especially important for `num_days=21` (`63` meal decisions).
- **Reward health term is component-normalized.** We changed health from
  raw L1 closure to mean per-component normalized closure so calories do
  not dominate macro errors.
- **Boundary reward is NGA-aligned.** Day-end reward is now positive for
  passing the 10% NGA threshold, zero at the threshold, and negative for
  missing it. This is why DQN's NGA became meaningful after the reward
  alignment sweep.
- **Mandatory action-scoring Q head.** SB3 DQN training now scores
  actions from fixed action features:
  `[meal_embedding, scaled_nutrition, portion]`. Internally it uses a
  dueling-style state value plus centered action advantage, with
  normalized state/action latents, so the 3000 discrete actions share
  structure instead of using unrelated output heads.
- **Experiment controls are CLI-configurable.** `train.py` and
  `scripts/agent_sweep.sh` now expose reward weights, `--gamma`,
  TensorBoard logging, catalog path, and Nutrition5K history input. This
  lets us run health-heavy and long-horizon ablations without changing
  code.

---

## §5. Experiments & evaluation (P1)

### Baseline policies
`baseline.py` now includes HealthGreedy, MultiObjectiveGreedy,
and RandomPolicy.

### Horizon-scaling driver — `scripts/agent_sweep.sh`
Basic driver is implemented.

- For `num_days ∈ {1, 3, 7, 21}`, hold all other hyperparams fixed.
- Train DQN once per horizon/seed.
- Evaluate selected policy tags with `python -m agent.evaluate`.
- Persist one aggregate `eval_metrics.csv` per run and a combined
  `sweep_metrics.csv`.
- Supports env overrides for reward weights, `GAMMA`, TensorBoard root, catalog path, Nutrition5K history path, total timesteps, horizons, and seeds.
- TODO: add optional per-episode CSV output if the report needs
  persona/style breakdowns instead of aggregate rows only.

### Logging
- `env.step(...)` emits `info["reward_terms"]`, and
  `RewardTermLogger` logs raw and weighted terms when `--tb_log_dir`
  is provided.

### Reward-weight sweep
Supported through CLI/env overrides. The useful current setting after
NGA reward alignment is:

```text
W_HEALTH=3.0
W_DIVERSITY=0.1
W_PREFERENCE=0.1
W_BOUNDARY=1.0
```

The health-only setting (`W_HEALTH=5.0`, diversity/preference `0.0`) improved episode closure but was worse for balanced NGA/DDS/PA tradeoffs.
For long-horizon DQN, `GAMMA=0.95` is exposed as an ablation knob to emphasize day-level boundary rewards over very long episode returns.

---

## §6. Potential algorithmic upgrades

### 6.1 Double DQN
`model.py:6-12` already calls this out. SB3 doesn't ship Double DQN
out of the box; either subclass `DQN` and override the loss, or swap
to `sb3-contrib`. Worth doing only after the baseline DQN results
are locked in, so we can A/B against them in the report.

### 6.2 Action masking
At inference, some actions are obviously bad (e.g. picking the same
meal back-to-back, or a meal whose calories exceed remaining daily
deficit by 3×). A small mask in `predict` would help the greedy
baselines and could be ported to DQN via `sb3-contrib`'s
`MaskablePPO`-style API.

### 6.3 CLIP-drift contrastive re-anchor
If the CV part fine-tunes CLIP, the agent's action
embeddings (recipe text) and observed embeddings (dish images) drift
apart. Track this risk; if it materialises, contribute a contrastive
re-anchoring step on recipe text–image pairs and re-extract the
catalog.
