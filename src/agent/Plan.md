# Agent — Plan & TODO List

This plan reconciles the milestone PDF (`DL_Project_Milestone.pdf`,
Sec. 3.3 / 3.4 / 4.2 / 5) with the current state of `src/agent/` and
ranks the remaining work by priority. Each item links to the file/symbol
that needs to change.

Legend: `[P0]` ship-blocker for the integrated demo · `[P1]` required
for the experiments / hypotheses in §3.4 / §4.2 · `[P2]` polish or
nice-to-have.

**Boundary with teammate:**
- *Teammate owns:* construction of the meal catalog (each entry =
  `dish_name`, `images`, `nutrition_metadata`), filtering the catalog
  by dietary style (Japanese, Mediterranean, …) into a list of
  matching templates, and the upstream CV pipeline.
- *Agent (us) owns:* everything that consumes those two artifacts —
  the catalog reader, preference embedding constructor, env, reward,
  DQN, baselines, evaluation, training loop.

---

## §1. State of the agent today

| Component | File | Status | Notes |
|---|---|---|---|
| Hyperparam config | `config.py` | ✅ done | `obs_dim` derived; serializable |
| Meal catalog | `catalog.py` | ✅ done | `load_dummy(num_meals=1000)` works; `load_from_artifact` stub awaits real format; vectorised `embeddings_matrix` / `nutrition_matrix` exposed |
| Simulated user | `user.py` | ✅ done | `from_templates` + `from_meal_history` implemented and consistent |
| Profiles | `profiles.py` | ✅ done | `TARGET_RANGES`, 5 `NUTRITION_PERSONAS`, `TRAIN_STYLES`/`EVAL_STYLES` partition, `make_training_resampler`, `build_eval_pool` (5×5×2 = 50), `make_dummy_style_template_lists` placeholder filter |
| Env (rollout) | `env.py` | ✅ mostly done | Nutrition5K/history pool sampled at reset; omitted pool means no pre-episode history; selected catalog/observed meals are appended during planning; EDC terminal bonus + report diversity formula are implemented |
| Modular encoder + DQN | `model.py` | ✅ runs | features_dim = 304; spec calls for 112 |
| Baselines | `baseline.py` | ✅ done | HealthGreedy + MultiObjectiveGreedy + RandomPolicy implemented and aligned with env interfaces |
| Training CLI | `train.py` | ✅ done | `--catalog_artifact / --catalog_embeddings / --styles_artifact / --dietary_profile / --randomize_(targets\|preference) / --bootstrap_pool_npy / --nutrition5k-data` wired; bootstrap pool is explicit Nutrition5K/history input |
| Evaluation harness | `evaluate.py` | ✅ done | Python API plus `python -m agent.evaluate`; supports policy tags `dqn`, `health`, `multi`, `random`; reports EDC, not WDC |
| Horizon-sweep driver | `scripts/agent_sweep.sh` | ✅ basic | loops over horizons/seeds, trains DQN, evaluates selected policy tags, and combines aggregate CSV rows |
| Tests | `tests/test_agent_section2*.py` | ✅ done | 7 end-to-end script tests cover the §2/§3/§4 plumbing; pytest is optional and may not be installed in the active env |

---

## §2. RL training-data construction (implemented map)

The milestone PDF (§3.3 *Training & Evaluation*) defines the agent's
episode generator as the combination of: a meal catalog, a user profile
(nutrition targets + preference embedding), bootstrapped recent meals,
the rollout environment, and a held-out evaluation pool. That plumbing
is now implemented and tested. The detailed algorithmic behavior lives
in the code docstrings/tests; this section is only the high-level map.

### Implementation map

| Area | Main symbols | File(s) | What to look for |
|---|---|---|---|
| Catalog contract | `MealTemplate`, `MealCatalog`, `load_dummy`, `load_from_artifact` | `catalog.py` | Stores recipe nutrition + CLIP embedding rows; exposes `embeddings_matrix` and `nutrition_matrix`; dummy catalog runs at 1,000-meal scale; real artifact loader is still a stub. |
| User preference | `SimulatedUser`, `from_templates`, `from_meal_history`, `preference_score` | `user.py` | Builds preference embeddings by mean-pooling meal/template embeddings and L2-normalising; random-init path exists only as a fallback. |
| Episode env | `MealPlanningEnv`, `reset`, `step`, `set_user`, `estimate_observed_meal_from_photo` | `env.py` | Builds the MDP observation/action space, bootstraps Nutrition5K-derived recent meal history at reset, indexes catalog nutrition/embedding in `step`, optionally accepts observed CV nutrition/embedding for demos, and lets evaluation swap users before reset. |
| Training/eval profiles | `TARGET_RANGES`, `NUTRITION_PERSONAS`, `TRAIN_STYLES`, `EVAL_STYLES`, `make_training_resampler`, `build_eval_pool` | `profiles.py` | Samples per-episode training targets/preferences and builds the 50-user held-out grid (`5 personas × 5 eval styles × 2 seeds`). |
| CLI wiring | `build_parser`, `_load_bootstrap_pool`, `_load_nutrition5k_history_pool`, `main` | `train.py` | Re-derives config from the loaded catalog, wires target/preference randomisation, persona pins, explicit Nutrition5K bootstrap pools, and dummy style partitions. |
| Evaluation | `EpisodeResult`, `evaluate_per_episode`, `evaluate_policy`, `compare_policies`, `main` | `evaluate.py` | Runs policies over either repeated seeds or a fixed `EvalUserSpec` sequence; CLI loads a run and writes aggregate metrics CSV. |
| Tests | `test_agent_section2*.py` | `tests/` | Scenario-style end-to-end tests covering catalog/user/env/profile/train/eval plumbing; currently 7 passing under `~/.venv/dl/bin/python`. |

### Data/artifact interfaces

The real catalog artifact is the action space. It should provide one
row per candidate recipe/dish with `dish_name`, one or more source
images or source text, pre-generated CLIP embedding, and nutrition
metadata (`calories`, `protein`, `carbs`, `fat`, optional `mass`).
The embedding array must be row-aligned with the manifest because
`env.step(action)` decodes the action to `(meal_idx, portion)` and
then indexes `catalog.get_embedding(meal_idx)` and
`catalog.get_nutrition(meal_idx, portion)`. `MealCatalog.load_from_artifact(...)`
is the single integration point once the manifest/embedding format is
frozen.

The dietary-style artifact should provide
`dict[str, list[MealTemplate]]` keyed by style name. Training uses the
union of `TRAIN_STYLES`; evaluation uses only `EVAL_STYLES`. The dummy
path in `make_dummy_style_template_lists(...)` enforces disjoint style
buckets so downstream code already matches the intended contract.

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

### Deviations from the milestone description

- The milestone expects RecipeNLG-derived catalog embeddings; current
  agent code uses `MealCatalog.load_dummy(...)` until
  `load_from_artifact(...)` is implemented.
- The milestone says user preferences are initialized by keyword/style
  matching against the recipe catalog; current code uses dummy disjoint
  style partitions unless a real style artifact is supplied.
- The milestone says past meals are bootstrapped from Nutrition5K; the
  code now follows that contract with explicit `--nutrition5k-data` or
  `--bootstrap_pool_npy` history inputs for pre-episode meals. If
  omitted, no prior meals are bootstrapped.
- The 50 held-out episodes are implemented as a structured
  persona × held-out-style × seed grid. This is stricter than the PDF,
  which only says "50 held-out episodes"; it is intentional because it
  enables per-persona/per-style breakdowns.
- Reward/metric deviations are not part of the data-construction layer;
  they are tracked in §4.

---

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

This means the teammate-facing requirement for the catalog is simple:
the manifest and embedding array must be row-aligned, and every row
must have both nutrition metadata and a CLIP embedding available before
agent training starts. The agent should not run CV inference inside
the SB3 rollout loop.

The env-side demo hook is now wired:

- Simulation-only training/eval can keep calling `env.step(action)`.
- A future real-photo demo can call
  `env.step(action, observed_nutrition=..., observed_embedding=...)`.
  Those observed values replace the catalog values for deficit updates
  and recent-meal history.
- `estimate_observed_meal_from_photo(photo_path)` is a placeholder
  function that currently raises `NotImplementedError`. The TODO for
  the CV teammate/integration layer is to implement that function or
  replace it with an equivalent API returning
  `(nutrition[4], embedding[embedding_dim])` in the same CLIP space as
  the catalog.

---

## §4. Reward / state alignment with the spec (implemented)

The report reward formula is now the implementation target:
`r_t = w₁Δhealth + w₂Δdiv + w₃Δpref + w₄b_boundary`.

Implemented pieces:

- ✅ Health: `(old_|deficit|₁ − new_|deficit|₁) / target_sum`. Uses
  `np.abs(deficit)` so overshoot is penalised.
- ✅ Diversity: `1 − cos(selected_embedding, mean_recent_embedding)`.
  The recent mean is a normalised centroid over the sliding
  `history_len` window, not every meal ever seen. `_build_obs` exposes
  that normalised recent mean, and
  `MultiObjectiveGreedy` consumes that same representation.
- ✅ Preference: cosine similarity, with user noise during rollout and
  deterministic dot products in evaluation.
- ✅ Per-day boundary bonus: fires at each last meal of day.
- ✅ Episode terminal bonus: tracks `_episode_deficit = daily_target ×
  num_days − selected_nutrition_sum` and rewards closure at episode end.
- ✅ Evaluation metric: `EDC = 1 − |_episode_deficit|₁ /
  |daily_target × num_days|₁`. `WDC` has been removed because it does
  not match the milestone metric or non-7-day horizon sweep.

End-to-end test coverage:

- `test_end_to_end_history_bootstrap_observed_feedback_and_rollout`
  covers Nutrition5K bootstrap history, observed-meal feedback, and
  catalog-indexed rollout in one scenario.
- `test_end_to_end_edc_uses_episode_target_not_weekly_target`
  verifies EDC uses the episode target; the old weekly denominator
  would fail this case.

### 4.4 `[P2]` Joint-representation dim mismatch (`model.py:38`)
PDF claims `S_t ∈ R^112`. Current `FEATURES_DIM = 304`
(32+16+128+128):

| Encoder input | Raw dim | Encoded dim |
|---|---:|---:|
| Daily + weekly deficits | 8 | 32 |
| Meal slot one-hot | `meals_per_day` (default 3) | 16 |
| Recent meal embedding | `embedding_dim` (default 512) | 128 |
| User preference embedding | `embedding_dim` (default 512) | 128 |

Raw observation dim with defaults is `8 + 3 + 2×512 = 1035`; encoder
output is `304`. Either shrink encoder widths to match the report, or
update the report to cite 304. **Recommendation: keep 304, edit the
report**; the larger latent is reasonable for two 512-dim CLIP inputs.

---

## §5. Experiments & evaluation (P1)

### 5.1 `[P1]` Baseline policies
Done. `baseline.py` now includes HealthGreedy, MultiObjectiveGreedy,
and RandomPolicy.

### 5.2 `[P1]` Horizon-scaling driver — `scripts/agent_sweep.sh`
Basic driver is implemented.

- For `num_days ∈ {1, 3, 7, 21}`, hold all other hyperparams fixed.
- Train DQN once per horizon/seed.
- Evaluate selected policy tags with `python -m agent.evaluate`.
- Persist one aggregate `eval_metrics.csv` per run and a combined
  `sweep_metrics.csv`.
- TODO: add optional per-episode CSV output if the report needs
  persona/style breakdowns instead of aggregate rows only.
- Optional: matplotlib script for the
  "DQN advantage grows with horizon" line plot.

### 5.3 `[P2]` Logging
- Wire `stable_baselines3.common.logger` with TensorBoard
  (`--tb_log_dir` flag). Log the four reward components separately
  to debug weight tuning.
- Add an SB3 `EvalCallback` that runs `evaluate_policy` every N
  steps on a small held-out user subset and saves the best
  checkpoint.

### 5.4 `[P2]` Reward-weight sweep
The four `w_*` knobs (`config.py:26-29`) currently have hand-set
defaults. Run a small grid (`w_diversity ∈ {0.1, 0.3, 0.6}`,
`w_preference ∈ {0.1, 0.2, 0.4}`) driven from a YAML/JSON file so
the report has a defensible weight-selection story.

---

## §6. Algorithmic upgrades (P2)

### 6.1 `[P2]` Double DQN
`model.py:6-12` already calls this out. SB3 doesn't ship Double DQN
out of the box; either subclass `DQN` and override the loss, or swap
to `sb3-contrib`. Worth doing only after the baseline DQN results
are locked in, so we can A/B against them in the report.

### 6.2 `[P2]` Action masking
At inference, some actions are obviously bad (e.g. picking the same
meal back-to-back, or a meal whose calories exceed remaining daily
deficit by 3×). A small mask in `predict` would help the greedy
baselines and could be ported to DQN via `sb3-contrib`'s
`MaskablePPO`-style API.

### 6.3 `[P2]` CLIP-drift contrastive re-anchor (PDF §5.2 mitigation)
If the CV team eventually fine-tunes CLIP, the agent's action
embeddings (recipe text) and observed embeddings (dish images) drift
apart. Track this risk; if it materialises, contribute a contrastive
re-anchoring step on recipe text–image pairs and re-extract the
catalog.

---

## §7. Hygiene (P2)

### 7.1 `[P2]` Keep tests scenario-level
The §2/§3/§4 plumbing tests now exist and pass. Avoid adding one test
per helper function unless a helper has caused a regression. New tests
should stay end-to-end and cover user-visible contracts:

- Catalog artifact integration: tiny real-format fixture once
  `MealCatalog.load_from_artifact(...)` is implemented.
- Horizon sweep: tiny dummy run for `{1, 3, 7, 21}` that asserts the
  output CSV schema and policy columns.
- Evaluate-as-CLI: load `config.json` + model artifact and print the
  comparison table.
- Optional photo-feedback demo path: only after the external CV API is
  wired behind §3's placeholder.

### 7.2 `[P2]` Evaluation ergonomics
`python -m agent.evaluate --run_dir runs/exp1` now loads
`config.json` + `dqn_model.zip`, prints the comparison table, and
writes aggregate metrics CSV. Remaining polish: add real-catalog/style
artifact loaders once those formats land, plus optional per-episode CSV.

---

## §8. Current TODO list / suggested work order

- ✅ **Done:** the §2 training-data plumbing: dummy catalog at
  1,000-meal scale, preference construction, user-history bootstrap,
  profile resampling, held-out eval pool, train CLI wiring, eval
  metadata, optional observed-meal feedback, §4 reward/metric alignment,
  and 7 end-to-end direct-script tests.
  `train.py` runs end-to-end on
  the dummy pipeline.

**P0 integration blockers:**
- Real catalog loader: implement `MealCatalog.load_from_artifact(...)`
  once the manifest/embedding format lands.
- Real style-artifact loader: replace `make_dummy_style_template_lists`
  in `train.py` when the dietary-style filter output lands.
- Real Nutrition5K/history bootstrap pool: pass `--nutrition5k-data`
  pointing at the teammate's embedding split, or `--bootstrap_pool_npy`
  for prepacked experiments, instead of relying on no-history episodes.

**P1 experiment blockers:**
- Per-episode CSV output if the report needs persona/style breakdowns.
- Real catalog/style loaders for final artifact-based experiments.

**P2 polish / optional upgrades:**
- Logging + EvalCallback (§5.3), reward-weight sweep (§5.4),
  Double DQN (§6.1), action masking (§6.2), CLIP-drift mitigation
  (§6.3), external photo-to-observed-meal CV API behind §3's placeholder,
  scenario-level tests for new integrations (§7.1), and evaluation
  ergonomics (§7.2).

1. **Next:** run a short smoke sweep with tiny timesteps to validate the
   full train/evaluate shell workflow on the local machine.
2. **Day 2–4:** use §5.2 horizon sweep to produce the §4.2 table. Headline
   deliverable.
3. **Day 5+:** Swap dummy catalog for the real artifact via
   `MealCatalog.load_from_artifact` (§2) — should be a one-line
   change at the CLI if the contract is honoured. Same for the
   per-style template lists feeding `build_eval_pool` and the
   optional `--nutrition5k-data` / `--bootstrap_pool_npy`
   real-history pool.
4. **Buffer:** external photo-to-observed-meal CV API if the demo
   requires post-recommendation photos, §5.3 logging, §7 hygiene.
5. **If time permits:** §6.1 Double DQN.
