# VisionMealRL

CLIP embedding extraction, ingredient classification, and nutrition regression for the official Nutrition5K dataset, plus a DQN-based meal planning agent.

This scaffold is aligned to the current maintained OpenCLIP package and the official Nutrition5K dataset layout:

- `open-clip-torch==3.3.0` on PyPI, released February 27, 2026
- Nutrition5K official dataset repo from Google Research (`google-research-datasets/Nutrition5k`), with 5,006 plates, official train/test splits, and dish-level nutrition metadata

## Project layout

```text
VisionMealRL/
├── configs/                  ← example TOML configs for visionmealrl CLI
├── scripts/
│   └── download_nutrition5k.sh
├── src/
│   ├── visionmealrl/         ← CV pipeline (CLIP extraction, regression, classification)
│   └── agent/                ← DQN meal planning agent
├── tests/
├── README.md
└── pyproject.toml
```

## Install

The project uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# From the VisionMealRL/ directory — creates .venv and installs all deps
uv sync
```

For GPU use, install PyTorch manually first so the wheel matches your CUDA setup, then run `uv sync`.

Alternatively with plain pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Download Nutrition5K

Use the helper script in [scripts/download_nutrition5k.sh](/Users/Jason-Yu/Desktop/GT/CS7643/VisionMealRL/scripts/download_nutrition5k.sh):

```bash
chmod +x scripts/download_nutrition5k.sh
scripts/download_nutrition5k.sh --output-dir data
```

For the full dataset, `gsutil` is the preferred route because the official dataset is published in a public Google Cloud Storage bucket. If you only want part of the dataset, pass `--path`, for example:

```bash
scripts/download_nutrition5k.sh --output-dir data --path metadata
```

If you prefer a single tarball download:

```bash
scripts/download_nutrition5k.sh --method tarball --output-dir data --extract
```

## Nutrition5K layout

The code expects the official Nutrition5K directory structure under a single root:

```text
nutrition5k_dataset/
├── dish_ids/splits/
├── imagery/realsense_overhead/
├── imagery/side_angles/
└── metadata/
```

The loader reads:

- `metadata/dish_metadata_cafe1.csv`
- `metadata/dish_metadata_cafe2.csv`
- split files under `dish_ids/splits/`

## Run the Baseline

This is the frozen-embedding baseline:

- extract L2-normalized CLIP dish embeddings
- train the linear nutrition regressor
- train the linear top-100 ingredient classifier
- write the benchmark summary and all intermediate outputs

```bash
visionmealrl run-baseline \
  --dataset-root /path/to/nutrition5k_dataset \
  --output-root /path/to/artifacts \
  --model-name ViT-B-32 \
  --pretrained laion2b_s34b_b79k \
  --image-source overhead_rgb
```

Outputs are written under:

```text
artifacts/
└── benchmarks/
    ├── benchmark_runs.csv
    └── open_clip_ViT-B-32_laion2b_s34b_b79k/
        └── overhead_rgb/
            └── linear_top100_at5_seed7/
                ├── benchmark_config.json
                ├── benchmark_summary.csv
                ├── benchmark_summary.json
                ├── classification/
                ├── regression/
                └── ...
```

The baseline still writes extracted embeddings in the same split format under
`artifacts/embeddings/...`. Each split contains:

- `per_image_embeddings.npy`
- `per_image_manifest.csv`
- `dish_embeddings.npy`
- `dish_manifest.csv`
- `metadata.json`

`dish_embeddings.npy` stores the dish-level embedding matrix directly as a NumPy
array. Its rows are aligned with `dish_manifest.csv`, which contains the
corresponding `dish_id`, image count, and nutrition targets.

For downstream code that needs the embeddings joined with nutrition metadata:

```python
from visionmealrl import load_dish_embedding_lookup

lookup = load_dish_embedding_lookup("artifacts/multitask/open_clip_ViT-B-32_laion2b_s34b_b79k_overhead_rgb/finetuned_embeddings/train")

dish_id = next(iter(lookup))
record = lookup[dish_id]
embedding = record["embedding"]
nutrition_metadata = record["nutrition_metadata"]
```

## Run the Multitask Pipeline

This command trains a shared CLIP visual encoder jointly for:

- nutrition regression (`total_calories`, `total_mass`, `total_fat`, `total_carb`, `total_protein`)
- ingredient multi-label classification

It starts by training the task heads with the encoder frozen, then selectively
unfreezes the last visual transformer blocks for end-to-end finetuning.

```bash
visionmealrl run-multitask \
  --dataset-root /path/to/nutrition5k_dataset \
  --output-root /path/to/artifacts \
  --model-name ViT-B-32 \
  --pretrained laion2b_s34b_b79k \
  --image-source overhead_rgb \
  --epochs 20 \
  --freeze-epochs 3 \
  --unfreeze-last-n-blocks 2
```

Outputs are written under:

```text
artifacts/
└── multitask/
    └── open_clip_ViT-B-32_laion2b_s34b_b79k_overhead_rgb/
        ├── best_model.pt
        ├── label_vocabulary.json
        ├── metrics.json
        ├── run_config.json
        ├── regression_predictions_test.csv
        ├── classification_predictions_test.csv
        ├── classification_per_class_metrics.csv
        └── finetuned_embeddings/
            ├── train/
            └── test/
```

The finetuned embedding export uses the same split artifact format as the
baseline extractor (`dish_embeddings.npy`, `dish_manifest.csv`, `metadata.json`,
plus per-image files), so downstream code can reuse the same loading path.

## Reconstruct Embedding Model From Cached Weights
The exported weights for the embedding model is stored in `artifacts/multitask/open_clip_ViT-B-32_laion2b_s34b_b79k_overhead_rgb/best_embedding_model.pt`.
```
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from visionmealrl.embedding import load_openclip_model_and_preprocess, resolve_device
from visionmealrl.multitask.artifacts import load_checkpoint

checkpoint_path = Path(
    "artifacts/multitask/open_clip_ViT-B-32_laion2b_s34b_b79k_overhead_rgb/best_embedding_model.pt"
)
image_path = Path("/path/to/dish_image.png")

device = resolve_device("auto")
checkpoint = load_checkpoint(checkpoint_path, device=device)

clip_model, preprocess = load_openclip_model_and_preprocess(
    model_name=checkpoint["model_name"],
    pretrained=checkpoint["pretrained"],
    device=device,
)
clip_model.load_state_dict(checkpoint["clip_model_state_dict"])
clip_model.eval()

with Image.open(image_path) as image:
    image_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

with torch.inference_mode():
    embedding = clip_model.encode_image(image_tensor)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp(min=1e-12)

embedding_np = embedding[0].detach().cpu().numpy().astype(np.float32)
print(embedding_np)
```

## DQN meal planning agent

`src/agent/` contains a Gymnasium + Stable-Baselines3 DQN agent that plans a
multi-day meal schedule. It is decoupled from the CV pipeline for now; the two
are wired together through stub methods once real embeddings are available.

### Agent layout

```text
src/agent/
├── config.py      AgentConfig dataclass — all hyperparams, derived properties
├── catalog.py     MealTemplate, MealCatalog (synthetic by default; load_from_file TODO)
├── user.py        SimulatedUser (from_config; from_meal_history TODO)
├── env.py         MealPlanningEnv — Gymnasium env, from_config factory
├── model.py       ModularEncoder (SB3 BaseFeaturesExtractor) + make_dqn()
├── baseline.py    HealthGreedy, MultiObjectiveGreedy
├── train.py       argparse CLI entrypoint
└── evaluate.py    5 evaluation metrics + comparison table printer
```

### Train

```bash
# Quick smoke test (embedding_dim=32 finishes in seconds)
uv run python -m agent.train \
  --num_days 1 --embedding_dim 32 --total_timesteps 500 \
  --output_dir runs/smoke

# Experiment 1: 3-day planning horizon
uv run python -m agent.train \
  --num_days 3 --total_timesteps 200000 --seed 42 \
  --output_dir runs/exp1

# Experiment 2: horizon scaling sweep
uv run python -m agent.train --num_days 7 --total_timesteps 500000 \
  --output_dir runs/exp2_7days
```

Each run saves `dqn_model.zip` and `config.json` to `--output_dir`.

All flags: `--num_days`, `--meals_per_day`, `--embedding_dim`,
`--total_timesteps`, `--seed`, `--output_dir` (required),
`--w_health`, `--w_diversity`, `--w_preference`, `--w_boundary`.

### Evaluate

```python
from agent import (AgentConfig, MealCatalog, SimulatedUser, MealPlanningEnv,
                   ModularEncoder, HealthGreedy, MultiObjectiveGreedy,
                   compare_policies, print_comparison_table)
from stable_baselines3 import DQN

cfg     = AgentConfig.from_json("runs/exp1/config.json")
catalog = MealCatalog(cfg.num_meals, cfg.embedding_dim, cfg.seed)
user    = SimulatedUser.from_config(cfg)
env     = MealPlanningEnv.from_config(cfg, catalog, user)
model   = DQN.load("runs/exp1/dqn_model",
                   custom_objects={"features_extractor_class": ModularEncoder})

results = compare_policies(
    {"DQN": model,
     "HealthGreedy": HealthGreedy(env),
     "MultiObjGreedy": MultiObjectiveGreedy(env, cfg)},
    env, cfg, n_episodes=50,
)
print_comparison_table(results)
```

Five metrics are reported (mean ± std over episodes):

| Metric | Description |
|--------|-------------|
| **NGA** | % of days with daily deficit within 10 % of target |
| **DDS** | Dietary diversity: `1 − mean pairwise cosine sim` of meal embeddings |
| **PA**  | Preference alignment: mean cosine sim to user preference embedding |
| **WDC** | Weekly diet completeness: `1 − L1(weekly_deficit) / weekly_target_sum` |
| **Return** | Sum of step rewards |

### Connecting to real CLIP embeddings

Once `visionmealrl extract-embeddings` has been run, replace the synthetic
catalog and user preference with real data via the stub methods:

```python
# Real meal catalog from extract-embeddings output
catalog = MealCatalog.load_from_file(
    npy_path="artifacts/embeddings/.../train/dish_embeddings.npy",
    manifest_csv="artifacts/embeddings/.../train/dish_manifest.csv",
)

# Real user preference from meal history embeddings
user = SimulatedUser.from_meal_history(meal_embeddings)
```

Both methods currently raise `NotImplementedError` and are the next
integration step.
