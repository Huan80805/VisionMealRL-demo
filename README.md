# VisionMealRL

VisionMealRL is an interactive meal-planning project that uses reinforcement
learning to recommend meals over a user-selected planning horizon. The user
sets daily nutrition goals, chooses example meals to define preference, and the
agent recommends meal templates and portion sizes while tracking nutrition
deficit, preference alignment, and diversity.

The current demo is built around a DQN meal-planning agent with a
three-component meal representation:

```text
ingredient representation + cuisine representation + recipe-name representation
```

The frontend is a React/Vite demo, the backend is a lightweight Python HTTP
service, and the model runtime loads trained Stable-Baselines3 DQN checkpoints.

## What The Demo Does

1. User enters a planning horizon from 1 to 21 days.
2. User enters daily calories, protein, carbs, and fat goals.
3. User selects several example meal templates to define taste preference.
4. For each breakfast/lunch/dinner step, the backend returns 3-4 ranked meal
   options with image, portion, nutrition, and projected deficit.
5. The UI updates live nutrition status, selected meal history, and final plan
   completion metrics.

Custom horizons are supported by mapping the requested length to the nearest
trained checkpoint greater than or equal to that length:

```text
1 -> 1-day model
2-3 -> 3-day model
4-7 -> 7-day model
8-21 -> 21-day model
```

## Functionality Map

```text
src/agent/
  RL environment, user model, catalog loader, reward logic, DQN model,
  training, evaluation, and demo backend.

frontend/
  React/Vite web app for the interactive demo. Includes a Vercel serverless
  proxy so private Hugging Face Spaces can be called without exposing HF_TOKEN.

deploy/huggingface_space/
  Docker Space files and deployment instructions for the Python demo backend.

scripts/
  Utility scripts for training sweeps, catalog artifact generation, catalog
  image download, and demo-image compression.

src/meal_catalog/
  Older meal-catalog preparation utilities. The current deploy path uses the
  script-level utilities in scripts/.

src/visionmealrl/
  Vision/model-training utilities from earlier project stages. Kept for
  reproducibility, but not required to understand or run the current demo.
```

## Key Runtime Artifacts

The active backend expects:

```text
runs/agent_three_component_2/
artifacts/catalog/three_component/train/
artifacts/catalog_demo_images/
```

Important files:

- `runs/agent_three_component_2/h{1,3,7,21}_seed42/dqn_model.zip`
- `runs/agent_three_component_2/h{1,3,7,21}_seed42/config.json`
- `artifacts/catalog/three_component/train/catalog_manifest.csv`
- `artifacts/catalog/three_component/train/ingredient_embeddings.npy`
- `artifacts/catalog/three_component/train/cuisine_embeddings.npy`
- `artifacts/catalog/three_component/train/name_embeddings.npy`
- `artifacts/catalog_demo_images/*.webp`

The trained agent run, current catalog artifacts, and text-embedding checkpoint
are tracked in the project repo. The large source catalog images and compressed
demo images are not tracked and can be regenerated from `data/meal_catalog.csv`.

## Setup

Recommended project Python for local work:

```bash
~/.venv/dl/bin/python
```

Install dependencies with your preferred environment manager, then run commands
from the repository root with:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl ~/.venv/dl/bin/python ...
```

For frontend dependencies:

```bash
cd frontend
npm install
```

## Run The Backend Locally

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl ~/.venv/dl/bin/python -m agent.demo_backend \
  --host 127.0.0.1 \
  --port 8000 \
  --run-root runs/agent_three_component_2 \
  --catalog-dir artifacts/catalog/three_component/train \
  --image-dir artifacts/catalog_demo_images
```

Smoke checks:

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/horizons
curl "http://127.0.0.1:8000/api/preference-templates?limit=8"
```

## Run The Frontend Locally

```bash
cd frontend
npm run dev
```

By default, the Vite dev server proxies `/agent-api/*` to
`http://127.0.0.1:8000`.

To test against a private Hugging Face Space locally, set these in `.env` or
`frontend/.env.local`:

```text
HF_SPACE_URL=https://your-space-name.hf.space
HF_TOKEN=hf_your_private_space_token
```

Do not put `HF_TOKEN` in a browser-exposed `VITE_*` variable.

## Train And Evaluate Agents

The detailed agent workflow is documented in `src/agent/README.md`.

Typical training entry point:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl ~/.venv/dl/bin/python -m agent.train \
  --num_days 7 \
  --total_timesteps 200000 \
  --catalog_dir artifacts/catalog/three_component/train \
  --output_dir runs/example_agent
```

Typical evaluation entry point:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl ~/.venv/dl/bin/python -m agent.evaluate \
  --run_dir runs/example_agent \
  --catalog_dir artifacts/catalog/three_component/train \
  --policies dqn health multi random \
  --output_csv runs/example_agent/eval_metrics.csv
```

Horizon sweeps:

```bash
TOTAL_TIMESTEPS=200000 \
HORIZONS="1 3 7 21" \
SEEDS="42" \
scripts/agent_sweep.sh
```

## Regenerate Catalog Images And Demo Images

If a fresh clone has `data/meal_catalog.csv` but not `data/catalog/images`,
download source images:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl ~/.venv/dl/bin/python \
  scripts/download_catalog_images.py \
  --recipe-file data/meal_catalog.csv \
  --image-root data/catalog/images
```

If the three-component catalog needs to be rebuilt:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl ~/.venv/dl/bin/python \
  scripts/build_agent_catalog_artifacts.py \
  --recipe-file data/meal_catalog.csv \
  --checkpoint-path artifacts/multitask/open_clip_ViT-B-32_laion2b_s34b_b79k_overhead_rgb/best_embedding_model.pt \
  --image-root data/catalog/images \
  --output-dir artifacts/catalog/three_component/train
```

Compress images for the demo backend:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl ~/.venv/dl/bin/python \
  scripts/prepare_demo_images.py \
  --manifest artifacts/catalog/three_component/train/catalog_manifest.csv \
  --repo-root . \
  --output-dir artifacts/catalog_demo_images \
  --format webp
```

## Deployment

Backend deployment to Hugging Face Spaces is documented in:

```text
deploy/huggingface_space/README.md
```

Frontend deployment to Vercel is documented in:

```text
frontend/README.md
```

The intended deployment split is:

- Hugging Face Space hosts the Python backend and model artifacts.
- Vercel hosts the React frontend.
- Vercel stores `HF_SPACE_URL` and `HF_TOKEN` as server-side environment
  variables and proxies browser requests through `/agent-api`.

## Current Status

The current maintained path is the agent demo:

- three-component catalog runtime,
- action-scoring DQN checkpoints for horizons 1, 3, 7, and 21,
- custom demo horizon support up to 21 days,
- template-based user preference setup,
- local backend/frontend preview,
- Hugging Face backend deployment path,
- Vercel frontend deployment path.

Older Nutrition5K and vision-training code remains in the repository for
project history and reproducibility, but the root README intentionally points
new users toward the current meal-planning demo workflow.
