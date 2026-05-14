---
title: VisionMealRL Agent Demo
emoji: 🍽️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# VisionMealRL Agent Demo

Docker Space for the interactive meal-planning backend.

Expected files in this Space repository:

- `Dockerfile`
- `requirements.txt`
- `.dockerignore`
- `README.md`
- `src/agent/**`
- `runs/agent_three_component_2/h1_seed42/{config.json,dqn_model.zip}`
- `runs/agent_three_component_2/h3_seed42/{config.json,dqn_model.zip}`
- `runs/agent_three_component_2/h7_seed42/{config.json,dqn_model.zip}`
- `runs/agent_three_component_2/h21_seed42/{config.json,dqn_model.zip}`
- `artifacts/catalog/three_component/train/catalog_manifest.csv`
- `artifacts/catalog/three_component/train/ingredient_embeddings.npy`
- `artifacts/catalog/three_component/train/cuisine_embeddings.npy`
- `artifacts/catalog/three_component/train/name_embeddings.npy`
- `artifacts/catalog/three_component/train/metadata.json`
- `artifacts/catalog_demo_images/*.webp`

The app serves JSON endpoints under `/api/*`.

Main demo endpoints:

- `GET /api/health`
- `GET /api/horizons`
- `GET /api/preference-templates?limit=128`
- `GET /api/meal-image/{meal_idx}`
- `POST /api/session`
- `POST /api/session/{session_id}/select`

`num_days` may be any integer from 1 to 21. The backend maps custom horizons to
the nearest trained checkpoint greater than or equal to the requested length.

The Dockerfile launches the backend explicitly with the current three-component
runtime:

```bash
python -m agent.demo_backend \
  --host 0.0.0.0 \
  --port 7860 \
  --run-root runs/agent_three_component_2 \
  --catalog-dir artifacts/catalog/three_component/train \
  --image-dir artifacts/catalog_demo_images
```

The runtime loads the three component matrices from the catalog directory and
concatenates them internally to the model embedding dimension.

Session creation should send `preference_meal_indices` when the user chooses
example meals to define preference:

```json
{
  "num_days": 7,
  "nutrition_goals": {
    "calories": 2000,
    "protein": 100,
    "carbs": 240,
    "fat": 65
  },
  "preference_meal_indices": [4, 12, 31, 48, 77],
  "top_k": 4
}
```

The backend builds the user preference with `SimulatedUser.from_templates(...)`,
which mean-pools the selected templates' ingredient, cuisine, and recipe-name
components separately before concatenating them for the policy.

## Generate Runtime Artifacts From A Fresh Repo

If a fresh deployment repo only contains `data/meal_catalog.csv`, regenerate
the runtime catalog and demo images before copying files to the Hugging Face
Space.

If your deployment clone already has these tracked paths, keep them and skip
the regeneration step unless `data/meal_catalog.csv` changed:

- `artifacts/multitask/open_clip_ViT-B-32_laion2b_s34b_b79k_overhead_rgb/best_embedding_model.pt`
- `artifacts/catalog/three_component/train/*`
- `runs/agent_three_component_2/h{1,3,7,21}_seed42/{config.json,dqn_model.zip}`

If your mirror or slim repo omitted the tracked artifacts, restore the
checkpoint and trained run files before building the catalog.

### Download Source Catalog Images

Download those source images using `data/meal_catalog.csv` first:

```bash
PYTHONPATH=src ~/.venv/dl/bin/python \
  scripts/download_catalog_images.py \
  --recipe-file data/meal_catalog.csv \
  --image-root data/catalog/images \
  --image-width 1920 \
  --image-height 1080
```

This writes images in the layout expected by the catalog build:

```text
data/catalog/images/catalog_0/rgb.png
data/catalog/images/catalog_1/rgb.png
...
```

Then build the three-component catalog from `meal_catalog.csv`:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl ~/.venv/dl/bin/python \
  scripts/build_agent_catalog_artifacts.py \
  --recipe-file data/meal_catalog.csv \
  --checkpoint-path artifacts/multitask/open_clip_ViT-B-32_laion2b_s34b_b79k_overhead_rgb/best_embedding_model.pt \
  --image-root data/catalog/images \
  --output-dir artifacts/catalog/three_component/train
```

This writes:

- `artifacts/catalog/three_component/train/catalog_manifest.csv`
- `artifacts/catalog/three_component/train/ingredient_embeddings.npy`
- `artifacts/catalog/three_component/train/cuisine_embeddings.npy`
- `artifacts/catalog/three_component/train/name_embeddings.npy`
- `artifacts/catalog/three_component/train/metadata.json`

Next compress demo images from the manifest image paths:

```bash
PYTHONPATH=src ~/.venv/dl/bin/python \
  scripts/prepare_demo_images.py \
  --manifest artifacts/catalog/three_component/train/catalog_manifest.csv \
  --repo-root . \
  --output-dir artifacts/catalog_demo_images \
  --size 512 \
  --quality 72 \
  --format webp
```

The backend serves these compressed files from
`/api/meal-image/{meal_idx}`. The filenames are keyed by `catalog_id`, so keep
the generated manifest and image directory from the same catalog build.

Optional local smoke test:

```bash
PYTHONPATH=src ~/.venv/dl/bin/python -m agent.demo_backend \
  --host 127.0.0.1 \
  --port 8000 \
  --run-root runs/agent_three_component_2 \
  --catalog-dir artifacts/catalog/three_component/train \
  --image-dir artifacts/catalog_demo_images

curl http://127.0.0.1:8000/api/health
curl "http://127.0.0.1:8000/api/preference-templates?limit=8"
```

## Copy From This Repository

If the Hugging Face Space lives in a separate folder, copy these paths from
the project root into the Space repository root:

```bash
cp deploy/huggingface_space/Dockerfile /path/to/space/Dockerfile
cp deploy/huggingface_space/requirements.txt /path/to/space/requirements.txt
cp deploy/huggingface_space/README.md /path/to/space/README.md
cp deploy/huggingface_space/.dockerignore /path/to/space/.dockerignore

rsync -a --delete src/agent/ /path/to/space/src/agent/
rsync -a --delete runs/agent_three_component_2/ /path/to/space/runs/agent_three_component_2/
rsync -a --delete artifacts/catalog/three_component/train/ /path/to/space/artifacts/catalog/three_component/train/
rsync -a --delete artifacts/catalog_demo_images/ /path/to/space/artifacts/catalog_demo_images/
```

Do not copy old run folders, old catalog folders, `frontend/dist`, or
`frontend/node_modules`.
