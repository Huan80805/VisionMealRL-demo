# VisionMealRL

CLIP embedding extraction and nutrition regression for the official Nutrition5K dataset.

This scaffold is aligned to the current maintained OpenCLIP package and the official Nutrition5K dataset layout:

- `open-clip-torch==3.3.0` on PyPI, released February 27, 2026
- Nutrition5K official dataset repo from Google Research (`google-research-datasets/Nutrition5k`), with 5,006 plates, official train/test splits, and dish-level nutrition metadata

## Project layout

```text
VisionMealRL/
├── configs/
├── src/visionmealrl/
├── tests/
├── README.md
└── pyproject.toml
```

## Install

For GPU use, install PyTorch from the official selector first so the wheel matches your CUDA or platform setup.

Then install this project:

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

## Extract CLIP embeddings

Default mode uses overhead RGB images and mean-pools view embeddings into one embedding per dish.

```bash
visionmealrl extract-embeddings \
  --dataset-root /path/to/nutrition5k_dataset \
  --output-root /path/to/artifacts \
  --model-name ViT-B-32 \
  --pretrained laion2b_s34b_b79k \
  --image-source overhead_rgb
```

Outputs are written under:

```text
artifacts/
└── embeddings/
    └── open_clip_ViT-B-32_laion2b_s34b_b79k/
        └── overhead_rgb/
            ├── train/
            └── test/
```

Each split contains:

- `per_image_embeddings.npy`
- `per_image_manifest.csv`
- `dish_embeddings.npy`
- `dish_manifest.csv`
- `metadata.json`

## Train a regressor (WIP)

```bash
visionmealrl train-regressor \
  --embeddings-root /path/to/artifacts/embeddings/open_clip_ViT-B-32_laion2b_s34b_b79k/overhead_rgb \
  --output-root /path/to/artifacts \
  --head mlp
```

Outputs are written under:

```text
artifacts/
└── regressors/
    └── mlp/
```

Each run includes:

- `best_model.pt`
- `metrics.json`
- `predictions_test.csv`
- `run_config.json`
