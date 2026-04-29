# VisionMealRL

CLIP embedding extraction, ingredient classification, and nutrition regression for the official Nutrition5K dataset.

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
