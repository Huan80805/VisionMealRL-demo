from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union


import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from visionmealrl.constants import TARGET_COLUMNS
from visionmealrl.logging_utils import configure_logging
from visionmealrl.nutrition5k import (
    DishRecord,
    ImageRecord,
    build_dish_records_by_split,
    image_records_from_dish_records,
)

LOGGER = logging.getLogger(__name__)


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def slugify_model_name(model_name: str, pretrained: str) -> str:
    safe_model = model_name.replace("/", "-")
    safe_pretrained = pretrained.replace("/", "-")
    return f"open_clip_{safe_model}_{safe_pretrained}"


def load_openclip_model_and_preprocess(model_name: str, pretrained: str, device: str):
    LOGGER.info(
        "Loading OpenCLIP model %s with weights %s on %s",
        model_name,
        pretrained,
        device,
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    model.eval()
    model.to(device)
    return model, preprocess


# when indexed each element is (tensor, ImageRecord)
class NutritionImageDataset(Dataset[tuple[torch.Tensor, ImageRecord]]):
    def __init__(self, records: Sequence[ImageRecord], preprocess) -> None:
        self.records = list(records)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ImageRecord]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            image_tensor = self.preprocess(image.convert("RGB"))
        # image_tensor: representation for a SINGLE image of a dish
        # record: ImageRecord, so we can know dish_id and dish_data
        return image_tensor, record


# batch process list of (image_tensor, record)
def collate_batch(batch):
    images, records = zip(*batch)
    return torch.stack(list(images), dim=0), list(records)


# a batch of image embeddings + image data
@dataclass
class BatchExtractionResult:
    embeddings: np.ndarray
    dish_ids: list[str]
    image_paths: list[str]
    # flattened fields in DishData
    targets: np.ndarray


def extract_embeddings_in_batches(
    records: Sequence[ImageRecord],
    model,
    preprocess,
    batch_size: int,
    num_workers: int,
    device: str,
) -> BatchExtractionResult:
    dataset = NutritionImageDataset(records, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        collate_fn=collate_batch,
    )

    all_embeddings: list[np.ndarray] = []
    dish_ids: list[str] = []
    image_paths: list[str] = []
    targets: list[list[float]] = []

    autocast_enabled = device == "cuda"
    autocast_device = "cuda" if device == "cuda" else "cpu"

    with torch.inference_mode():
        for images, batch_records in tqdm(dataloader, desc="embedding", leave=False):
            images = images.to(device)
            with torch.autocast(device_type=autocast_device, enabled=autocast_enabled):
                features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)

            features_np = features.detach().cpu().numpy().astype(np.float32)
            all_embeddings.append(features_np)
            dish_ids.extend(record.dish_id for record in batch_records)
            image_paths.extend(str(record.image_path) for record in batch_records)
            targets.extend(record.targets.as_list() for record in batch_records)

    if not all_embeddings:
        raise ValueError("No embeddings were extracted. Check dataset paths and image source.")

    return BatchExtractionResult(
        embeddings=np.concatenate(all_embeddings, axis=0),
        dish_ids=dish_ids,
        image_paths=image_paths,
        targets=np.asarray(targets, dtype=np.float32),
    )


# multiple images for a dish -> one single dish
# embeddings are mean-pooled
def aggregate_dish_embeddings(artifacts: BatchExtractionResult) -> tuple[np.ndarray, List[Dict[str, Union[str, float, int]]]]:
    grouped_rows: dict[str, list[int]] = defaultdict(list)
    for idx, dish_id in enumerate(artifacts.dish_ids):
        grouped_rows[dish_id].append(idx)

    dish_embeddings: list[np.ndarray] = []
    dish_manifest: List[Dict[str, Union[str, float, int]]] = []

    for dish_id in sorted(grouped_rows):
        indices = grouped_rows[dish_id]
        pooled = artifacts.embeddings[indices].mean(axis=0)
        target_row = artifacts.targets[indices[0]]
        dish_embeddings.append(pooled.astype(np.float32))
        # to be written to the manifest file
        dish_manifest.append(
            {
                "dish_id": dish_id,
                "image_count": len(indices),
                "total_calories": float(target_row[0]),
                "total_mass": float(target_row[1]),
                "total_fat": float(target_row[2]),
                "total_carb": float(target_row[3]),
                "total_protein": float(target_row[4]),
            }
        )

    return np.stack(dish_embeddings, axis=0), dish_manifest


def write_manifest_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    split_output_dir: Path,
    split: str,
    artifacts: BatchExtractionResult,
    dish_embeddings: np.ndarray,
    dish_manifest: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    split_output_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_output_dir / "per_image_embeddings.npy", artifacts.embeddings)
    np.save(split_output_dir / "dish_embeddings.npy", dish_embeddings)

    per_image_rows = []
    for dish_id, image_path, target_row in zip(
        artifacts.dish_ids,
        artifacts.image_paths,
        artifacts.targets,
    ):
        row = {
            "split": split,
            "dish_id": dish_id,
            "image_path": image_path,
        }
        for index, column in enumerate(TARGET_COLUMNS):
            row[column] = float(target_row[index])
        per_image_rows.append(row)

    write_manifest_csv(split_output_dir / "per_image_manifest.csv", per_image_rows)
    write_manifest_csv(split_output_dir / "dish_manifest.csv", dish_manifest)

    with (split_output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def export_dish_embeddings_with_encoder(
    dish_records_by_split: dict[str, list[DishRecord]],
    model,
    preprocess,
    output_base: Path,
    image_source: str,
    batch_size: int,
    num_workers: int,
    device: str,
    metadata_overrides: dict[str, object] | None = None,
) -> None:
    metadata_overrides = metadata_overrides or {}

    for split in ("train", "test"):
        split_dish_records = dish_records_by_split.get(split, [])
        if not split_dish_records:
            LOGGER.warning("Skipping empty split: %s", split)
            continue

        split_records = image_records_from_dish_records(split_dish_records)

        LOGGER.info(
            "Exporting finetuned embeddings for %s split with %d dishes / %d images",
            split,
            len(split_dish_records),
            len(split_records),
        )
        artifacts = extract_embeddings_in_batches(
            records=split_records,
            model=model,
            preprocess=preprocess,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        dish_embeddings, dish_manifest = aggregate_dish_embeddings(artifacts)
        metadata = {
            "split": split,
            "image_source": image_source,
            "embedding_dim": int(artifacts.embeddings.shape[1]),
            "num_images": int(artifacts.embeddings.shape[0]),
            "num_dishes": int(dish_embeddings.shape[0]),
            "device": device,
            **metadata_overrides,
        }
        write_outputs(
            split_output_dir=output_base / split,
            split=split,
            artifacts=artifacts,
            dish_embeddings=dish_embeddings,
            dish_manifest=dish_manifest,
            metadata=metadata,
        )
        LOGGER.info("Wrote %s outputs to %s", split, output_base / split)


def extract_embeddings_main(args) -> None:
    configure_logging()

    model_dir_name = slugify_model_name(args.model_name, args.pretrained)
    output_base = args.output_root / "embeddings" / model_dir_name / args.image_source
    device = resolve_device(args.device)

    LOGGER.info("Loading Nutrition5K records from %s", args.dataset_root)

    model, preprocess = load_openclip_model_and_preprocess(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=device,
    )
    dish_records_by_split = build_dish_records_by_split(args.dataset_root, args.image_source)
    export_dish_embeddings_with_encoder(
        dish_records_by_split=dish_records_by_split,
        model=model,
        preprocess=preprocess,
        output_base=output_base,
        image_source=args.image_source,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        metadata_overrides={
            "model_name": args.model_name,
            "pretrained": args.pretrained,
        },
    )
