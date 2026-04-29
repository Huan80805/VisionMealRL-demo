from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from visionmealrl.nutrition5k import DishRecord


@dataclass
class MultiTaskBatch:
    images: torch.Tensor
    view_counts: list[int]
    dish_ids: list[str]
    regression_targets: torch.Tensor
    classification_targets: torch.Tensor


class NutritionDishDataset(Dataset[tuple[list[torch.Tensor], str, np.ndarray, np.ndarray]]):
    def __init__(
        self,
        dish_records: Sequence[DishRecord],
        preprocess,
        regression_targets: np.ndarray,
        classification_targets: np.ndarray,
    ) -> None:
        if len(dish_records) != len(regression_targets):
            raise ValueError("Dish records and regression targets must have matching lengths.")
        if len(dish_records) != len(classification_targets):
            raise ValueError("Dish records and classification targets must have matching lengths.")

        self.dish_records = list(dish_records)
        self.preprocess = preprocess
        self.regression_targets = regression_targets.astype(np.float32)
        self.classification_targets = classification_targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.dish_records)

    def __getitem__(self, index: int) -> tuple[list[torch.Tensor], str, np.ndarray, np.ndarray]:
        dish_record = self.dish_records[index]
        image_tensors: list[torch.Tensor] = []
        for image_path in dish_record.image_paths:
            with Image.open(image_path) as image:
                image_tensors.append(self.preprocess(image.convert("RGB")))
        if not image_tensors:
            raise ValueError(f"Dish {dish_record.dish_id} has no usable images.")
        return (
            image_tensors,
            dish_record.dish_id,
            self.regression_targets[index],
            self.classification_targets[index],
        )


def collate_dish_batch(
    batch: Sequence[tuple[list[torch.Tensor], str, np.ndarray, np.ndarray]]
) -> MultiTaskBatch:
    flat_images: list[torch.Tensor] = []
    view_counts: list[int] = []
    dish_ids: list[str] = []
    regression_targets: list[np.ndarray] = []
    classification_targets: list[np.ndarray] = []

    for image_tensors, dish_id, regression_target, classification_target in batch:
        flat_images.extend(image_tensors)
        view_counts.append(len(image_tensors))
        dish_ids.append(dish_id)
        regression_targets.append(regression_target)
        classification_targets.append(classification_target)

    return MultiTaskBatch(
        images=torch.stack(flat_images, dim=0),
        view_counts=view_counts,
        dish_ids=dish_ids,
        regression_targets=torch.from_numpy(np.asarray(regression_targets, dtype=np.float32)),
        classification_targets=torch.from_numpy(np.asarray(classification_targets, dtype=np.float32)),
    )


def mean_pool_view_embeddings(image_embeddings: torch.Tensor, view_counts: Sequence[int]) -> torch.Tensor:
    if image_embeddings.ndim != 2:
        raise ValueError(
            f"Expected image embeddings to be rank 2, got shape {tuple(image_embeddings.shape)}"
        )
    if sum(view_counts) != int(image_embeddings.shape[0]):
        raise ValueError(
            "Sum of view counts does not match image embedding rows: "
            f"{sum(view_counts)} != {int(image_embeddings.shape[0])}"
        )

    pooled_embeddings = [
        chunk.mean(dim=0) for chunk in torch.split(image_embeddings, tuple(int(count) for count in view_counts), dim=0)
    ]
    return torch.stack(pooled_embeddings, dim=0)


def build_multitask_dataloader(
    dish_records: Sequence[DishRecord],
    preprocess,
    regression_targets: np.ndarray,
    classification_targets: np.ndarray,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: str,
) -> DataLoader:
    dataset = NutritionDishDataset(
        dish_records=dish_records,
        preprocess=preprocess,
        regression_targets=regression_targets,
        classification_targets=classification_targets,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        collate_fn=collate_dish_batch,
    )
