from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from visionmealrl.multitask.model import MultiTaskNutritionModel


def compute_multitask_loss(
    regression_predictions: torch.Tensor,
    regression_targets: torch.Tensor,
    classification_logits: torch.Tensor,
    classification_targets: torch.Tensor,
    lambda_reg: float,
    lambda_cls: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    regression_loss = nn.functional.mse_loss(regression_predictions, regression_targets)
    classification_loss = nn.functional.binary_cross_entropy_with_logits(
        classification_logits,
        classification_targets,
    )
    total_loss = lambda_reg * regression_loss + lambda_cls * classification_loss
    return total_loss, regression_loss, classification_loss


def train_multitask_epoch(
    model: MultiTaskNutritionModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    lambda_reg: float,
    lambda_cls: float,
) -> dict[str, float]:
    model.train(True)
    autocast_enabled = device == "cuda"
    autocast_device = "cuda" if device == "cuda" else "cpu"

    total_loss_sum = 0.0
    regression_loss_sum = 0.0
    classification_loss_sum = 0.0
    total_examples = 0

    for batch in dataloader:
        images = batch.images.to(device)
        regression_targets = batch.regression_targets.to(device)
        classification_targets = batch.classification_targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled):
            _dish_embeddings, regression_predictions, classification_logits = model(images, batch.view_counts)
            total_loss, regression_loss, classification_loss = compute_multitask_loss(
                regression_predictions=regression_predictions,
                regression_targets=regression_targets,
                classification_logits=classification_logits,
                classification_targets=classification_targets,
                lambda_reg=lambda_reg,
                lambda_cls=lambda_cls,
            )
        total_loss.backward()
        optimizer.step()

        batch_size = regression_targets.shape[0]
        total_loss_sum += float(total_loss.item()) * batch_size
        regression_loss_sum += float(regression_loss.item()) * batch_size
        classification_loss_sum += float(classification_loss.item()) * batch_size
        total_examples += batch_size

    return {
        "total_loss": total_loss_sum / max(total_examples, 1),
        "regression_loss": regression_loss_sum / max(total_examples, 1),
        "classification_loss": classification_loss_sum / max(total_examples, 1),
    }


def evaluate_multitask(
    model: MultiTaskNutritionModel,
    dataloader: DataLoader,
    device: str,
    lambda_reg: float,
    lambda_cls: float,
) -> dict[str, object]:
    model.train(False)
    autocast_enabled = device == "cuda"
    autocast_device = "cuda" if device == "cuda" else "cpu"

    total_loss_sum = 0.0
    regression_loss_sum = 0.0
    classification_loss_sum = 0.0
    total_examples = 0

    dish_ids: list[str] = []
    regression_predictions: list[np.ndarray] = []
    regression_targets: list[np.ndarray] = []
    classification_logits: list[np.ndarray] = []
    classification_targets: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in dataloader:
            images = batch.images.to(device)
            batch_regression_targets = batch.regression_targets.to(device)
            batch_classification_targets = batch.classification_targets.to(device)

            with torch.autocast(device_type=autocast_device, enabled=autocast_enabled):
                _dish_embeddings, batch_regression_predictions, batch_classification_logits = model(
                    images,
                    batch.view_counts,
                )
                total_loss, regression_loss, classification_loss = compute_multitask_loss(
                    regression_predictions=batch_regression_predictions,
                    regression_targets=batch_regression_targets,
                    classification_logits=batch_classification_logits,
                    classification_targets=batch_classification_targets,
                    lambda_reg=lambda_reg,
                    lambda_cls=lambda_cls,
                )

            batch_size = batch_regression_targets.shape[0]
            total_loss_sum += float(total_loss.item()) * batch_size
            regression_loss_sum += float(regression_loss.item()) * batch_size
            classification_loss_sum += float(classification_loss.item()) * batch_size
            total_examples += batch_size

            dish_ids.extend(batch.dish_ids)
            regression_predictions.append(batch_regression_predictions.detach().cpu().numpy().astype(np.float32))
            regression_targets.append(batch.regression_targets.numpy().astype(np.float32))
            classification_logits.append(batch_classification_logits.detach().cpu().numpy().astype(np.float32))
            classification_targets.append(batch.classification_targets.numpy().astype(np.float32))

    logits = np.concatenate(classification_logits, axis=0) if classification_logits else np.zeros((0, 0), dtype=np.float32)
    return {
        "dish_ids": dish_ids,
        "total_loss": total_loss_sum / max(total_examples, 1),
        "regression_loss": regression_loss_sum / max(total_examples, 1),
        "classification_loss": classification_loss_sum / max(total_examples, 1),
        "regression_predictions": np.concatenate(regression_predictions, axis=0).astype(np.float32),
        "regression_targets": np.concatenate(regression_targets, axis=0).astype(np.float32),
        "classification_logits": logits,
        "classification_probabilities": (1.0 / (1.0 + np.exp(-logits))).astype(np.float32),
        "classification_targets": np.concatenate(classification_targets, axis=0).astype(np.float32),
    }
