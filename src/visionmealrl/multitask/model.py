from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from visionmealrl.constants import TARGET_COLUMNS
from visionmealrl.multitask.data import mean_pool_view_embeddings


class MultiTaskNutritionModel(nn.Module):
    def __init__(
        self,
        clip_model,
        embedding_dim: int,
        num_labels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.regression_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, len(TARGET_COLUMNS)),
        )
        self.classification_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_labels),
        )

    def encode_dishes(self, images: torch.Tensor, view_counts: Sequence[int]) -> torch.Tensor:
        image_embeddings = self.clip_model.encode_image(images)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return mean_pool_view_embeddings(image_embeddings, view_counts)

    def forward(self, images: torch.Tensor, view_counts: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dish_embeddings = self.encode_dishes(images, view_counts)
        regression_predictions = self.regression_head(dish_embeddings)
        classification_logits = self.classification_head(dish_embeddings)
        return dish_embeddings, regression_predictions, classification_logits


def set_trainable_visual_layers(
    model: MultiTaskNutritionModel,
    unfreeze_last_n_blocks: int,
    unfreeze_projection: bool,
) -> None:
    for parameter in model.clip_model.parameters():
        parameter.requires_grad = False

    if unfreeze_last_n_blocks <= 0 and not unfreeze_projection:
        return

    visual = model.clip_model.visual
    transformer = getattr(visual, "transformer", None)
    resblocks = list(getattr(transformer, "resblocks", []))
    if unfreeze_last_n_blocks > 0 and not resblocks:
        raise ValueError("Requested visual block unfreezing, but the CLIP visual tower has no resblocks.")

    blocks_to_unfreeze = resblocks[-unfreeze_last_n_blocks:] if unfreeze_last_n_blocks > 0 else []
    for block in blocks_to_unfreeze:
        for parameter in block.parameters():
            parameter.requires_grad = True

    if blocks_to_unfreeze and hasattr(visual, "ln_post"):
        for parameter in visual.ln_post.parameters():
            parameter.requires_grad = True

    visual_proj = getattr(visual, "proj", None)
    if unfreeze_projection and visual_proj is not None:
        if isinstance(visual_proj, torch.nn.Parameter):
            visual_proj.requires_grad = True
        else:
            for parameter in visual_proj.parameters():
                parameter.requires_grad = True
        if hasattr(visual, "ln_post"):
            for parameter in visual.ln_post.parameters():
                parameter.requires_grad = True


def set_encoder_trainability(
    model: MultiTaskNutritionModel,
    should_unfreeze: bool,
    unfreeze_last_n_blocks: int,
    unfreeze_projection: bool,
) -> None:
    for parameter in model.regression_head.parameters():
        parameter.requires_grad = True
    for parameter in model.classification_head.parameters():
        parameter.requires_grad = True

    if should_unfreeze:
        set_trainable_visual_layers(
            model=model,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            unfreeze_projection=unfreeze_projection,
        )
        return

    set_trainable_visual_layers(model=model, unfreeze_last_n_blocks=0, unfreeze_projection=False)


def build_optimizer(
    model: MultiTaskNutritionModel,
    head_lr: float,
    encoder_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    parameter_groups: list[dict[str, object]] = []

    head_parameters = [
        parameter
        for module in (model.regression_head, model.classification_head)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "lr": head_lr})

    encoder_parameters = [
        parameter for parameter in model.clip_model.parameters() if parameter.requires_grad
    ]
    if encoder_parameters:
        parameter_groups.append({"params": encoder_parameters, "lr": encoder_lr})

    if not parameter_groups:
        raise ValueError("No trainable parameters were found for the optimizer.")

    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)
