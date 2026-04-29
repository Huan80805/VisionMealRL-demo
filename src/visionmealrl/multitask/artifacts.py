from __future__ import annotations

import json
from pathlib import Path

import torch

from visionmealrl.artifacts import write_split_manifest
from visionmealrl.classification import (
    write_per_class_metrics_csv,
    write_predictions_csv as write_classification_predictions_csv,
)
from visionmealrl.constants import (
    DEFAULT_INGREDIENT_MIN_FRACTION,
    DEFAULT_INGREDIENT_MIN_MASS,
    DEFAULT_INGREDIENT_TOP_K,
    DEFAULT_RANKING_K,
)
from visionmealrl.embedding import export_dish_embeddings_with_encoder
from visionmealrl.labels import IngredientLabelConfig
from visionmealrl.multitask.model import MultiTaskNutritionModel
from visionmealrl.nutrition5k import DishRecord
from visionmealrl.regression import write_predictions_csv as write_regression_predictions_csv

CLIP_MODEL_PREFIX = "clip_model."


def save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: str) -> dict[str, object]:
    return torch.load(path, map_location=device, weights_only=False)


def extract_clip_model_state_dict(multitask_state_dict: dict[str, object]) -> dict[str, object]:
    clip_state_dict: dict[str, object] = {}
    for key, value in multitask_state_dict.items():
        if key.startswith(CLIP_MODEL_PREFIX):
            clip_state_dict[key.removeprefix(CLIP_MODEL_PREFIX)] = value
    if not clip_state_dict:
        raise ValueError("No clip_model weights found in multitask state_dict.")
    return clip_state_dict


def build_embedding_model_payload(
    *,
    clip_model_state_dict: dict[str, object],
    model_name: str,
    pretrained: str,
    image_source: str,
    embedding_dim: int,
    checkpoint_source: str,
    unfreeze_last_n_blocks: int | None = None,
    unfreeze_projection: bool | None = None,
    best_epoch: int | None = None,
    best_val_loss: float | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "export_type": "open_clip_image_embedding_model",
        "clip_model_state_dict": clip_model_state_dict,
        "model_name": model_name,
        "pretrained": pretrained,
        "image_source": image_source,
        "embedding_dim": embedding_dim,
        "normalize_embeddings": True,
        "checkpoint_source": checkpoint_source,
    }
    if unfreeze_last_n_blocks is not None:
        payload["unfreeze_last_n_blocks"] = unfreeze_last_n_blocks
    if unfreeze_projection is not None:
        payload["unfreeze_projection"] = unfreeze_projection
    if best_epoch is not None:
        payload["best_epoch"] = best_epoch
    if best_val_loss is not None:
        payload["best_val_loss"] = best_val_loss
    return payload


def save_embedding_model_checkpoint(
    *,
    path: Path,
    model: MultiTaskNutritionModel,
    model_name: str,
    pretrained: str,
    image_source: str,
    embedding_dim: int,
    checkpoint_source: str,
    unfreeze_last_n_blocks: int | None = None,
    unfreeze_projection: bool | None = None,
    best_epoch: int | None = None,
    best_val_loss: float | None = None,
) -> None:
    payload = build_embedding_model_payload(
        clip_model_state_dict=model.clip_model.state_dict(),
        model_name=model_name,
        pretrained=pretrained,
        image_source=image_source,
        embedding_dim=embedding_dim,
        checkpoint_source=checkpoint_source,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        unfreeze_projection=unfreeze_projection,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
    )
    save_checkpoint(path, payload)


def export_embedding_model_from_multitask_checkpoint(
    *,
    checkpoint_path: Path,
    output_path: Path,
    device: str,
) -> None:
    checkpoint_payload = load_checkpoint(checkpoint_path, device=device)
    multitask_state_dict = checkpoint_payload["model_state_dict"]
    clip_model_state_dict = extract_clip_model_state_dict(multitask_state_dict)
    payload = build_embedding_model_payload(
        clip_model_state_dict=clip_model_state_dict,
        model_name=str(checkpoint_payload["model_name"]),
        pretrained=str(checkpoint_payload["pretrained"]),
        image_source=str(checkpoint_payload["image_source"]),
        embedding_dim=int(checkpoint_payload["embedding_dim"]),
        checkpoint_source=str(checkpoint_path),
        unfreeze_last_n_blocks=int(checkpoint_payload.get("unfreeze_last_n_blocks", 0)),
        unfreeze_projection=bool(checkpoint_payload.get("unfreeze_projection", False)),
    )
    save_checkpoint(output_path, payload)


def export_finetuned_split_embeddings(
    model: MultiTaskNutritionModel,
    preprocess,
    dish_records_by_split: dict[str, list[DishRecord]],
    output_dir: Path,
    args,
    device: str,
) -> None:
    output_base = output_dir / "finetuned_embeddings"
    export_dish_embeddings_with_encoder(
        dish_records_by_split=dish_records_by_split,
        model=model.clip_model,
        preprocess=preprocess,
        output_base=output_base,
        image_source=args.image_source,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        metadata_overrides={
            "source": "multitask_finetuned_encoder",
            "model_name": args.model_name,
            "pretrained": args.pretrained,
            "checkpoint": str(output_dir / "best_model.pt"),
        },
    )


def write_multitask_outputs(
    *,
    output_dir: Path,
    args,
    device: str,
    model: MultiTaskNutritionModel,
    target_mean,
    target_std,
    labels: list[str],
    label_config: IngredientLabelConfig,
    embedding_dim: int,
    best_epoch: int,
    best_val_loss: float,
    best_val_micro_f1: float,
    history: list[dict[str, float | int | bool]],
    final_val_outputs: dict[str, object],
    final_test_outputs: dict[str, object],
    val_regression_metrics: dict[str, object],
    test_regression_metrics: dict[str, object],
    val_classification_metrics: dict[str, object],
    test_classification_metrics: dict[str, object],
    test_records: list[DishRecord],
    train_subset: list[DishRecord],
    val_subset: list[DishRecord],
    y_test,
    test_regression_predictions,
    dish_records_by_split: dict[str, list[DishRecord]],
    preprocess,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "image_source": args.image_source,
        "unfreeze_last_n_blocks": args.unfreeze_last_n_blocks,
        "unfreeze_projection": args.unfreeze_projection,
        "dropout": args.dropout,
        "target_mean": target_mean,
        "target_std": target_std,
        "labels": labels,
        "embedding_dim": embedding_dim,
        "top_k_ingredients": DEFAULT_INGREDIENT_TOP_K,
        "ingredient_min_mass": DEFAULT_INGREDIENT_MIN_MASS,
        "ingredient_min_fraction": DEFAULT_INGREDIENT_MIN_FRACTION,
    }
    save_checkpoint(output_dir / "best_model.pt", checkpoint_payload)
    save_embedding_model_checkpoint(
        path=output_dir / "best_embedding_model.pt",
        model=model,
        model_name=args.model_name,
        pretrained=args.pretrained,
        image_source=args.image_source,
        embedding_dim=embedding_dim,
        checkpoint_source=str(output_dir / "best_model.pt"),
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        unfreeze_projection=args.unfreeze_projection,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
    )

    with (output_dir / "label_vocabulary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "labels": label_config.labels,
                "top_k": label_config.top_k,
                "ingredient_min_mass": label_config.min_mass,
                "ingredient_min_fraction": label_config.min_fraction,
                "ranking_k": DEFAULT_RANKING_K,
            },
            handle,
            indent=2,
        )

    metrics_payload = {
        "device": device,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_micro_f1": best_val_micro_f1,
        "lambda_reg": args.lambda_reg,
        "lambda_cls": args.lambda_cls,
        "val_losses": {
            "total": float(final_val_outputs["total_loss"]),
            "regression": float(final_val_outputs["regression_loss"]),
            "classification": float(final_val_outputs["classification_loss"]),
        },
        "test_losses": {
            "total": float(final_test_outputs["total_loss"]),
            "regression": float(final_test_outputs["regression_loss"]),
            "classification": float(final_test_outputs["classification_loss"]),
        },
        "val_regression_metrics": val_regression_metrics,
        "test_regression_metrics": test_regression_metrics,
        "val_classification_metrics": val_classification_metrics,
        "test_classification_metrics": test_classification_metrics,
        "history": history,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    run_config = {
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "run_name": getattr(args, "run_name", None),
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "image_source": args.image_source,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "freeze_epochs": args.freeze_epochs,
        "head_lr": args.head_lr,
        "encoder_lr": args.encoder_lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "lambda_reg": args.lambda_reg,
        "lambda_cls": args.lambda_cls,
        "unfreeze_last_n_blocks": args.unfreeze_last_n_blocks,
        "unfreeze_projection": args.unfreeze_projection,
        "top_k_ingredients": DEFAULT_INGREDIENT_TOP_K,
        "ingredient_min_mass": DEFAULT_INGREDIENT_MIN_MASS,
        "ingredient_min_fraction": DEFAULT_INGREDIENT_MIN_FRACTION,
        "validation_size": args.validation_size,
        "seed": args.seed,
        "device": device,
        "ranking_k": DEFAULT_RANKING_K,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    write_regression_predictions_csv(
        output_dir / "regression_predictions_test.csv",
        dish_ids=[record.dish_id for record in test_records],
        predictions=test_regression_predictions,
        targets=y_test,
    )
    write_classification_predictions_csv(
        output_dir / "classification_predictions_test.csv",
        dish_ids=[record.dish_id for record in test_records],
        probabilities=final_test_outputs["classification_probabilities"],
        targets=final_test_outputs["classification_targets"],
        labels=labels,
        ranking_k=DEFAULT_RANKING_K,
    )
    write_per_class_metrics_csv(
        output_dir / "classification_per_class_metrics.csv",
        labels=labels,
        targets=final_test_outputs["classification_targets"],
        per_class_ap=test_classification_metrics["per_class_average_precision"],
    )
    write_split_manifest(
        output_dir / "train_split_manifest.csv",
        "train",
        [record.dish_id for record in train_subset],
    )
    write_split_manifest(
        output_dir / "val_split_manifest.csv",
        "val",
        [record.dish_id for record in val_subset],
    )

    export_finetuned_split_embeddings(
        model=model,
        preprocess=preprocess,
        dish_records_by_split=dish_records_by_split,
        output_dir=output_dir,
        args=args,
        device=device,
    )
