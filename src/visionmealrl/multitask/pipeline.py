from __future__ import annotations

import logging

import numpy as np
import torch

from visionmealrl.checkpointing import (
    capture_rng_state,
    epoch_checkpoint_path,
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)
from visionmealrl.classification import (
    compute_metrics as compute_classification_metrics,
    select_threshold,
)
from visionmealrl.constants import (
    DEFAULT_INGREDIENT_MIN_FRACTION,
    DEFAULT_INGREDIENT_MIN_MASS,
    DEFAULT_INGREDIENT_TOP_K,
    DEFAULT_RANKING_K,
)
from visionmealrl.embedding import (
    export_dish_embeddings_with_encoder,
    load_openclip_model_and_preprocess,
    resolve_device,
    slugify_model_name,
)
from visionmealrl.labels import IngredientLabelConfig, build_ingredient_vocabulary, encode_multi_hot_ingredients
from visionmealrl.logging_utils import configure_logging
from visionmealrl.multitask.artifacts import load_checkpoint, write_multitask_outputs
from visionmealrl.multitask.data import build_multitask_dataloader
from visionmealrl.multitask.model import (
    MultiTaskNutritionModel,
    build_optimizer,
    set_encoder_trainability,
)
from visionmealrl.multitask.train import evaluate_multitask, train_multitask_epoch
from visionmealrl.nutrition5k import DishRecord, build_dish_records_by_split, load_dish_annotations
from visionmealrl.regression import (
    compute_metrics as compute_regression_metrics,
    set_seed,
    standardize_targets,
    train_val_indices,
)

LOGGER = logging.getLogger(__name__)


def _default_run_name(args) -> str:
    if getattr(args, "run_name", None):
        return str(args.run_name)
    return f"{slugify_model_name(args.model_name, args.pretrained)}_{args.image_source}"


def _build_label_config(train_records: list[DishRecord], annotations) -> IngredientLabelConfig:
    labels = build_ingredient_vocabulary(
        dish_ids=[record.dish_id for record in train_records],
        annotations=annotations,
        top_k=DEFAULT_INGREDIENT_TOP_K,
        min_mass=DEFAULT_INGREDIENT_MIN_MASS,
        min_fraction=DEFAULT_INGREDIENT_MIN_FRACTION,
    )
    if not labels:
        raise ValueError("No ingredient labels were constructed for multitask training.")
    return IngredientLabelConfig(
        labels=labels,
        top_k=DEFAULT_INGREDIENT_TOP_K,
        min_mass=DEFAULT_INGREDIENT_MIN_MASS,
        min_fraction=DEFAULT_INGREDIENT_MIN_FRACTION,
    )


def _split_train_records(train_records: list[DishRecord], validation_size: float, seed: int) -> tuple[list[DishRecord], list[DishRecord]]:
    train_indices, val_indices = train_val_indices(
        num_rows=len(train_records),
        validation_size=validation_size,
        seed=seed,
    )
    train_subset = [train_records[idx] for idx in train_indices]
    val_subset = [train_records[idx] for idx in val_indices]
    return train_subset, val_subset


def _build_regression_targets(
    train_subset: list[DishRecord],
    val_subset: list[DishRecord],
    test_records: list[DishRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_train = np.asarray([record.targets.as_list() for record in train_subset], dtype=np.float32)
    y_val = np.asarray([record.targets.as_list() for record in val_subset], dtype=np.float32)
    y_test = np.asarray([record.targets.as_list() for record in test_records], dtype=np.float32)
    y_train_scaled, y_val_scaled, target_mean, target_std = standardize_targets(y_train, y_val)
    _unused_train_scaled, y_test_scaled, _unused_mean, _unused_std = standardize_targets(y_train, y_test)
    return y_train, y_val, y_test, y_train_scaled, y_val_scaled, y_test_scaled, target_std, target_mean


def _build_classification_targets(
    *,
    train_subset: list[DishRecord],
    val_subset: list[DishRecord],
    test_records: list[DishRecord],
    annotations,
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cls_train = encode_multi_hot_ingredients(
        dish_ids=[record.dish_id for record in train_subset],
        annotations=annotations,
        labels=labels,
        min_mass=DEFAULT_INGREDIENT_MIN_MASS,
        min_fraction=DEFAULT_INGREDIENT_MIN_FRACTION,
    )
    cls_val = encode_multi_hot_ingredients(
        dish_ids=[record.dish_id for record in val_subset],
        annotations=annotations,
        labels=labels,
        min_mass=DEFAULT_INGREDIENT_MIN_MASS,
        min_fraction=DEFAULT_INGREDIENT_MIN_FRACTION,
    )
    cls_test = encode_multi_hot_ingredients(
        dish_ids=[record.dish_id for record in test_records],
        annotations=annotations,
        labels=labels,
        min_mass=DEFAULT_INGREDIENT_MIN_MASS,
        min_fraction=DEFAULT_INGREDIENT_MIN_FRACTION,
    )
    return cls_train, cls_val, cls_test


def _train_model(
    *,
    model: MultiTaskNutritionModel,
    train_loader,
    val_loader,
    args,
    device: str,
    output_dir,
) -> tuple[list[dict[str, float | int | bool]], dict[str, object], int, float]:
    should_unfreeze = args.freeze_epochs <= 0 and (
        args.unfreeze_last_n_blocks > 0 or args.unfreeze_projection
    )
    set_encoder_trainability(
        model=model,
        should_unfreeze=should_unfreeze,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        unfreeze_projection=args.unfreeze_projection,
    )
    optimizer = build_optimizer(
        model=model,
        head_lr=args.head_lr,
        encoder_lr=args.encoder_lr,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, float | int | bool]] = []
    best_state = None
    best_epoch = 0
    best_val_loss = float("inf")
    encoder_unfrozen = should_unfreeze
    checkpoint_dir = output_dir / "epoch_checkpoints"

    for epoch in range(1, args.epochs + 1):
        should_unfreeze_epoch = epoch > args.freeze_epochs and (
            args.unfreeze_last_n_blocks > 0 or args.unfreeze_projection
        )
        if should_unfreeze_epoch != encoder_unfrozen:
            set_encoder_trainability(
                model=model,
                should_unfreeze=should_unfreeze_epoch,
                unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
                unfreeze_projection=args.unfreeze_projection,
            )
            optimizer = build_optimizer(
                model=model,
                head_lr=args.head_lr,
                encoder_lr=args.encoder_lr,
                weight_decay=args.weight_decay,
            )
            encoder_unfrozen = should_unfreeze_epoch

        checkpoint_path = epoch_checkpoint_path(checkpoint_dir, epoch)
        if checkpoint_path.exists():
            checkpoint = load_training_checkpoint(checkpoint_path, device=device)
            checkpoint_encoder_unfrozen = bool(checkpoint["encoder_unfrozen"])
            if checkpoint_encoder_unfrozen != encoder_unfrozen:
                set_encoder_trainability(
                    model=model,
                    should_unfreeze=checkpoint_encoder_unfrozen,
                    unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
                    unfreeze_projection=args.unfreeze_projection,
                )
                optimizer = build_optimizer(
                    model=model,
                    head_lr=args.head_lr,
                    encoder_lr=args.encoder_lr,
                    weight_decay=args.weight_decay,
                )
                encoder_unfrozen = checkpoint_encoder_unfrozen

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            history = list(checkpoint["history"])
            best_epoch = int(checkpoint["best_epoch"])
            best_val_loss = float(checkpoint["best_val_loss"])
            checkpoint_best_state = checkpoint.get("best_state_dict")
            best_state = (
                None
                if checkpoint_best_state is None
                else {
                    key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in checkpoint_best_state.items()
                }
            )
            restore_rng_state(checkpoint.get("rng_state"))
            LOGGER.info("Loaded existing multitask checkpoint for epoch %d/%d", epoch, args.epochs)
            continue

        train_stats = train_multitask_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_reg=args.lambda_reg,
            lambda_cls=args.lambda_cls,
        )
        val_outputs = evaluate_multitask(
            model=model,
            dataloader=val_loader,
            device=device,
            lambda_reg=args.lambda_reg,
            lambda_cls=args.lambda_cls,
        )

        history.append(
            {
                "epoch": epoch,
                "encoder_unfrozen": encoder_unfrozen,
                "train_total_loss": train_stats["total_loss"],
                "train_regression_loss": train_stats["regression_loss"],
                "train_classification_loss": train_stats["classification_loss"],
                "val_total_loss": float(val_outputs["total_loss"]),
                "val_regression_loss": float(val_outputs["regression_loss"]),
                "val_classification_loss": float(val_outputs["classification_loss"]),
            }
        )

        if float(val_outputs["total_loss"]) < best_val_loss:
            best_val_loss = float(val_outputs["total_loss"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        LOGGER.info(
            "Epoch %d/%d | train_total=%.6f | val_total=%.6f | val_reg=%.6f | val_cls=%.6f | encoder_unfrozen=%s",
            epoch,
            args.epochs,
            train_stats["total_loss"],
            float(val_outputs["total_loss"]),
            float(val_outputs["regression_loss"]),
            float(val_outputs["classification_loss"]),
            encoder_unfrozen,
        )

        save_training_checkpoint(
            checkpoint_path,
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "best_state_dict": best_state,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "encoder_unfrozen": encoder_unfrozen,
                "rng_state": capture_rng_state(),
            },
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a valid multitask checkpoint.")
    return history, best_state, best_epoch, best_val_loss


def train_multitask_main(args) -> None:
    configure_logging()
    set_seed(args.seed)

    device = resolve_device(args.device)
    output_dir = getattr(args, "output_dir", None) or (args.output_root / "multitask" / _default_run_name(args))
    annotations = load_dish_annotations(args.dataset_root)
    dish_records_by_split = build_dish_records_by_split(args.dataset_root, args.image_source)
    train_records = dish_records_by_split.get("train", [])
    test_records = dish_records_by_split.get("test", [])
    if not train_records or not test_records:
        raise ValueError("Both train and test splits must contain dish records for multitask training.")

    label_config = _build_label_config(train_records, annotations)
    labels = label_config.labels
    train_subset, val_subset = _split_train_records(train_records, args.validation_size, args.seed)
    y_train, y_val, y_test, y_train_scaled, y_val_scaled, y_test_scaled, target_std, target_mean = _build_regression_targets(
        train_subset,
        val_subset,
        test_records,
    )
    cls_train, cls_val, cls_test = _build_classification_targets(
        train_subset=train_subset,
        val_subset=val_subset,
        test_records=test_records,
        annotations=annotations,
        labels=labels,
    )

    clip_model, preprocess = load_openclip_model_and_preprocess(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=device,
    )

    embedding_dim = int(clip_model.visual.output_dim)
    model = MultiTaskNutritionModel(
        clip_model=clip_model,
        embedding_dim=embedding_dim,
        num_labels=len(labels),
        dropout=args.dropout,
    ).to(device)

    train_loader = build_multitask_dataloader(
        dish_records=train_subset,
        preprocess=preprocess,
        regression_targets=y_train_scaled,
        classification_targets=cls_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        device=device,
    )
    val_loader = build_multitask_dataloader(
        dish_records=val_subset,
        preprocess=preprocess,
        regression_targets=y_val_scaled,
        classification_targets=cls_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        device=device,
    )
    test_loader = build_multitask_dataloader(
        dish_records=test_records,
        preprocess=preprocess,
        regression_targets=y_test_scaled,
        classification_targets=cls_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        device=device,
    )

    LOGGER.info(
        "Training multitask model on %d train dishes with %d validation dishes using %s",
        len(train_subset),
        len(val_subset),
        device,
    )

    history, best_state, best_epoch, best_val_loss = _train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        device=device,
        output_dir=output_dir,
    )

    model.load_state_dict(best_state)
    final_val_outputs = evaluate_multitask(
        model=model,
        dataloader=val_loader,
        device=device,
        lambda_reg=args.lambda_reg,
        lambda_cls=args.lambda_cls,
    )
    final_test_outputs = evaluate_multitask(
        model=model,
        dataloader=test_loader,
        device=device,
        lambda_reg=args.lambda_reg,
        lambda_cls=args.lambda_cls,
    )

    val_regression_predictions = final_val_outputs["regression_predictions"] * target_std + target_mean
    test_regression_predictions = final_test_outputs["regression_predictions"] * target_std + target_mean
    val_regression_metrics = compute_regression_metrics(val_regression_predictions, y_val)
    test_regression_metrics = compute_regression_metrics(test_regression_predictions, y_test)

    threshold, best_val_micro_f1 = select_threshold(
        final_val_outputs["classification_probabilities"],
        final_val_outputs["classification_targets"],
    )
    val_classification_metrics = compute_classification_metrics(
        probabilities=final_val_outputs["classification_probabilities"],
        targets=final_val_outputs["classification_targets"],
        threshold=threshold,
        ranking_k=DEFAULT_RANKING_K,
    )
    test_classification_metrics = compute_classification_metrics(
        probabilities=final_test_outputs["classification_probabilities"],
        targets=final_test_outputs["classification_targets"],
        threshold=threshold,
        ranking_k=DEFAULT_RANKING_K,
    )

    write_multitask_outputs(
        output_dir=output_dir,
        args=args,
        device=device,
        model=model,
        target_mean=target_mean,
        target_std=target_std,
        labels=labels,
        label_config=label_config,
        embedding_dim=embedding_dim,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_micro_f1=best_val_micro_f1,
        history=history,
        final_val_outputs=final_val_outputs,
        final_test_outputs=final_test_outputs,
        val_regression_metrics=val_regression_metrics,
        test_regression_metrics=test_regression_metrics,
        val_classification_metrics=val_classification_metrics,
        test_classification_metrics=test_classification_metrics,
        test_records=test_records,
        train_subset=train_subset,
        val_subset=val_subset,
        y_test=y_test,
        test_regression_predictions=test_regression_predictions,
        dish_records_by_split=dish_records_by_split,
        preprocess=preprocess,
    )
    LOGGER.info("Saved multitask outputs to %s", output_dir)


def export_finetuned_embeddings_main(args) -> None:
    configure_logging()

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    image_source = args.image_source or str(checkpoint["image_source"])
    clip_model, preprocess = load_openclip_model_and_preprocess(
        model_name=str(checkpoint["model_name"]),
        pretrained=str(checkpoint["pretrained"]),
        device=device,
    )
    model = MultiTaskNutritionModel(
        clip_model=clip_model,
        embedding_dim=int(checkpoint["embedding_dim"]),
        num_labels=len(checkpoint["labels"]),
        dropout=float(checkpoint.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dish_records_by_split = build_dish_records_by_split(args.dataset_root, image_source)
    checkpoint_name = args.checkpoint.parent.name if args.checkpoint.stem == "best_model" else args.checkpoint.stem
    output_base = args.output_root / "multitask_exports" / checkpoint_name
    export_dish_embeddings_with_encoder(
        dish_records_by_split=dish_records_by_split,
        model=model.clip_model,
        preprocess=preprocess,
        output_base=output_base,
        image_source=image_source,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        metadata_overrides={
            "source": "multitask_finetuned_encoder",
            "model_name": checkpoint["model_name"],
            "pretrained": checkpoint["pretrained"],
            "checkpoint": str(args.checkpoint),
        },
    )
    LOGGER.info("Exported finetuned embeddings to %s", output_base)
