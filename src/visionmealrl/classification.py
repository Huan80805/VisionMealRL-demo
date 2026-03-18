from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from visionmealrl.embedding import resolve_device
from visionmealrl.logging_utils import configure_logging
from visionmealrl.nutrition5k import DishAnnotation, load_dish_annotations
from visionmealrl.regression import load_manifest_csv, set_seed, train_val_indices

LOGGER = logging.getLogger(__name__)


@dataclass
class ClassificationSplit:
    features: np.ndarray
    targets: np.ndarray
    dish_ids: list[str]
    labels: list[str]


class EmbeddingClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def filtered_ingredient_names(
    annotation: DishAnnotation,
    min_mass: float,
    min_fraction: float,
) -> set[str]:
    dish_mass = max(annotation.targets.total_mass, 1e-6)
    names = set()
    for ingredient in annotation.ingredients:
        if ingredient.mass < min_mass:
            continue
        if ingredient.mass < min_fraction * dish_mass:
            continue
        names.add(ingredient.name)
    return names


def build_ingredient_vocabulary(
    dish_ids: Sequence[str],
    annotations: dict[str, DishAnnotation],
    top_k: int,
    min_mass: float,
    min_fraction: float,
) -> list[str]:
    counter: dict[str, int] = {}
    for dish_id in dish_ids:
        annotation = annotations.get(dish_id)
        if annotation is None:
            continue
        for name in filtered_ingredient_names(annotation, min_mass=min_mass, min_fraction=min_fraction):
            counter[name] = counter.get(name, 0) + 1

    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [name for name, _count in ranked[:top_k]]


def build_multi_hot_targets(
    dish_ids: Sequence[str],
    annotations: dict[str, DishAnnotation],
    labels: Sequence[str],
    min_mass: float,
    min_fraction: float,
) -> np.ndarray:
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    targets = np.zeros((len(dish_ids), len(labels)), dtype=np.float32)

    for row_idx, dish_id in enumerate(dish_ids):
        annotation = annotations.get(dish_id)
        if annotation is None:
            continue
        for name in filtered_ingredient_names(annotation, min_mass=min_mass, min_fraction=min_fraction):
            label_idx = label_to_index.get(name)
            if label_idx is not None:
                targets[row_idx, label_idx] = 1.0

    return targets


def load_classification_split(
    split_dir: Path,
    annotations: dict[str, DishAnnotation],
    labels: Sequence[str],
    min_mass: float,
    min_fraction: float,
) -> ClassificationSplit:
    features = np.load(split_dir / "dish_embeddings.npy")
    manifest = load_manifest_csv(split_dir / "dish_manifest.csv")
    dish_ids = [row["dish_id"] for row in manifest]
    targets = build_multi_hot_targets(
        dish_ids=dish_ids,
        annotations=annotations,
        labels=labels,
        min_mass=min_mass,
        min_fraction=min_fraction,
    )
    return ClassificationSplit(
        features=features,
        targets=targets,
        dish_ids=dish_ids,
        labels=list(labels),
    )


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.Module,
    device: str,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_examples = 0

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        logits = model(features)
        loss = loss_fn(logits, targets)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = features.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def predict_logits(model: nn.Module, features: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        inputs = torch.from_numpy(features.astype(np.float32)).to(device)
        logits = model(inputs).detach().cpu().numpy()
    return logits.astype(np.float32)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def binary_average_precision(scores: np.ndarray, targets: np.ndarray) -> float | None:
    positives = targets.astype(bool)
    positive_count = int(positives.sum())
    if positive_count == 0:
        return None

    order = np.argsort(-scores, kind="stable")
    sorted_targets = positives[order].astype(np.float64)
    true_positives = np.cumsum(sorted_targets)
    precision = true_positives / np.arange(1, len(sorted_targets) + 1, dtype=np.float64)
    return float((precision * sorted_targets).sum() / positive_count)


def compute_micro_f1(predictions: np.ndarray, targets: np.ndarray) -> float:
    true_positives = float(np.logical_and(predictions == 1, targets == 1).sum())
    false_positives = float(np.logical_and(predictions == 1, targets == 0).sum())
    false_negatives = float(np.logical_and(predictions == 0, targets == 1).sum())
    precision = _safe_divide(true_positives, true_positives + false_positives)
    recall = _safe_divide(true_positives, true_positives + false_negatives)
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def compute_macro_f1(predictions: np.ndarray, targets: np.ndarray) -> float:
    f1_values: list[float] = []
    for column_idx in range(targets.shape[1]):
        target_column = targets[:, column_idx]
        if target_column.sum() == 0:
            continue
        prediction_column = predictions[:, column_idx]
        true_positives = float(np.logical_and(prediction_column == 1, target_column == 1).sum())
        false_positives = float(np.logical_and(prediction_column == 1, target_column == 0).sum())
        false_negatives = float(np.logical_and(prediction_column == 0, target_column == 1).sum())
        precision = _safe_divide(true_positives, true_positives + false_positives)
        recall = _safe_divide(true_positives, true_positives + false_negatives)
        if precision + recall == 0.0:
            f1_values.append(0.0)
            continue
        f1_values.append(float(2.0 * precision * recall / (precision + recall)))
    if not f1_values:
        return 0.0
    return float(np.mean(f1_values))


def compute_mean_average_precision(probabilities: np.ndarray, targets: np.ndarray) -> tuple[float, float, list[float | None]]:
    micro_ap = binary_average_precision(probabilities.reshape(-1), targets.reshape(-1))
    per_class_ap: list[float | None] = []
    for column_idx in range(targets.shape[1]):
        per_class_ap.append(binary_average_precision(probabilities[:, column_idx], targets[:, column_idx]))

    valid_class_aps = [value for value in per_class_ap if value is not None]
    macro_ap = float(np.mean(valid_class_aps)) if valid_class_aps else 0.0
    return float(micro_ap or 0.0), macro_ap, per_class_ap


def compute_ranking_metrics(probabilities: np.ndarray, targets: np.ndarray, k: int) -> dict[str, float]:
    k = min(k, probabilities.shape[1])
    topk_indices = np.argsort(-probabilities, axis=1)[:, :k]
    hits = np.take_along_axis(targets, topk_indices, axis=1).sum(axis=1)
    positives_per_row = targets.sum(axis=1)

    precision_at_k = float(np.mean(hits / max(k, 1)))
    recall_values = np.zeros_like(hits, dtype=np.float64)
    valid_rows = positives_per_row > 0
    recall_values[valid_rows] = hits[valid_rows] / positives_per_row[valid_rows]
    recall_at_k = float(np.mean(recall_values))
    return {
        f"precision_at_{k}": precision_at_k,
        f"recall_at_{k}": recall_at_k,
    }


def select_threshold(probabilities: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    candidate_thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidate_thresholds:
        predictions = (probabilities >= threshold).astype(np.int32)
        f1 = compute_micro_f1(predictions, targets)
        if f1 > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(f1)
    return best_threshold, best_f1


def write_predictions_csv(
    path: Path,
    dish_ids: Sequence[str],
    probabilities: np.ndarray,
    targets: np.ndarray,
    labels: Sequence[str],
    ranking_k: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dish_id", "predicted_top_labels", "predicted_top_scores", "true_labels"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dish_id, probability_row, target_row in zip(dish_ids, probabilities, targets):
            top_indices = np.argsort(-probability_row)[:ranking_k]
            predicted_labels = [labels[idx] for idx in top_indices]
            predicted_scores = [float(probability_row[idx]) for idx in top_indices]
            true_labels = [labels[idx] for idx, value in enumerate(target_row) if value > 0.5]
            writer.writerow(
                {
                    "dish_id": dish_id,
                    "predicted_top_labels": "|".join(predicted_labels),
                    "predicted_top_scores": "|".join(f"{score:.6f}" for score in predicted_scores),
                    "true_labels": "|".join(true_labels),
                }
            )


def write_per_class_metrics_csv(
    path: Path,
    labels: Sequence[str],
    targets: np.ndarray,
    per_class_ap: Sequence[float | None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", "support", "average_precision"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, ap_value, target_column in zip(labels, per_class_ap, targets.T):
            writer.writerow(
                {
                    "label": label,
                    "support": int(target_column.sum()),
                    "average_precision": "" if ap_value is None else float(ap_value),
                }
            )


def write_split_manifest(path: Path, split_name: str, dish_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "dish_id"])
        writer.writeheader()
        for dish_id in dish_ids:
            writer.writerow({"split": split_name, "dish_id": dish_id})


def compute_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    ranking_k: int,
) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(np.int32)
    micro_ap, macro_ap, per_class_ap = compute_mean_average_precision(probabilities, targets)
    ranking_metrics = compute_ranking_metrics(probabilities, targets, ranking_k)
    return {
        "threshold": float(threshold),
        "micro_map": micro_ap,
        "macro_map": macro_ap,
        "micro_f1": compute_micro_f1(predictions, targets),
        "macro_f1": compute_macro_f1(predictions, targets),
        "per_class_average_precision": per_class_ap,
        **ranking_metrics,
    }


def train_classifier_main(args) -> None:
    configure_logging()
    set_seed(args.seed)

    device = resolve_device(args.device)
    annotations = load_dish_annotations(args.dataset_root)

    train_manifest = load_manifest_csv(args.embeddings_root / "train" / "dish_manifest.csv")
    train_dish_ids = [row["dish_id"] for row in train_manifest]
    labels = build_ingredient_vocabulary(
        dish_ids=train_dish_ids,
        annotations=annotations,
        top_k=args.top_k,
        min_mass=args.ingredient_min_mass,
        min_fraction=args.ingredient_min_fraction,
    )

    train_split = load_classification_split(
        split_dir=args.embeddings_root / "train",
        annotations=annotations,
        labels=labels,
        min_mass=args.ingredient_min_mass,
        min_fraction=args.ingredient_min_fraction,
    )
    test_split = load_classification_split(
        split_dir=args.embeddings_root / "test",
        annotations=annotations,
        labels=labels,
        min_mass=args.ingredient_min_mass,
        min_fraction=args.ingredient_min_fraction,
    )

    train_indices, val_indices = train_val_indices(
        num_rows=train_split.features.shape[0],
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    x_train = train_split.features[train_indices]
    y_train = train_split.targets[train_indices]
    train_dish_ids = [train_split.dish_ids[idx] for idx in train_indices]

    x_val = train_split.features[val_indices]
    y_val = train_split.targets[val_indices]
    val_dish_ids = [train_split.dish_ids[idx] for idx in val_indices]

    train_dataset = EmbeddingClassificationDataset(x_train, y_train)
    val_dataset = EmbeddingClassificationDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = LinearClassifier(
        input_dim=train_split.features.shape[1],
        output_dim=len(labels),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    history: List[Dict[str, Union[float, int]]] = []

    LOGGER.info(
        "Training linear classifier on %d train dishes with %d validation dishes using %s",
        x_train.shape[0],
        x_val.shape[0],
        device,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = run_epoch(model, val_loader, None, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        LOGGER.info(
            "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a valid classifier checkpoint.")

    model.load_state_dict(best_state)

    val_logits = predict_logits(model, x_val, device)
    test_logits = predict_logits(model, test_split.features, device)
    val_probabilities = 1.0 / (1.0 + np.exp(-val_logits))
    test_probabilities = 1.0 / (1.0 + np.exp(-test_logits))

    threshold, best_val_micro_f1 = select_threshold(val_probabilities, y_val)
    val_metrics = compute_metrics(
        probabilities=val_probabilities,
        targets=y_val,
        threshold=threshold,
        ranking_k=args.ranking_k,
    )
    test_metrics = compute_metrics(
        probabilities=test_probabilities,
        targets=test_split.targets,
        threshold=threshold,
        ranking_k=args.ranking_k,
    )

    if getattr(args, "output_dir", None) is not None:
        output_dir = args.output_dir
    else:
        output_dir = args.output_root / "classifiers" / "linear"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": train_split.features.shape[1],
            "labels": labels,
            "head": "linear",
            "ingredient_min_mass": args.ingredient_min_mass,
            "ingredient_min_fraction": args.ingredient_min_fraction,
            "top_k": args.top_k,
        },
        output_dir / "best_model.pt",
    )

    label_payload = {
        "labels": labels,
        "top_k": args.top_k,
        "ingredient_min_mass": args.ingredient_min_mass,
        "ingredient_min_fraction": args.ingredient_min_fraction,
        "ranking_k": args.ranking_k,
    }
    with (output_dir / "label_vocabulary.json").open("w", encoding="utf-8") as handle:
        json.dump(label_payload, handle, indent=2)

    metrics_payload = {
        "head": "linear",
        "device": device,
        "best_val_loss": best_val_loss,
        "best_val_micro_f1": best_val_micro_f1,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    run_config = {
        "dataset_root": str(args.dataset_root),
        "embeddings_root": str(args.embeddings_root),
        "output_root": str(args.output_root),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "device": device,
        "top_k": args.top_k,
        "ingredient_min_mass": args.ingredient_min_mass,
        "ingredient_min_fraction": args.ingredient_min_fraction,
        "ranking_k": args.ranking_k,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    write_predictions_csv(
        output_dir / "predictions_test.csv",
        dish_ids=test_split.dish_ids,
        probabilities=test_probabilities,
        targets=test_split.targets,
        labels=labels,
        ranking_k=args.ranking_k,
    )
    write_per_class_metrics_csv(
        output_dir / "per_class_metrics.csv",
        labels=labels,
        targets=test_split.targets,
        per_class_ap=test_metrics["per_class_average_precision"],
    )
    write_split_manifest(output_dir / "train_split_manifest.csv", "train", train_dish_ids)
    write_split_manifest(output_dir / "val_split_manifest.csv", "val", val_dish_ids)
    np.save(output_dir / "test_probabilities.npy", test_probabilities.astype(np.float32))
    LOGGER.info("Saved classifier outputs to %s", output_dir)
