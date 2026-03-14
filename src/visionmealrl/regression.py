from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from visionmealrl.constants import TARGET_COLUMNS
from visionmealrl.embedding import resolve_device
from visionmealrl.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@dataclass
class RegressionSplit:
    features: np.ndarray
    targets: np.ndarray
    dish_ids: list[str]


def load_regression_split(split_dir: Path) -> RegressionSplit:
    features = np.load(split_dir / "dish_embeddings.npy")
    manifest = load_manifest_csv(split_dir / "dish_manifest.csv")
    targets = np.asarray(
        [[float(row[column]) for column in TARGET_COLUMNS] for row in manifest],
        dtype=np.float32,
    )
    dish_ids = [row["dish_id"] for row in manifest]
    return RegressionSplit(features=features, targets=targets, dish_ids=dish_ids)


class EmbeddingRegressionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


class LinearRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def train_val_indices(num_rows: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(num_rows)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_size = max(1, int(round(num_rows * val_fraction)))
    val_indices = np.sort(indices[:val_size])
    train_indices = np.sort(indices[val_size:])
    if train_indices.size == 0:
        raise ValueError("Validation split consumed all training samples.")
    return train_indices, val_indices


def standardize_targets(train_targets: np.ndarray, other_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_targets.mean(axis=0)
    std = train_targets.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (
        (train_targets - mean) / std,
        (other_targets - mean) / std,
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    absolute_error = np.abs(predictions - targets)
    squared_error = (predictions - targets) ** 2

    per_target = {}
    for idx, column in enumerate(TARGET_COLUMNS):
        target_values = targets[:, idx]
        denominator = np.maximum(np.abs(target_values), 1e-6)
        per_target[column] = {
            "mae": float(absolute_error[:, idx].mean()),
            "rmse": float(np.sqrt(squared_error[:, idx].mean())),
            "mape": float((absolute_error[:, idx] / denominator).mean()),
        }

    return {
        "overall_mae": float(absolute_error.mean()),
        "overall_rmse": float(np.sqrt(squared_error.mean())),
        "per_target": per_target,
    }


def build_model(args, input_dim: int, output_dim: int) -> nn.Module:
    if args.head == "linear":
        return LinearRegressor(input_dim=input_dim, output_dim=output_dim)
    return MLPRegressor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        dropout=args.dropout,
    )


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
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

        predictions = model(features)
        loss = loss_fn(predictions, targets)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = features.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def predict(model: nn.Module, features: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        inputs = torch.from_numpy(features.astype(np.float32)).to(device)
        outputs = model(inputs).detach().cpu().numpy()
    return outputs.astype(np.float32)


def write_predictions_csv(path: Path, dish_ids: list[str], predictions: np.ndarray, targets: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dish_id"]
    for column in TARGET_COLUMNS:
        fieldnames.append(f"pred_{column}")
    for column in TARGET_COLUMNS:
        fieldnames.append(f"target_{column}")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dish_id, prediction_row, target_row in zip(dish_ids, predictions, targets):
            row = {"dish_id": dish_id}
            for idx, column in enumerate(TARGET_COLUMNS):
                row[f"pred_{column}"] = float(prediction_row[idx])
                row[f"target_{column}"] = float(target_row[idx])
            writer.writerow(row)


def train_regressor_main(args) -> None:
    configure_logging()
    set_seed(args.seed)

    device = resolve_device(args.device)
    train_split = load_regression_split(args.embeddings_root / "train")
    test_split = load_regression_split(args.embeddings_root / "test")

    train_indices, val_indices = train_val_indices(
        num_rows=train_split.features.shape[0],
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    x_train = train_split.features[train_indices]
    y_train = train_split.targets[train_indices]
    x_val = train_split.features[val_indices]
    y_val = train_split.targets[val_indices]

    y_train_scaled, y_val_scaled, target_mean, target_std = standardize_targets(y_train, y_val)

    train_dataset = EmbeddingRegressionDataset(x_train, y_train_scaled)
    val_dataset = EmbeddingRegressionDataset(x_val, y_val_scaled)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(
        args=args,
        input_dim=train_split.features.shape[1],
        output_dim=len(TARGET_COLUMNS),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    history: List[Dict[str, Union[float, int]]] = []

    LOGGER.info(
        "Training %s regressor on %d train dishes with %d validation dishes using %s",
        args.head,
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
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    val_predictions_scaled = predict(model, x_val, device)
    test_predictions_scaled = predict(model, test_split.features, device)

    val_predictions = val_predictions_scaled * target_std + target_mean
    test_predictions = test_predictions_scaled * target_std + target_mean

    val_metrics = compute_metrics(val_predictions, y_val)
    test_metrics = compute_metrics(test_predictions, test_split.targets)

    output_dir = args.output_root / "regressors" / args.head
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "target_mean": target_mean,
            "target_std": target_std,
            "input_dim": train_split.features.shape[1],
            "targets": TARGET_COLUMNS,
            "head": args.head,
        },
        output_dir / "best_model.pt",
    )

    metrics_payload = {
        "head": args.head,
        "device": device,
        "best_val_loss": best_val_loss,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    run_config = {
        "embeddings_root": str(args.embeddings_root),
        "output_root": str(args.output_root),
        "head": args.head,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "device": device,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    write_predictions_csv(
        output_dir / "predictions_test.csv",
        dish_ids=test_split.dish_ids,
        predictions=test_predictions,
        targets=test_split.targets,
    )
    LOGGER.info("Saved regressor outputs to %s", output_dir)
