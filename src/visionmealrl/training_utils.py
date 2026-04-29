from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def run_supervised_epoch(
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

        outputs = model(features)
        loss = loss_fn(outputs, targets)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = features.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def predict_numpy(model: nn.Module, features: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        inputs = torch.from_numpy(features.astype(np.float32)).to(device)
        outputs = model(inputs).detach().cpu().numpy()
    return outputs.astype(np.float32)
