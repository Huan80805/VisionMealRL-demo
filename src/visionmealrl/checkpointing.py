from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def epoch_checkpoint_path(checkpoint_dir: Path, epoch: int) -> Path:
    return checkpoint_dir / f"epoch_{epoch:03d}.pt"


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object] | None) -> None:
    if not state:
        return
    python_state = state.get("python_random_state")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = state.get("numpy_random_state")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    torch_state = state.get("torch_random_state")
    if torch_state is not None:
        torch.random.set_rng_state(torch_state)

    cuda_state = state.get("torch_cuda_random_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def save_training_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_training_checkpoint(path: Path, device: str) -> dict[str, object]:
    return torch.load(path, map_location=device, weights_only=False)
