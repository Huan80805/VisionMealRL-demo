from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class AgentConfig:
    """Central hyperparameter store for all agent experiments.

    Pass --num_days from the CLI to vary the planning horizon (Experiment 2).
    embedding_dim defaults to 512 to match CLIP ViT-B/32 output.
    """

    num_meals: int = 30
    embedding_dim: int = 512          # CLIP ViT-B/32 output
    num_days: int = 3                 # TUNABLE — Exp 2 variable
    meals_per_day: int = 3
    history_len: int = 6
    portion_levels: tuple = (0.75, 1.0, 1.25)
    daily_cal: float = 2000.0
    daily_protein: float = 80.0
    daily_carbs: float = 250.0
    daily_fat: float = 65.0
    w_health: float = 1.0
    w_diversity: float = 0.3
    w_preference: float = 0.2
    w_boundary: float = 0.5
    total_timesteps: int = 1_000_000
    learning_rate: float = 1e-4
    buffer_size: int = 100_000
    batch_size: int = 128
    gamma: float = 0.99
    target_update_interval: int = 500
    exploration_fraction: float = 0.3
    learning_starts: int = 1000
    seed: int = 42
    verbose: int = 0

    # --- Derived properties (not dataclass fields; not set manually) ---

    @property
    def horizon(self) -> int:
        """Total number of meal decisions per episode."""
        return self.num_days * self.meals_per_day

    @property
    def num_actions(self) -> int:
        """Total number of discrete actions (meals × portion levels)."""
        return self.num_meals * len(self.portion_levels)

    @property
    def obs_dim(self) -> int:
        """Flat observation dimension.

        Layout:
          [0:4]                         daily_deficit  (normalized)
          [4:8]                         episode_deficit (normalized)
          [8:12]                        daily_target   (scaled)
          [12:13]                       remaining_steps_fraction
          [13 : 13+meals_per_day]       time slot one-hot
          [13+mpd : 13+mpd+emb]         mean recent meal embedding
          [13+mpd+emb : 13+mpd+2*emb]  user preference embedding
        """
        return 13 + self.meals_per_day + 2 * self.embedding_dim

    # --- Utilities ---

    def validate(self) -> None:
        assert self.num_meals > 0, "num_meals must be positive"
        assert self.embedding_dim > 0, "embedding_dim must be positive"
        assert self.num_days > 0, "num_days must be positive"
        assert self.meals_per_day > 0, "meals_per_day must be positive"
        assert self.history_len > 0, "history_len must be positive"
        assert len(self.portion_levels) > 0, "portion_levels must not be empty"
        assert self.daily_cal > 0 and self.daily_protein > 0
        assert self.daily_carbs > 0 and self.daily_fat > 0

    def to_json(self, path: Path) -> None:
        """Serialize config to JSON for experiment reproducibility."""
        path = Path(path)
        data = asdict(self)          # tuple fields serialized as JSON arrays
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "AgentConfig":
        """Restore config from a JSON file written by to_json()."""
        path = Path(path)
        data = json.loads(path.read_text())
        # Keep old run folders evaluable after removing the architecture flag.
        data.pop("policy_arch", None)
        if "portion_levels" in data:
            data["portion_levels"] = tuple(data["portion_levels"])
        return cls(**data)
