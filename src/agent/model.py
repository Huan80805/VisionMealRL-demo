"""Neural network modules for the DQN meal planning agent.

ModularEncoder encodes each observation modality independently before
concatenation, replacing the flat MLP used in the original agent.py.

NOTE — SB3 DQN is plain vanilla DQN, not Double DQN.  The target network
computes next-state Q-values via .max(dim=1) without decoupling action
selection from evaluation, which can cause Q-value overestimation.
Double DQN (Hasselt et al., 2016) uses the *online* network to select the
greedy action and the *target* network only to evaluate it.  Upgrading to
Double DQN is a natural next step once baseline results are established.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from agent.catalog import MealCatalog
from agent.config import AgentConfig


class ModularEncoder(BaseFeaturesExtractor):
    """Modular feature extractor for MealPlanningEnv observations.

    Encodes each state modality separately:

      deficit_enc  : Linear(13, 32)  + ReLU  — concat(daily_deficit, episode_deficit, daily_target, remaining_steps)
      time_enc     : Linear(mpd, 16) + ReLU  — time slot one-hot
      meal_emb_enc : Linear(emb, 128)+ ReLU  — mean recent meal embedding
      user_emb_enc : Linear(emb, 128)+ ReLU  — user preference embedding

    Output: 32 + 16 + 128 + 128 = 304-dim feature vector fed into the
    action-scoring Q-network.
    """

    FEATURES_DIM = 32 + 16 + 128 + 128  # = 304

    def __init__(self, observation_space: spaces.Box, cfg: AgentConfig):
        super().__init__(observation_space, features_dim=self.FEATURES_DIM)

        assert observation_space.shape[0] == cfg.obs_dim, (
            f"observation_space.shape[0]={observation_space.shape[0]} "
            f"!= cfg.obs_dim={cfg.obs_dim}.  "
            f"Ensure the env was built with the same AgentConfig."
        )

        mpd = cfg.meals_per_day
        emb = cfg.embedding_dim

        # Obs slice boundaries (must match MealPlanningEnv._build_obs)
        self._s_deficit = slice(0, 13)                       # daily(4) + episode(4) + target(4) + remaining(1)
        self._s_time = slice(13, 13 + mpd)                   # time slot one-hot
        self._s_meal_emb = slice(13 + mpd, 13 + mpd + emb)  # recent mean embedding
        self._s_user_pref = slice(13 + mpd + emb, 13 + mpd + 2 * emb)  # user pref

        self.deficit_enc = nn.Sequential(nn.Linear(13, 32), nn.ReLU())
        self.time_enc = nn.Sequential(nn.Linear(mpd, 16), nn.ReLU())
        self.meal_emb_enc = nn.Sequential(nn.Linear(emb, 128), nn.ReLU())
        self.user_emb_enc = nn.Sequential(nn.Linear(emb, 128), nn.ReLU())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        deficit = obs[:, self._s_deficit]
        time_slot = obs[:, self._s_time]
        meal_emb = obs[:, self._s_meal_emb]
        user_pref = obs[:, self._s_user_pref]

        d = self.deficit_enc(deficit)
        t = self.time_enc(time_slot)
        m = self.meal_emb_enc(meal_emb)
        u = self.user_emb_enc(user_pref)

        return torch.cat([d, t, m, u], dim=1)


def build_action_features(catalog: MealCatalog, cfg: AgentConfig) -> np.ndarray:
    """Build fixed per-action features aligned with env action decoding.

    Action order is meal-major and portion-minor:
    ``action = meal_idx * num_portions + portion_idx``.
    """
    nutrition_scale = np.array([2500.0, 150.0, 350.0, 100.0], dtype=np.float32)
    max_portion = float(max(cfg.portion_levels))
    rows: list[np.ndarray] = []
    for meal_idx in range(catalog.num_meals):
        embedding = catalog.get_embedding(meal_idx).astype(np.float32, copy=False)
        for portion in cfg.portion_levels:
            nutrition = catalog.get_nutrition(meal_idx, float(portion)).astype(
                np.float32, copy=False
            )
            rows.append(np.concatenate([
                embedding,
                nutrition / nutrition_scale,
                np.array([float(portion) / max_portion], dtype=np.float32),
            ]))
    return np.stack(rows, axis=0).astype(np.float32)


class ActionScoringQNetwork(QNetwork):
    """DQN-compatible Q-network that scores actions from fixed features.

    SB3 still receives ``q_values.shape == (batch_size, num_actions)``.
    Internally, state and action features are projected into a shared
    latent space and scored by a bounded cosine-style dot product plus a
    small learned action bias. This shares learning across nutritionally/
    visually similar actions instead of treating each action id as unrelated,
    while keeping Q-values on the same order as episode returns.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        action_features: np.ndarray,
        action_latent_dim: int = 128,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=[],
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )
        del net_arch
        self.q_net = nn.Identity()

        features = np.asarray(action_features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(
                f"action_features must be 2D, got shape {features.shape}"
            )
        if features.shape[0] != int(action_space.n):
            raise ValueError(
                f"action_features rows {features.shape[0]} != "
                f"num_actions {int(action_space.n)}"
            )

        self.action_latent_dim = int(action_latent_dim)
        self.register_buffer(
            "action_features",
            torch.as_tensor(features, dtype=torch.float32),
        )
        action_dim = int(features.shape[1])

        self.state_projector = nn.Sequential(
            nn.Linear(features_dim, 256),
            activation_fn(),
            nn.Linear(256, self.action_latent_dim),
            nn.Identity(),
        )
        self.action_projector = nn.Sequential(
            nn.Linear(action_dim, 256),
            activation_fn(),
            nn.Linear(256, self.action_latent_dim),
            nn.Identity(),
        )
        self.action_bias = nn.Sequential(
            nn.Linear(action_dim, 64),
            activation_fn(),
            nn.Linear(64, 1),
        )
        # State-value branch for dueling-style scoring. This carries the
        # horizon-dependent return level so the cosine action scorer does
        # not have to saturate every action just to represent future value.
        self.state_value = nn.Sequential(
            nn.Linear(self.action_latent_dim, 64),
            activation_fn(),
            nn.Linear(64, 1),
        )

    def forward(self, obs) -> torch.Tensor:
        state_features = self.extract_features(obs, self.features_extractor)
        state_latent = F.normalize(
            self.state_projector(state_features),
            p=2,
            dim=1,
            eps=1e-8,
        )
        action_latent = F.normalize(
            self.action_projector(self.action_features),
            p=2,
            dim=1,
            eps=1e-8,
        )
        advantage = state_latent @ action_latent.T
        q_scale = 10.0
        action_bias = 0.25 * torch.tanh(self.action_bias(self.action_features).T)
        advantage = q_scale * advantage + action_bias
        advantage = advantage - advantage.mean(dim=1, keepdim=True)
        state_value = 25.0 * torch.tanh(self.state_value(state_latent))
        return state_value + advantage

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                action_features=self.action_features.detach().cpu().numpy(),
                action_latent_dim=self.action_latent_dim,
            )
        )
        return data


class ActionScoringDQNPolicy(DQNPolicy):
    """SB3 DQN policy that swaps in ``ActionScoringQNetwork``."""

    def __init__(
        self,
        *args,
        action_features: np.ndarray,
        action_latent_dim: int = 128,
        **kwargs,
    ) -> None:
        self._action_features = np.asarray(action_features, dtype=np.float32)
        self._action_latent_dim = int(action_latent_dim)
        super().__init__(*args, **kwargs)

    def make_q_net(self) -> ActionScoringQNetwork:
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None
        )
        return ActionScoringQNetwork(
            **net_args,
            action_features=self._action_features,
            action_latent_dim=self._action_latent_dim,
        ).to(self.device)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                action_features=self._action_features,
                action_latent_dim=self._action_latent_dim,
            )
        )
        return data


def make_dqn(
    env,
    cfg: AgentConfig,
    catalog: MealCatalog | None = None,
    tensorboard_log: str | None = None,
):
    """Instantiate SB3 DQN with the catalog-aware action-scoring Q-network."""
    from stable_baselines3 import DQN

    if catalog is None:
        raise ValueError("make_dqn requires a MealCatalog for action scoring")

    policy_kwargs = dict(
        features_extractor_class=ModularEncoder,
        features_extractor_kwargs=dict(cfg=cfg),
        action_features=build_action_features(catalog, cfg),
        action_latent_dim=128,
    )

    return DQN(
        policy=ActionScoringDQNPolicy,
        env=env,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        target_update_interval=cfg.target_update_interval,
        exploration_fraction=cfg.exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=cfg.verbose,
        seed=cfg.seed,
    )
