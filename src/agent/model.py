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

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from agent.config import AgentConfig


class ModularEncoder(BaseFeaturesExtractor):
    """Modular feature extractor for MealPlanningEnv observations.

    Encodes each state modality separately:

      deficit_enc  : Linear(8, 32)   + ReLU  — concat(daily_deficit, weekly_deficit)
      time_enc     : Linear(mpd, 16) + ReLU  — time slot one-hot
      meal_emb_enc : Linear(emb, 128)+ ReLU  — mean recent meal embedding
      user_emb_enc : Linear(emb, 128)+ ReLU  — user preference embedding

    Output: 32 + 16 + 128 + 128 = 304-dim feature vector fed into SB3's
    shared MLP (net_arch=[256, 256]) before Q-value heads.
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
        self._s_deficit = slice(0, 8)                        # daily(4) + weekly(4)
        self._s_time = slice(8, 8 + mpd)                     # time slot one-hot
        self._s_meal_emb = slice(8 + mpd, 8 + mpd + emb)    # recent mean embedding
        self._s_user_pref = slice(8 + mpd + emb, 8 + mpd + 2 * emb)  # user pref

        self.deficit_enc = nn.Sequential(nn.Linear(8, 32), nn.ReLU())
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


def make_dqn(env, cfg: AgentConfig):
    """Instantiate a SB3 DQN with ModularEncoder and cfg-derived hyperparams."""
    from stable_baselines3 import DQN

    policy_kwargs = dict(
        features_extractor_class=ModularEncoder,
        features_extractor_kwargs=dict(cfg=cfg),
        net_arch=[256, 256],
    )

    return DQN(
        policy="MlpPolicy",
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
        verbose=1,
        seed=cfg.seed,
    )
