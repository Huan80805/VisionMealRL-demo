"""agent — DQN meal planning agent package.

Public re-exports for convenient top-level access
"""

from agent.config import AgentConfig
from agent.catalog import MealTemplate, MealCatalog
from agent.user import SimulatedUser
from agent.env import MealPlanningEnv
from agent.model import ModularEncoder, make_dqn
from agent.baseline import HealthGreedy, MultiObjectiveGreedy, RandomPolicy
from agent.profiles import (
    NUTRITION_PERSONAS,
    TRAIN_STYLES,
    EVAL_STYLES,
    EvalUserSpec,
    apply_persona,
    make_training_resampler,
    no_op_resampler,
    build_eval_pool,
    make_dummy_style_template_lists,
)

__all__ = [
    "AgentConfig",
    "MealTemplate",
    "MealCatalog",
    "SimulatedUser",
    "MealPlanningEnv",
    "ModularEncoder",
    "make_dqn",
    "HealthGreedy",
    "MultiObjectiveGreedy",
    "RandomPolicy",
    "NUTRITION_PERSONAS",
    "TRAIN_STYLES",
    "EVAL_STYLES",
    "EvalUserSpec",
    "apply_persona",
    "make_training_resampler",
    "no_op_resampler",
    "build_eval_pool",
    "make_dummy_style_template_lists",
]
