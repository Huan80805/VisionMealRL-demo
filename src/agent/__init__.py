"""agent — DQN meal planning agent package.

Public re-exports for convenient top-level access:

  from agent import AgentConfig, MealCatalog, MealPlanningEnv, make_dqn, ...
"""

from agent.config import AgentConfig
from agent.catalog import MealTemplate, MealCatalog
from agent.user import SimulatedUser
from agent.env import MealPlanningEnv
from agent.model import ModularEncoder, make_dqn
from agent.baseline import HealthGreedy, MultiObjectiveGreedy
from agent.evaluate import (
    EpisodeResult,
    AggregatedMetrics,
    evaluate_policy,
    compare_policies,
    print_comparison_table,
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
    "EpisodeResult",
    "AggregatedMetrics",
    "evaluate_policy",
    "compare_policies",
    "print_comparison_table",
]
