"""User-profile helpers

Two regimes:
  - **Training distribution** (sampled IID per episode): continuous random
    targets from ``TARGET_RANGES`` + preference embedding sampled from a
    pool of *training-style* templates.
  - **Held-out eval pool**: a fixed grid of personas x held-out styles x
    seeds (default 5x8x2 = 80 episodes). Eval styles are disjoint from
    training styles; the policy's preference-embedding generalisation
    is the headline test.

Eval personas sit *inside* the training support — nothing is held out
on the target axis, the persona presets just discretise it for clean
per-persona breakdowns at evaluation time.

The shipped catalog artifact includes a ``style`` column. For real
catalogs, ``make_style_template_lists`` groups templates by that metadata.
For dummy catalogs with no style labels, it falls back to a deterministic
disjoint partition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from agent.catalog import MealCatalog, MealTemplate
from agent.user import SimulatedUser


# ---------------------------------------------------------------------------
# Training nutrition distribution
# ---------------------------------------------------------------------------

# Independent uniform per component; loose USDA / DRI envelope for adults
# widened so every persona below sits inside the support.
TARGET_RANGES: dict[str, tuple[float, float]] = {
    "daily_cal":     (1500.0, 3200.0),
    "daily_protein": (50.0,   200.0),
    "daily_carbs":   (130.0,  450.0),
    "daily_fat":     (35.0,   100.0),
}


def sample_random_targets(rng: np.random.Generator) -> dict[str, float]:
    """One IID draw from ``TARGET_RANGES``."""
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in TARGET_RANGES.items()}


# ---------------------------------------------------------------------------
# Eval personas (gridded targets inside the training support)
# ---------------------------------------------------------------------------

NUTRITION_PERSONAS: dict[str, dict[str, float]] = {
    "sedentary_female":    {"daily_cal": 1800.0, "daily_protein": 65.0,  "daily_carbs": 225.0, "daily_fat": 60.0},
    "active_male":         {"daily_cal": 2800.0, "daily_protein": 130.0, "daily_carbs": 320.0, "daily_fat": 80.0},
    "endurance_athlete":   {"daily_cal": 3000.0, "daily_protein": 100.0, "daily_carbs": 400.0, "daily_fat": 75.0},
    "high_protein_lifter": {"daily_cal": 2500.0, "daily_protein": 180.0, "daily_carbs": 220.0, "daily_fat": 70.0},
    "weight_loss":         {"daily_cal": 1600.0, "daily_protein": 90.0,  "daily_carbs": 150.0, "daily_fat": 55.0},
}


# ---------------------------------------------------------------------------
# Dietary-style partition
# ---------------------------------------------------------------------------

TRAIN_STYLES: tuple[str, ...] = (
    "american",
    "asian",
    "mediterranean",
    "central europe",
    "nordic",
    "chinese",
    "indian",
    "japanese",
)
EVAL_STYLES: tuple[str, ...] = (
    "french",
    "italian",
    "mexican",
    "south american",
    "eastern europe",
    "british",
    "middle eastern",
    "south east asian",
)


# ---------------------------------------------------------------------------
# Per-episode user mutators (primitives for the env.episode_resampler hook)
# ---------------------------------------------------------------------------

def apply_random_targets(user: SimulatedUser, rng: np.random.Generator) -> None:
    """Mutate ``user.daily_target`` / ``weekly_target`` from ``TARGET_RANGES``."""
    targets = sample_random_targets(rng)
    user.daily_target = np.array(
        [targets["daily_cal"], targets["daily_protein"],
         targets["daily_carbs"], targets["daily_fat"]],
        dtype=np.float64,
    )
    user.weekly_target = user.daily_target * 7.0


def apply_persona(user: SimulatedUser, persona_name: str) -> None:
    """Pin ``user`` targets to a named persona preset."""
    if persona_name not in NUTRITION_PERSONAS:
        raise KeyError(
            f"unknown persona {persona_name!r}; "
            f"available: {sorted(NUTRITION_PERSONAS)}"
        )
    p = NUTRITION_PERSONAS[persona_name]
    user.daily_target = np.array(
        [p["daily_cal"], p["daily_protein"], p["daily_carbs"], p["daily_fat"]],
        dtype=np.float64,
    )
    user.weekly_target = user.daily_target * 7.0


def apply_random_preference(
    user: SimulatedUser,
    template_pool: Sequence[MealTemplate],
    k_range: tuple[int, int],
    rng: np.random.Generator,
) -> None:
    """Redraw ``user.preference_embedding`` as the L2-normalised mean of
    *k* random templates from ``template_pool``.

    *k* is uniform in ``[k_range[0], k_range[1])`` (inclusive low,
    exclusive high — matches ``Generator.integers``). Capped at the pool
    size if the upper bound exceeds it.
    """
    if len(template_pool) == 0:
        raise ValueError("template_pool must be non-empty")
    lo, hi = k_range
    if hi <= lo:
        raise ValueError(f"k_range must satisfy hi > lo, got {k_range}")
    k = int(rng.integers(low=lo, high=hi))
    k = min(k, len(template_pool))
    indices = rng.choice(len(template_pool), size=k, replace=False)
    chosen = [template_pool[int(i)] for i in indices]
    embeddings = np.stack([t.embedding for t in chosen], axis=0)
    pref = embeddings.mean(axis=0)
    pref = pref / (np.linalg.norm(pref) + 1e-8)
    user.preference_embedding = pref.astype(np.float32)


# ---------------------------------------------------------------------------
# Training resampler factory (the canonical episode_resampler)
# ---------------------------------------------------------------------------

EpisodeResampler = Callable[[SimulatedUser, np.random.Generator], None]


def make_training_resampler(
    template_pool: Sequence[MealTemplate],
    k_range: tuple[int, int] = (5, 31),
    randomize_targets: bool = True,
    randomize_preference: bool = True,
) -> EpisodeResampler:
    """Build the callable consumed by ``MealPlanningEnv.episode_resampler``.

    On each ``env.reset()`` the user is mutated in place: targets are
    redrawn from ``TARGET_RANGES`` (if ``randomize_targets``) and the
    preference embedding is rebuilt from a random subset of
    ``template_pool`` (if ``randomize_preference``).

    ``template_pool`` must be the union of templates from the *training*
    styles (``TRAIN_STYLES``) so the policy never sees eval-style
    preferences during training.
    """
    def _resample(user: SimulatedUser, rng: np.random.Generator) -> None:
        if randomize_targets:
            apply_random_targets(user, rng)
        if randomize_preference:
            apply_random_preference(user, template_pool, k_range, rng)
    return _resample


def no_op_resampler() -> EpisodeResampler:
    """A resampler that does nothing — used at evaluation to keep the
    pre-built held-out user fixed across reset()s."""
    def _noop(user: SimulatedUser, rng: np.random.Generator) -> None:
        del user, rng
    return _noop


# ---------------------------------------------------------------------------
# Held-out eval pool (5 personas x 8 styles x n_seeds = 80 default)
# ---------------------------------------------------------------------------

@dataclass
class EvalUserSpec:
    """Single entry in the held-out eval pool."""

    persona: str
    style: str
    seed: int
    user: SimulatedUser


def build_eval_pool(
    style_template_lists: dict[str, Sequence[MealTemplate]],
    personas: Sequence[str] = tuple(NUTRITION_PERSONAS.keys()),
    eval_styles: Sequence[str] = EVAL_STYLES,
    n_seeds: int = 2,
    seed: int = 0,
) -> list[EvalUserSpec]:
    """Build the held-out evaluation pool.

    Cardinality = ``len(personas) x len(eval_styles) x n_seeds``
    (default ``5 x 8 x 2 = 80``). Each entry pins (persona, style, seed)
    into a ``SimulatedUser`` whose targets are the persona preset and
    whose preference embedding is the L2-normalised mean of
    ``style_template_lists[style]``.

    Args:
        style_template_lists: style filter; must contain every name in
            ``eval_styles``.
        personas: subset of ``NUTRITION_PERSONAS`` to grid over.
        eval_styles: subset of ``EVAL_STYLES`` to grid over.
        n_seeds: per-cell seed multiplicity (drives noise in
            ``preference_score`` and the env's bootstrap sampling).
        seed: base seed; per-cell seeds are derived from it deterministically.
    """
    missing = [s for s in eval_styles if s not in style_template_lists]
    if missing:
        raise KeyError(
            f"style_template_lists missing required eval styles: {missing}. "
            f"Got keys: {sorted(style_template_lists)}"
        )

    pool: list[EvalUserSpec] = []
    base_rng = np.random.default_rng(seed)
    for persona_name in personas:
        for style in eval_styles:
            for _ in range(n_seeds):
                cell_seed = int(base_rng.integers(low=0, high=2**31 - 1))
                templates = style_template_lists[style]
                preset = NUTRITION_PERSONAS[persona_name]
                user = SimulatedUser.from_templates(
                    templates,
                    daily_cal=preset["daily_cal"],
                    daily_protein=preset["daily_protein"],
                    daily_carbs=preset["daily_carbs"],
                    daily_fat=preset["daily_fat"],
                    seed=cell_seed,
                )
                pool.append(EvalUserSpec(
                    persona=persona_name,
                    style=style,
                    seed=cell_seed,
                    user=user,
                ))
    return pool


# ---------------------------------------------------------------------------
# Style-template construction
# ---------------------------------------------------------------------------

def make_catalog_style_template_lists(
    catalog: MealCatalog,
    style_names: Sequence[str],
) -> dict[str, list[MealTemplate]]:
    """Group real catalog templates by the catalog's style metadata."""
    out: dict[str, list[MealTemplate]] = {style: [] for style in style_names}
    wanted = set(style_names)
    for meal in catalog.meals:
        if meal.style in wanted:
            out[meal.style].append(meal)

    missing = [style for style in style_names if not out[style]]
    if missing:
        available = sorted({m.style for m in catalog.meals if m.style})
        raise KeyError(
            f"catalog missing required styles: {missing}. "
            f"Available styles: {available}"
        )
    return out


def make_style_template_lists(
    catalog: MealCatalog,
    style_names: Sequence[str],
    per_style: int = 30,
    seed: int = 0,
) -> dict[str, list[MealTemplate]]:
    """Use real style metadata when present, otherwise dummy partitioning.

    Dummy fallback is only for synthetic catalogs, whose ``MealTemplate``
    rows intentionally have no style labels.
    """
    has_style_metadata = any(m.style for m in catalog.meals)
    if has_style_metadata:
        return make_catalog_style_template_lists(catalog, style_names)
    return make_dummy_style_template_lists(
        catalog,
        style_names=style_names,
        per_style=per_style,
        seed=seed,
    )


def make_dummy_style_template_lists(
    catalog: MealCatalog,
    style_names: Sequence[str],
    per_style: int = 30,
    seed: int = 0,
) -> dict[str, list[MealTemplate]]:
    """Partition catalog templates into disjoint per-style buckets.

    The partition is *disjoint* across ``style_names`` (no template
    appears in two styles), mirroring the contract we expect from the
    real artifact: train styles and eval styles must not share recipes.
    """
    if per_style * len(style_names) > catalog.num_meals:
        raise ValueError(
            f"need >= {per_style * len(style_names)} templates in catalog, "
            f"got {catalog.num_meals}"
        )
    rng = np.random.default_rng(seed)
    indices = rng.permutation(catalog.num_meals)
    out: dict[str, list[MealTemplate]] = {}
    for i, style in enumerate(style_names):
        chunk = indices[i * per_style : (i + 1) * per_style]
        out[style] = [catalog.meals[int(j)] for j in chunk]
    return out
