"""
Filters the recipes in data/recipes-with-nutrition.csv down to only those which contain the same subset of ingredients
as the Nutrition5k dataset (with the exclusion of spices).

Downloads just the metadata of Nutrition5k and filters the recipes by ingredient.
It also applies some additional filters to remove desserts, sauces and other recipes of that type.
"""

import os
import re
import csv
import math
import json
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ──────────────────────────────────────────────────────────────────────────────
 
def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter DataHiveAI recipes down to Nutrition5k-compatible meal recipes."
    )
    parser.add_argument(
        "count", type=int, default=1000,
        help="How many recipes to be selected for the final embedding"
    )
    parser.add_argument(
        "--stats", type=bool, default=False,
        help="Whether or not the script will collect statistics about ingredient occurences,"
        "used for further specifying the script (default: False)"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("./data"),
        help="Directory where nutrition50k/metadata will be deposited (default: ./data)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./data"),
        help="Directory where results are saved (default: ./data)"
    )
    parser.add_argument(
        "--stats-dir", type=Path, default=Path("./stats"),
        help="Directory where stats are saved (default: ./stats)"
    )
    parser.add_argument(
        "--recipe-file", type=Path, default="recipes-with-nutrition.csv",
        help="Path to recipes with nutrition CSV (default: recipes-with-nutrition.csv)"
    )
    parser.add_argument(
        "--min_freq", type=int, default=5,
        help="Minimum number of times an ingredient must appear"
    )
    parser.add_argument(
        "--decay", type=float, default=1.0,
        help="How rapidly the frequency of a given ingredient contributes to its deprioritization"
    )

    return parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# Google Cloud Storage paths (no account needed for public bucket)
GCS_BUCKET = "gs://nutrition5k_dataset/nutrition5k_dataset"
METADATA_FILES = [
    "metadata/ingredients_metadata.csv"
]

CLIP_MODEL_NAME = "ViT-B/32"   # fast; swap to "ViT-L/14" for higher quality

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Download Nutrition5k metadata only
# ──────────────────────────────────────────────────────────────────────────────

def download_metadata(data_dir: Path):
    """Download only the metadata CSVs from Google Cloud Storage."""
    print("\n=== STEP 1: Downloading Nutrition5k metadata ===")
    meta_dir = data_dir / "nutrition5k"
    meta_dir.mkdir(parents=True, exist_ok=True)

    for rel_path in METADATA_FILES:
        dest = meta_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            print(f"  [skip] {rel_path} already exists")
            continue

        gcs_path = f"{GCS_BUCKET}/{rel_path}"
        print(f"  Downloading {gcs_path} → {dest}")
        result = subprocess.run(
            ["gsutil", "cp", gcs_path, str(dest)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"gsutil failed for {gcs_path}:\n{result.stderr}\n\n"
                "Make sure gsutil is installed: pip install gsutil\n"
                "No Google account is needed for this public bucket."
            )
    print("  Metadata download complete.")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Extract unique ingredient names from Nutrition5k
# ──────────────────────────────────────────────────────────────────────────────

def extract_ingredients(data_dir: Path, stats_dir: Path, stats: bool):
    """
    Parse ingredient_metadata.csv to collect
    every unique ingredient name that appears in the dataset.
 
    ingredient_metadata.csv columns: ingr_id, name, ...nutrition fields...
    """
    print("\n=== STEP 2: Extracting unique ingredients ===")
 
    ingredients = {}   # {id: ingredient name}
 
    # --- Primary source: ingredient_metadata.csv ---
    ingr_meta = data_dir / "nutrition5k/metadata/ingredients_metadata.csv"
    if ingr_meta.exists():
        df = pd.read_csv(ingr_meta, header=None)
        # Columns: ingr_id, name, [per-gram nutrition values ...]
        for _, row in df.iterrows():
            ingr_id = str(row.iloc[1]).strip()
            name    = str(row.iloc[0]).strip()
            if name and name.lower() != "nan":
                ingredients[ingr_id] = name
        print(f"  ingredient_metadata.csv → {len(ingredients)} entries")
    else:
        print("  WARNING: ingredient_metadata.csv not found; falling back to dish CSVs only")
 
    # Build sorted list of names
    all_ingredient_names = sorted({v.lower() for v in ingredients.values()})
    print(f"\n  Total unique ingredients: {len(all_ingredient_names)}")

    if stats:
        # --- Multi-word analysis for Nutrition5k ingredients ---
        ingredient_name_freq = {name : 1 for name in all_ingredient_names}
        n5k_word_freq, n5k_word_to_phrases = build_multiword_index(ingredient_name_freq)
    
        # Save ingredient lists
        ingr_json = stats_dir / "nutrition5k_ingredients.json"
    
        with open(ingr_json, "w") as f:
            json.dump({
                "ingredient_count": len(all_ingredient_names),
                "ingredients":      all_ingredient_names,
            }, f, indent=2)
    
        print(f"  Saved → {ingr_json}")
    
        # Save multi-word analysis
        save_multiword_analysis(
            n5k_word_freq,
            n5k_word_to_phrases,
            stats_dir / "nutrition5k_multiword_word_freq.csv",
            stats_dir / "nutrition5k_multiword_word_to_phrases.json",
            label="Nutrition5k",
        )
 
    return set(all_ingredient_names)

# ──────────────────────────────────────────────────────────────────────────────
# MULTI-WORD INGREDIENT ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
 
def build_multiword_index(phrases: dict[str, int]) -> tuple[dict, dict]:
    """
    Given a list of ingredient name strings, find all multi-word phrases and
    return two mappings:
 
      word_freq:        word → total occurrence count across all phrases
      word_to_phrases:  word → {phrase: count_of_that_phrase}
 
    Only phrases with more than one word (after normalisation) are considered.
    Single-word ingredients are ignored.
    """
    word_freq: dict[str, int] = defaultdict(int)
    # word → Counter-style dict of phrase → occurrence count
    word_to_phrases: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
 
    for phrase, freq in phrases.items():
        norm = normalise(phrase)
        tokens = norm.split()
        # if len(tokens) < 2:
        #     continue   # skip single-word entries
        for token in tokens:
            if len(token) < 2:
                continue
            word_freq[token] += freq
            word_to_phrases[token][norm] += freq
 
    # Convert defaultdicts to plain dicts for serialisation
    return (
        dict(sorted(word_freq.items(), key=lambda x: -x[1])),
        {w: dict(sorted(phrases.items(), key=lambda x: -x[1]))
         for w, phrases in sorted(word_to_phrases.items(), key=lambda x: -word_freq[x[0]])},
    )
 
 
def save_multiword_analysis(
    word_freq: dict,
    word_to_phrases: dict,
    freq_csv_path: Path,
    phrases_json_path: Path,
    label: str = "",
):
    """Write the two multi-word mappings to disk."""
    # word_freq → CSV
    pd.DataFrame(
        [{"word": w, "occurrences": c} for w, c in word_freq.items()]
    ).to_csv(freq_csv_path, index=False)
 
    # word_to_phrases → JSON
    with open(phrases_json_path, "w") as f:
        json.dump(word_to_phrases, f, indent=2)
 
    prefix = f"  [{label}] " if label else "  "
    print(f"{prefix}Multi-word word-freq  → {freq_csv_path}")
    print(f"{prefix}Multi-word word→phrases → {phrases_json_path}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Filter Recipes-With-Nutrition for Nutrition5k-compatible recipes
# ──────────────────────────────────────────────────────────────────────────────
def normalise(raw: str) -> str:
    """
    Lowercase, strip numbers/punctuation, collapse whitespace to normalize.

    Remove common prefixes like fresh, ground, dried
    """
    text = raw.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    prefixes = {"fresh", "ground", "dried", "canned", "diced", "coarse", "chopped", "can", "frozen", 
                "crushed", "firm", "silken", "grated", "light", "skinless"}
    
    suffixes = {"leaves", "kernels", "florets", "root", "sprigs", "halves", "stick", "pods"}
    
    components = text.split(" ")
    if components[0] in prefixes:
        text = " ".join(components[1:])

    components = text.split(" ")
    if components[-1] in suffixes:
        text = " ".join(components[:-1])

    if text in EXCEPTIONS:
        return EXCEPTIONS[text]
    if text.endswith("es"):
        if text[:-2] in EXCEPTIONS:
            return EXCEPTIONS[text[:-2]]
    if text.endswith("s"):
        if text[:-1] in EXCEPTIONS:
            return EXCEPTIONS[text[:-1]]
    elif text + "s" in EXCEPTIONS:
        return EXCEPTIONS[text + "s"]
    elif text + "es" in EXCEPTIONS:
        return EXCEPTIONS[text + "es"]
    
    return text
    

def ingredient_is_allowed(text: str, allowed: set) -> bool:
    """
    Return True if the recipe ingredient string refers only to items in
    `allowed` (Nutrition5k names) or `spices` (allowlist).

    Allow for plural forms and manual exceptions.
    """
    # empty after stripping
    if not text:
        return False
    
    # Direct match against allowed sets
    if text in allowed:
        return True
    if text.endswith("es"): # plurals
        if text[:-2] in allowed:
            return True
    if text.endswith("s"):
        if text[:-1] in allowed:
            return True
    elif text + "s" in allowed:
        return True
    elif text + "es" in allowed:
        return True
    
    # If it's just an adjective or some other descriptor before an allowed ingredient
    components = text.split(" ")
    if len(components) > 1:
        last = components[-1]
        return ingredient_is_allowed(last, allowed)
    
    return False


# Spice/seasoning allowlist — anything here is always permitted in recipes,
# not considered to add substantial macronutrient value
SPICE_ALLOWLIST = {
    "salt", "pepper", "black pepper", "white pepper", "red pepper flakes",
    "cayenne", "cayenne pepper", "paprika", "smoked paprika", "cumin", "coriander", "turmeric",
    "cinnamon", "nutmeg", "cloves", "cardamom", "ginger", "garlic powder",
    "onion powder", "chili powder", "oregano", "thyme", "rosemary", "basil",
    "parsley", "bay leaf", "bay leaves", "bay leaf", "dill", "sage", "chive",
    "marjoram", "tarragon", "allspice", "anise", "star anise", "fennel seeds", 
    "celery seed", "mustard seeds", "mustard powder", "saffron", "sumac", "za'atar",
    "five spice", "curry powder", "garam masala", "italian seasoning",
    "herbs de provence", "old bay", "seasoning", "spice mix",
    "salt and pepper", "kosher salt", "sea salt", "fleur de sel", "worcestershire sauce",
    "mint", "sweet paprika", "chile powder", "chilli powder", "old bay seasoning", "five spice powder",
    "fenugreek"
}

EXCEPTIONS = {"extra virgin olive oil": "olive oil", "virgin olive oil": "olive oil", "red onion": "onions", 
                  "unsalted butter": "butter", "spinach": "spinach (cooked)", "all purpose flour": "flour", 
                  "dijon mustard": "mustard", "lime juice": "lime juice", "chicken broth": "chicken stock",
                  "scallions": "green onions", "yellow onion": "onions", "red bell pepper": "bell peppers", 
                  "canola oil": "vegetable oil", "red wine vinegar": "vinegar", "chicken stock": "chicken stock",
                  "parmesan": "parmesan cheese", "heavy cream": "cream", "green bell pepper": "bell pepper", 
                  "apple cider vinegar": "vinegar", "vegetable broth": "vegetable stock", "dry white wine": "white wine",
                  "cornstarch": "corn starch", "boneless skinless chicken breasts": "chicken breast",
                  "vegetable stock": "vegetable stock", "red pepper": "bell pepper", "breadcrumbs": "bread crumbs",
                  "mozzarella": "mozzarella cheese", "white wine vinegar": "vinegar", "white onion": "onions",
                  "butternut squash": "squash", "tomato sauce": "tomato sauce", "low sodium soy sauce": "soy sauce",
                  "beans": "black beans", "herbs": "herbs", "baking powder": "baking powder", "beef stock": "beef stock",
                  "panko breadcrumbs": "bread crumbs", "baking soda": "baking soda", "crabmeat": "crab",
                  "garbanzo beans": "chickpeas", "reduced sodium soy sauce": "soy sauce", "mayo": "mayonnaise",
                  "cheddar": "cheddar cheese", "cannellini beans": "white beans", "marinara sauce": "marinara sauce",
                  "feta": "feta cheese", "skinless boneless chicken breast": "chicken breast", 
                  "panko bread crumbs": "bread crumbs", "panko": "bread crumbs", "low sodium chicken stock": "chicken stock",
                  "yoghurt": "yogurt"}

ALLOWED_DISH_TYPE = {"main course", "salad", "soup", "sandwiches", "starter"}


def filter_recipes(nutrition5k_ingredients: set, output_dir: Path, recipe_dir: Path, stats: bool, stats_dir: Path):
    """
    Load recipes-with-nutrition and keep recipes where every ingredient is either in
    nutrition5k_ingredients or SPICE_ALLOWLIST.

    The ingredients column of the recipe dataset comes in objects which come with the keys:
    - food - food name
    - text - raw text of the line of the recipe the ingredient was extracted from
    - weight - the weight of the ingredient in the recipe
    - measure - what sized measuring tool the ingredient is measured in
    - quantity - how many of that measuring tool was requested

    We just use the "food" key to filter our recipes.

    The main dish_types that we want to accept are: 'main course', 'salad', 'soup', 'sandwiches', 'starter'.
    """
    print("\n=== STEP 3: Filtering recipes-with-nutrition recipes ===")

    if not recipe_dir.exists():
        print(f"recipes-with-nutrition dataset not found at {recipe_dir}.")
        return []

    print(f"  Loading {recipe_dir} …")

    filtered_indices = []
    normalised_ingredients = []
    disallowed_ingr = dict()

    dataset = pd.read_csv(recipe_dir)
    dataset["cuisine_type"] = dataset["cuisine_type"].apply(json.loads)
    dataset["dish_type"] = dataset["dish_type"].apply(json.loads)
    dataset["ingredients"] = dataset["ingredients"].apply(json.loads)

    dataset = dataset.drop_duplicates(subset=["image_url"], keep=False).reset_index(drop=True)

    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        corr_dish_type = False
        for dish_type in row["dish_type"]:
            if dish_type in ALLOWED_DISH_TYPE:
                corr_dish_type = True
                break
        # skip recipes not of the correct dish type
        if not corr_dish_type:
            continue

        # skip recipes without an image
        if row["image_url"] == None:
            continue

        # skip recipes with empty cuisine type
        if not row["cuisine_type"]:
            continue

        ingredients_list = [item["food"] for item in row["ingredients"]]
        cleaned_ingr = [normalise(item) for item in ingredients_list]
        filtered_ingr = [item for item in cleaned_ingr if item not in SPICE_ALLOWLIST]

        # Select only those recipes which have ingredients in the nutrition5k dataset
        if (len(filtered_ingr) > 2) and all(ingredient_is_allowed(item, nutrition5k_ingredients) 
                                            for item in filtered_ingr):
            filtered_indices.append(index)
            normalised_ingredients.append(filtered_ingr)
        else:
            for item in filtered_ingr:
                if not ingredient_is_allowed(item, nutrition5k_ingredients):
                    disallowed_ingr[item] = disallowed_ingr.setdefault(item, 0) + 1

    print(f"  Scanned {dataset.shape[0]:,} recipes. {len(filtered_indices):,} recipes pass the filter:"
        f"({100*len(filtered_indices)/dataset.shape[0]:.1f}%)")
    
    if stats:
        with open(stats_dir / "disallowed_ingr.json", "w") as f:
            json.dump(dict(sorted(disallowed_ingr.items(), key=lambda item: item[1], reverse=True)), f, indent=2)
    
    if stats:
        rnlg_word_freq, rnlg_word_to_phrases = build_multiword_index(disallowed_ingr)
        save_multiword_analysis(
            rnlg_word_freq,
            rnlg_word_to_phrases,
            stats_dir / "rnlg_multiword_word_freq.csv",
            stats_dir / "rnlg_multiword_word_to_phrases.json",
            label="Recipe_NLG",
        )

    recipes_csv  = output_dir / "filtered_recipes.csv"
    filtered_recipes = dataset.iloc[filtered_indices].reset_index(drop=True)
    filtered_recipes["norm_ingredients"] = pd.Series(normalised_ingredients)

    # Consolidate food classes
    for index, row in filtered_recipes.iterrows():
        cuisines = row["cuisine_type"]
        if "mediterranean" in cuisines: # if mediterranean and greek or italian or middle eastern, remove mediterranean
            if ("greek" in cuisines) or ("italian" in cuisines) or ("middle eastern" in cuisines):
                cuisines.remove("mediterranean")
                filtered_recipes.loc[index, "cuisine_type"] = cuisines
        if "asian" in cuisines: # if asian and any asian cuisine, remove asian
            if ("indian" in cuisines) or ("chinese" in cuisines) or ("south east asian" in cuisines) or \
                ("japanese" in cuisines) or ("korean" in cuisines):
                cuisines.remove("asian")
                filtered_recipes.loc[index, "cuisine_type"] = cuisines
        if "world" in cuisines: # if world and any other cuisine, remove world
            if len(cuisines) > 1:
                cuisines.remove("world")
                filtered_recipes.loc[index, "cuisine_type"] = cuisines

    # Make cuisine a single value
    filtered_recipes["cuisine_type"] = filtered_recipes["cuisine_type"].apply(lambda x: x[0])

    # Save
    filtered_recipes.to_csv(recipes_csv, index=False)
    print(f"  Saved to {recipes_csv}")
    return filtered_recipes


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 - Recipe Selection
# ──────────────────────────────────────────────────────────────────────────────
def select_recipes(
    recipes: pd.DataFrame,
    n: int,
    min_ingredient_freq: int = 5,
    decay: float = 1.0,
) -> list:
    """
    Select a diverse, ingredient-balanced subset of N recipes from a larger
    pool using log-frequency ingredient weights.

    Also selects an equal amount of recipes from each cuisine_type, distributing the
    excess to other cuisine types when not possible.

    Algorithm:
      1. Compute per-cuisine quotas: divide N as evenly as possible across
         cuisine types; if a cuisine has fewer recipes than its quota, its
         shortfall is redistributed to remaining cuisines proportionally.
      2. Compute w(i) = log(1 + f(i)) for each ingredient i, where f(i) is
         how many recipes in the pool contain i. Ingredients appearing in
         fewer than `min_ingredient_freq` recipes get w(i) = 0. Weights are
         computed once from the full pool and never recomputed.
      3. Normalise all weights to [0, 1].
      4. Score each recipe as the mean weight of its ingredients (length-
         normalised to avoid penalising longer recipes).
      5. Greedily pick the highest-scoring recipe, then re-score remaining
         recipes discounting already-covered ingredients:
              w_adjusted(i) = w(i) / (1 + decay * c(i))
         where c(i) is how many times ingredient i appears in the selected
         set so far. When a cuisine's quota is filled, all its remaining
         recipes are dropped from the candidate pool and scores are
         renormalised to sum to 1 across remaining candidates. Repeat
         until N recipes are chosen.

    Args:
        recipes:              DataFrame of recipes (must have "ner" and
                              "cuisine_type" columns)
        n:                    number of recipes to select
        min_ingredient_freq:  ingredients rarer than this are ignored (weight=0)
        decay:                coverage decay rate. Higher → more diversity,
                              lower → more popularity bias. Default 1.0.

    Returns:
        List of N selected recipe dicts, in selection order.
    """
    records = recipes.to_dict("records")

    if n >= len(records):
        print(f"  [select] Requested {n} >= pool size {len(records)}, returning all.")
        return records

    print(f"\n=== STEP 4: Selecting {n} recipes from {len(records):,} ===")

    # Compute per-cuisine quotas
    cuisine_indices = {}
    for idx, r in enumerate(records):
        cuisine = r["cuisine_type"]
        if cuisine in cuisine_indices:
            cuisine_indices[cuisine].append(idx)
        else:
            cuisine_indices[cuisine] = [idx]

    cuisines = list(cuisine_indices.keys())
    n_cuisines = len(cuisines)
    base_quota, remainder = divmod(n, n_cuisines)
    # Give one extra slot to the first `remainder` cuisines (arbitrary but deterministic)
    quotas = {c: base_quota + (1 if i < remainder else 0) for i, c in enumerate(cuisines)}

    # Redistribute quotas for cuisines that don't have enough recipes
    # Iterate until stable (redistribution may cascade)
    changed = True
    while changed:
        changed = False
        shortfall = 0
        capped = set()
        for c, quota in quotas.items():
            available = len(cuisine_indices[c])
            if available < quota:
                shortfall += quota - available
                quotas[c] = available
                capped.add(c)
                changed = True

        if shortfall:
            # Spread shortfall across uncapped cuisines, largest-remainder method
            uncapped = [c for c in cuisines if c not in capped]
            if not uncapped:
                break
            per, leftover = divmod(shortfall, len(uncapped))
            for i, c in enumerate(uncapped):
                quotas[c] += per + (1 if i < leftover else 0)

    print(f"  Cuisine quotas: { {c: quotas[c] for c in cuisines} }")

    # Build ingredient frequency table from the full pool
    freq = Counter()
    ingredients_lists = []
    for r in tqdm(records):
        items = r["norm_ingredients"]
        ingredients_lists.append(items)
        freq.update(items)

    # Compute log-frequency weights, zero out rare ingredients
    raw_w = {}
    for ingr, f in freq.items():
        raw_w[ingr] = math.log(1 + f) if f >= min_ingredient_freq else 0.0

    max_w = max(raw_w.values()) or 1.0
    w = {ingr: v / max_w for ingr, v in raw_w.items()} # normalized weights

    def recipe_score(recipe_ingredients: list[str], coverage: Counter) -> float:
        if not recipe_ingredients:
            return 0.0
        total = sum(
            w.get(ingr, 0.0) / (1.0 + decay * coverage[ingr])
            for ingr in recipe_ingredients
        )
        return total / len(recipe_ingredients)

    def renormalise(scores: dict[int, float]) -> dict[int, float]:
        """Rescale scores so they sum to 1 over current candidates."""
        total = sum(scores.values())
        if total <= 0:
            # Uniform fallback if all scores collapse to zero
            n_remaining = len(scores)
            return {idx: 1.0 / n_remaining for idx in scores} if n_remaining else {}
        return {idx: s / total for idx, s in scores.items()}

    # --- Greedy selection with per-cuisine quota enforcement ---
    remaining = set(range(len(records)))
    selected_indices = []
    coverage = Counter()
    cuisine_selected = Counter()

    scores = {idx: recipe_score(ingredients_lists[idx], coverage) for idx in remaining}
    scores = renormalise(scores)

    from tqdm import tqdm as _tqdm
    for step in _tqdm(range(n), desc="  Selecting recipes"):
        best_idx = max(remaining, key=lambda idx: scores[idx])
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

        cuisine = records[best_idx]["cuisine_type"]
        cuisine_selected[cuisine] += 1
        coverage.update(ingredients_lists[best_idx])

        # Drop entire cuisine from pool if its quota is now filled
        quota_filled = cuisine_selected[cuisine] >= quotas[cuisine]
        if quota_filled:
            evicted = {idx for idx in remaining
                       if (cuisine == records[idx]["cuisine_type"])}
            remaining -= evicted

        if not remaining:
            break

        # Re-score recipes that share an ingredient with the just-selected one
        touched_ingrs = set(ingredients_lists[best_idx])
        for idx in remaining:
            if touched_ingrs & set(ingredients_lists[idx]):
                scores[idx] = recipe_score(ingredients_lists[idx], coverage)

        # Prune scores dict to match remaining, then renormalise
        scores = {idx: scores[idx] for idx in remaining}
        scores = renormalise(scores)

    selected = [records[i] for i in selected_indices]

    # --- Report ---
    all_ingrs = {ingr for r in selected for ingr in r.get("norm_ingredients", [])}
    print(f"  Selected {len(selected):,} recipes covering {len(all_ingrs):,} unique ingredients")
    print(f"  Per-cuisine counts: {dict(cuisine_selected)}")
    top_covered = coverage.most_common(10)
    print(f"  Top covered ingredients: {', '.join(f'{i}({c})' for i, c in top_covered)}")

    return pd.DataFrame(selected)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    COUNT = args.count
    MIN_FREQ = args.min_freq
    DECAY = args.decay
    DATA_DIR   = args.data_dir
    OUTPUT_DIR = args.output_dir
    STATS_DIR = args.stats_dir
    RECIPE_FILE = args.recipe_file
    RECIPE_PATH = DATA_DIR / RECIPE_FILE
    STATS = args.stats
 
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Download metadata CSVs ---
    download_metadata(DATA_DIR)

    # --- Step 2: Extract ingredients ---
    nutrition5k_ingredients = extract_ingredients(DATA_DIR, STATS_DIR, stats=STATS)

    # --- Step 3: Filter recipes ---
    filtered_recipes = filter_recipes(nutrition5k_ingredients, OUTPUT_DIR, RECIPE_PATH, stats=STATS, stats_dir=STATS_DIR)

    print("\nRecipe filtering complete. Outputs saved to:", OUTPUT_DIR.resolve())

    final_recipes = select_recipes(filtered_recipes, COUNT, MIN_FREQ, DECAY)
    final_recipes.to_csv(OUTPUT_DIR / "final_recipes.csv", index=False)