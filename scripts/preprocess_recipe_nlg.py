"""
Filters the recipes in data/recipes-with-nutrition.csv down to only those which contain the same subset of ingredients
as the Nutrition5k dataset (with the exclusion of spices).

Downloads just the metadata of Nutrition5k and filters the recipes by ingredient.
It also applies some additional filters to remove desserts, sauces and other recipes of that type.
"""

import os
import re
import csv
import json
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict

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
        "--stats", type=bool, default=False,
        help="Whether or not the script will collect statistics about ingredient occurences,"
        "used for further specifying the script (default: False)"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("./data"),
        help="Directory where nutrition50k/metadata will be deposited (default: ./data)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./out"),
        help="Directory where results are saved (default: ./out)"
    )
    parser.add_argument(
        "--recipe-file", type=Path, default="recipes-with-nutrition.csv",
        help="Path to recipes with nutrition CSV (default: recipes-with-nutrition.csv)"
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

def extract_ingredients(data_dir: Path, output_dir: Path, stats: bool):
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
        ingr_json = output_dir / "nutrition5k_ingredients.json"
    
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
            output_dir / "nutrition5k_multiword_word_freq.csv",
            output_dir / "nutrition5k_multiword_word_to_phrases.json",
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
# STEP 3 — Filter RecipeNLG for Nutrition5k-compatible recipes
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


def filter_recipes(nutrition5k_ingredients: set, output_dir: Path, recipe_dir: Path, stats: bool):
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

    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        corr_dish_type = False
        for dish_type in row["dish_type"]:
            if dish_type in ALLOWED_DISH_TYPE:
                corr_dish_type = True
                break
        
        # skip recipes not of the correct dish type
        if not corr_dish_type:
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
        with open(output_dir / "disallowed_ingr.json", "w") as f:
            json.dump(dict(sorted(disallowed_ingr.items(), key=lambda item: item[1], reverse=True)), f, indent=2)
    
    if stats:
        rnlg_word_freq, rnlg_word_to_phrases = build_multiword_index(disallowed_ingr)
        save_multiword_analysis(
            rnlg_word_freq,
            rnlg_word_to_phrases,
            output_dir / "rnlg_multiword_word_freq.csv",
            output_dir / "rnlg_multiword_word_to_phrases.json",
            label="Recipe_NLG",
        )

    # Save
    recipes_csv  = output_dir / "filtered_recipes.csv"
    filtered_recipes = dataset.iloc[filtered_indices]
    filtered_recipes["norm_ingredients"] = pd.Series(normalised_ingredients)

    filtered_recipes.to_csv(recipes_csv, index=False)
    print(f"  Saved to {recipes_csv}")
    return filtered_recipes

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
 
    DATA_DIR   = args.data_dir
    OUTPUT_DIR = args.output_dir
    RECIPE_FILE = args.recipe_file
    RECIPE_PATH = DATA_DIR / RECIPE_FILE
    STATS = args.stats
 
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Download metadata CSVs ---
    download_metadata(DATA_DIR)

    # --- Step 2: Extract ingredients ---
    nutrition5k_ingredients = extract_ingredients(DATA_DIR, OUTPUT_DIR, stats=STATS)

    # --- Step 3: Filter recipes ---
    filtered_recipes = filter_recipes(nutrition5k_ingredients, OUTPUT_DIR, RECIPE_PATH, stats=STATS)

    print("\nRecipe filtering complete. Outputs saved to:", OUTPUT_DIR.resolve())

    # Output recipe stats
    cuisine_freq = {}
    for cuisine_l in filtered_recipes.cuisine_type:
        for cuisine in cuisine_l:
            cuisine_freq[cuisine] = cuisine_freq.get(cuisine, 0) + 1
        
    sorted_freq = dict(sorted(cuisine_freq.items(), key=lambda item: item[1], reverse=True))
    print("---- Cuisine Frequencies ----")
    for cuisine, freq in sorted_freq.items():
        print(f"{cuisine}: {freq}")