"""
Requirements:
  pip install gsutil              # for downloading from Google Cloud Storage
  pip install pandas tqdm         # data handling
  pip install torch torchvision   # PyTorch (CPU or CUDA)
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
        description="Filter RecipeNLG recipes to Nutrition5k-compatible recipes."
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
        "--recipe-dir", type=Path, default=None,
        help="Path to RecipeNLG CSV (default: <data-dir>/recipe_nlg.csv)"
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
    "salt and pepper", "kosher salt", "sea salt", "fleur de sel"
}

# List for filtering out recipes that are baking recipes, consisting entirely of 
# baking ingredients
BAKING_LIST = {
    "flour", "baking soda", "baking powder", "white sugar", "powdered sugar", "sugar", "brown sugar",
    "vanilla", "vanilla extract", "eggs", "egg whites", "egg yolks", "milk", "butter", "sour cream", 
    "heavy cream", "cream", "lemon juice", "orange juice", "apple juice", "almonds", "cream cheese",
    "yeast", "water", "corn starch", "arrowroot", "lime juice"
}

COMBINED_LIST = BAKING_LIST | SPICE_ALLOWLIST

CLIP_MODEL_NAME = "ViT-B/32"   # fast; swap to "ViT-L/14" for higher quality

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Download Nutrition5k metadata (metadata CSVs only, not 190 GB videos)
# ──────────────────────────────────────────────────────────────────────────────

def download_metadata(data_dir: Path):
    """Download only the lightweight metadata CSVs from Google Cloud Storage."""
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
 
    ingredients = {}   # id → canonical name
 
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
        ingr_csv  = output_dir / "nutrition5k_ingredients.csv"
        ingr_json = output_dir / "nutrition5k_ingredients.json"
    
        pd.DataFrame({
            "ingredient_id":   list(ingredients.keys()),
            "ingredient_name": list(ingredients.values()),
        }).sort_values("ingredient_name").to_csv(ingr_csv, index=False)
    
        with open(ingr_json, "w") as f:
            json.dump({
                "ingredient_count": len(all_ingredient_names),
                "ingredients":      all_ingredient_names,
            }, f, indent=2)
    
        print(f"  Saved → {ingr_csv}")
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

def normalise(text: str) -> str:
    """Lowercase, strip numbers/punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    unit_words = {
        "cup", "cups", "tablespoon", "tablespoons", "tbsp", "teaspoon",
        "teaspoons", "tsp", "pound", "pounds", "lb", "lbs", "ounce", "ounces",
        "oz", "gram", "grams", "g", "kg", "kilogram", "kilograms",
        "liter", "liters", "milliliter", "milliliters", "ml", "l",
        "pinch", "dash", "handful", "bunch", "can", "cans", "jar", "jars",
        "package", "packages", "pkg", "slice", "slices", "piece", "pieces",
        "clove", "cloves", "sprig", "sprigs", "stalk", "stalks", "head",
        "medium", "large", "small", "extra", "fresh", "dried", "frozen",
        "chopped", "diced", "minced", "sliced", "grated", "peeled",
        "shredded", "cooked", "raw", "whole", "halved", "quartered",
        "optional", "to taste", "or more", "about", "approximately",
        "of", "ground", "drizzle", "lump", "knob", "lots", "rack", "drop",
        "drops", "turns", "turn", "julienne", "washed", "julienned",
        "grilled", "roast", "halves", "cold", "warm", "boiling", "very", "unsalted"
    }

    # keyword: ([exact matches], replacement phrase)
    mappings = {"green onion": (["green onion"], "green onions"),
                "onion": (["onion", "red onion", "yellow onion", "white onion", "brown onion",
                           "sweet onion", "purple onion"], "onions"), 
                "pepper": (["green pepper", "red pepper", "red bell pepper", "bell pepper",
                            "green bell pepper", "yellow bell pepper", "yellow pepper",
                            "orange bell pepper"], "bell peppers"),
                "chicken breasts": (["chicken breasts"], "chicken breast"),
                "chicken thigh": (["chicken thigh"], "chicken thighs"),
                "egg white": (["egg white"], "egg whites"),
                "egg yolk": (["egg yolk"], "egg yolks"), "egg": (["egg"], "eggs"),
                "virgin olive oil": (["extra virgin olive oil", "virgin olive oil"], "olive oil"),
                "oil": (["canola oil", "cooking oil", "corn oil", "sunflower oil", 
                         "grapeseed oil"], "vegetable oil"),
                "brown sugar": (["light brown sugar", "dark brown sugar"], "brown sugar"),
                "sugar": (["white sugar", "granulated sugar"], "sugar"),
                "tomato": (["tomato"], "tomatoes"), "vinegar": (["white vinegar"], "vinegar"),
                "cream": (["heavy cream"], "cream"), "broth": (["chicken broth", "beef broth"], "broth"),
                "mushroom": (["mushrooms"], "mushroom"), "carrot": (["carrots"], "carrot"),
                "coconut": (["coconut"], "coconuts"), "cornstarch": (["cornstarch"], "corn starch"),
                "beans": (["beans"], "black beans"), "kidney beans": (["red kidney beans"], "kidney beans"), 
                "garbanzo beans": (["garbanzo beans"], "chickpeas"), 
                "cannellini beans": (["cannellini beans"], "white beans"), "apples": (["apples"], "apple")}
    
    tokens = [t for t in text.split() if t not in unit_words and len(t) > 1]
    core = " ".join(tokens)

    for key, (matches, replacement) in mappings.items():
        if key in core:
            if core in matches:
                return replacement

    return core


def ingredient_is_allowed(raw: str, allowed: set, spices: set) -> bool:
    """
    Return True if the recipe ingredient string refers only to items in
    `allowed` (Nutrition5k names) or `spices` (allowlist).
 
    Strategy: strip quantities/units/prep notes, then check if the core
    noun phrase is covered. Uses a generous substring match.
    """
    if not raw:
        return True # empty after stripping - harmless
 
    # Direct match against allowed sets
    if raw in allowed or raw in spices:
        return True
    
    return False


def title_is_allowed(title: str):
    title = title.lower()
    disallowed = {"cake", "muffins", "muffin", "cookies", "cookie", "pie", "pudding", "punch", 
    "cobbler", "brownies", "cider", "sauce", "dressing"}
    return not any(item in title for item in disallowed)


def filter_recipes(nutrition5k_ingredients: set, output_dir: Path, recipe_dir: Path, stats: bool):
    """
    Load RecipeNLG and keep recipes where every ingredient is either in
    nutrition5k_ingredients or SPICE_ALLOWLIST.

    RecipeNLG CSV columns: (index), title, ingredients, directions, link, source, NER
    We use a normalized version of the NER (Named Entity Recognition) column to determine compatibility
    with the allowed ingredients.
    """
    print("\n=== STEP 3: Filtering RecipeNLG recipes ===")

    if not recipe_dir.exists():
        print(f"RecipeNLG dataset not found at {recipe_dir}.")
        return []

    print(f"  Loading {recipe_dir} …")

    filtered = []
    chunk_size = 10_000
    disallowed_ingr = dict()
    total_seen = 0
    
    for chunk in tqdm(pd.read_csv(recipe_dir, chunksize=chunk_size, on_bad_lines="skip")):
        for _, row in chunk.iterrows():
            # Parse NER field (pre-cleaned ingredient names as JSON list)
            try:
                ner_items = json.loads(str(row.get("NER", "[]")))
                title = str(row.get("title", ""))
            except (json.JSONDecodeError, TypeError):
                ner_raw = str(row.get("NER", ""))
                ner_items = [x.strip().strip("'\"") for x in ner_raw.strip("[]").split(",")]

            # Check every NER item
            cleaned_ingr = [item for item in [normalise(item) for item in ner_items] if item.strip()]
            filtered_ingr = [item for item in cleaned_ingr if item not in COMBINED_LIST]
            if (len(filtered_ingr) > 2) and title_is_allowed(title) and \
                all(ingredient_is_allowed(item, nutrition5k_ingredients, SPICE_ALLOWLIST)
                for item in cleaned_ingr) and not all(item in COMBINED_LIST for item in cleaned_ingr):
                
                filtered.append({
                    "title":       title,
                    "ner":         cleaned_ingr,
                    "ingredients":  str(row.get("ingredients", "")),
                    "directions":  str(row.get("directions", "")),
                    "source_url":  str(row.get("link", "")),
                })
            else:
                for item in cleaned_ingr:
                    if not ingredient_is_allowed(item, nutrition5k_ingredients, SPICE_ALLOWLIST):
                        disallowed_ingr[item] = disallowed_ingr.setdefault(item, 0) + 1
            
            total_seen += 1

    print(f"  Scanned {total_seen:,} recipes → {len(filtered):,} pass the filter "
          f"({100*len(filtered)/max(total_seen,1):.1f}%)")
    
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
    recipes_json = output_dir / "filtered_recipes.json"
    recipes_csv  = output_dir / "filtered_recipes.csv"

    with open(recipes_json, "w") as f:
        json.dump(filtered, f, indent=2)

    pd.DataFrame([{
        "title":       r["title"],
        "ingredients":  r["ingredients"],
        "directions":  r["directions"],
        "ner":         ", ".join(r["ner"]),
        "source_url":  r["source_url"],
    } for r in filtered]).to_csv(recipes_csv, index=False)

    print(f"  Saved → {recipes_json}")
    print(f"  Saved → {recipes_csv}")
    return filtered

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
 
    DATA_DIR   = args.data_dir
    OUTPUT_DIR = args.output_dir
    RECIPE_NLG_CSV = args.recipe_dir or DATA_DIR / "recipe_nlg.csv"
    STATS = args.stats
 
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Download metadata CSVs ---
    download_metadata(DATA_DIR)

    # --- Step 2: Extract ingredients ---
    nutrition5k_ingredients = extract_ingredients(DATA_DIR, OUTPUT_DIR, stats=STATS)

    # # --- Step 3: Filter recipes ---
    filtered_recipes = filter_recipes(nutrition5k_ingredients, OUTPUT_DIR, RECIPE_NLG_CSV, stats=STATS)

    print("\n✓ Recipe filtering complete. Outputs saved to:", OUTPUT_DIR.resolve())