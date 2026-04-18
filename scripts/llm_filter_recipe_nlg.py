"""
llm_filter_recipe_nlg.py
 
Uses gpt-oss-120b via the HuggingFace transformers library to filter out:
  - Dessert recipes (e.g. caramel popcorn, cocoa mounds, brownies)
  - Branded/personal recipes (e.g. "Aunt Elsie's Lefse", "Dorice's Brownies")
 
Requirements:
  pip install transformers torch accelerate
 
Usage:
  python llm_filter_recipe_nlg.py
  python llm_filter_recipe_nlg.py --input out/filtered_recipes.json --output out/llm_filtered_recipes.json
  python llm_filter_recipe_nlg.py --batch-size 8 --max-recipes 500 --device cuda
"""
 
import json
import argparse
from pathlib import Path
 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
 
# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
 
MODEL_ID = "gpt-oss-120b"
 
FEW_SHOT_EXAMPLES = [
    # --- DESSERT examples (reject) ---
    {
        "title": "Caramel Popcorn",
        "ner": ["popcorn", "butter", "brown sugar", "corn syrup", "baking soda"],
        "label": "DESSERT",
        "reason": "Sweet snack/confection made with caramel — a dessert food.",
    },
    {
        "title": "Cocoa Mounds",
        "ner": ["cocoa", "sugar", "butter", "coconut", "chocolate chips"],
        "label": "DESSERT",
        "reason": "Chocolate/coconut confection — clearly a dessert.",
    },
    {
        "title": "Classic Chocolate Chip Cookies",
        "ner": ["flour", "butter", "sugar", "eggs", "chocolate chips", "vanilla"],
        "label": "DESSERT",
        "reason": "Cookies are a dessert baked good.",
    },
    {
        "title": "Strawberry Cheesecake",
        "ner": ["cream cheese", "sugar", "eggs", "strawberries", "graham crackers", "butter"],
        "label": "DESSERT",
        "reason": "Cheesecake is a dessert.",
    },
    # --- BRANDED examples (reject) ---
    {
        "title": "Aunt Elsie's Lefse",
        "ner": ["potatoes", "butter", "flour", "cream"],
        "label": "BRANDED",
        "reason": "Named after a specific person ('Aunt Elsie') — a personal/branded recipe.",
    },
    {
        "title": "Dorice's Brownies",
        "ner": ["chocolate", "butter", "sugar", "eggs", "flour"],
        "label": "BRANDED",
        "reason": "Named after a specific person ('Dorice') — a personal/branded recipe.",
    },
    {
        "title": "Grandma Helen's Pierogi",
        "ner": ["flour", "eggs", "potato", "cheddar cheese", "onion", "butter"],
        "label": "BRANDED",
        "reason": "Named after a specific person ('Grandma Helen') — a personal/branded recipe.",
    },
    {
        "title": "McDonald's Copycat Big Mac Sauce",
        "ner": ["mayonnaise", "relish", "mustard", "vinegar", "onion powder"],
        "label": "BRANDED",
        "reason": "References a commercial brand (McDonald's) — a branded recipe.",
    },
    # --- KEEP examples (pass) ---
    {
        "title": "Chicken Stir Fry",
        "ner": ["chicken breast", "broccoli", "soy sauce", "garlic", "ginger", "sesame oil"],
        "label": "KEEP",
        "reason": "Generic savory dish with no personal branding.",
    },
    {
        "title": "Beef and Vegetable Soup",
        "ner": ["beef", "carrots", "celery", "potatoes", "onion", "tomatoes", "beef broth"],
        "label": "KEEP",
        "reason": "Generic savory soup, not a dessert, not personally branded.",
    },
    {
        "title": "Spaghetti Bolognese",
        "ner": ["ground beef", "tomatoes", "onion", "garlic", "pasta", "olive oil"],
        "label": "KEEP",
        "reason": "Generic savory pasta dish.",
    },
]
 
SYSTEM_PROMPT = """\
You are a recipe classifier. For each recipe you are given, respond with exactly one of:
  KEEP      — a generic, savory (non-dessert), non-branded recipe
  DESSERT   — a dessert, sweet treat, confection, or sweet baked good
  BRANDED   — a recipe named after a specific person, family name, or commercial brand
 
Rules:
- A recipe is DESSERT if its primary character is sweet and it would typically be served as a dessert or sweet snack. This includes cakes, cookies, pies, candies, ice cream, sweet breads, etc.
- A recipe is BRANDED if the title contains a person's name (first, last, or relational like "Grandma", "Aunt", "Uncle"), a possessive ("'s"), or a commercial brand name.
- If a recipe is BOTH a dessert and branded, label it DESSERT.
- Respond with ONLY the label on the first line, then a brief one-sentence reason on the second line. No other text.
 
Example response format:
KEEP
Generic savory dish with no personal branding.
"""
 
 
def build_prompt(title: str, ner: list[str]) -> str:
    """Build the full few-shot chat prompt for a single recipe."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
 
    for ex in FEW_SHOT_EXAMPLES:
        user_content = f'Title: {ex["title"]}\nIngredients: {", ".join(ex["ner"])}'
        assistant_content = f'{ex["label"]}\n{ex["reason"]}'
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})
 
    # The actual recipe to classify
    messages.append({
        "role": "user",
        "content": f'Title: {title}\nIngredients: {", ".join(ner)}',
    })
 
    return messages
 
 
def parse_response(text: str) -> tuple[str, str]:
    """
    Extract label and reason from the model's response.
    Returns (label, reason) where label is one of KEEP / DESSERT / BRANDED / UNKNOWN.
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return "UNKNOWN", ""
 
    raw_label = lines[0].upper()
    label = raw_label if raw_label in {"KEEP", "DESSERT", "BRANDED"} else "UNKNOWN"
    reason = lines[1] if len(lines) > 1 else ""
    return label, reason
 
 
# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ──────────────────────────────────────────────────────────────────────────────
 
def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-based filter: remove dessert and branded recipes from filtered_recipes.json"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("out/filtered_recipes.json"),
        help="Path to input filtered_recipes.json (default: out/filtered_recipes.json)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("out/llm_filtered_recipes.json"),
        help="Path to write kept recipes (default: out/llm_filtered_recipes.json)"
    )
    parser.add_argument(
        "--rejected-output", type=Path, default=Path("out/llm_rejected_recipes.json"),
        help="Path to write rejected recipes with labels (default: out/llm_rejected_recipes.json)"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_ID,
        help=f"HuggingFace model ID (default: {MODEL_ID})"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to run on: cuda, cpu, mps (default: auto-detect)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Number of recipes to process in parallel (default: 4)"
    )
    parser.add_argument(
        "--max-recipes", type=int, default=None,
        help="Cap the number of recipes evaluated (default: no limit)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=64,
        help="Max tokens for the model's response per recipe (default: 64)"
    )
    return parser.parse_args()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
 
def main():
    args = parse_args()
 
    # --- Device ---
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
 
    # --- Load model ---
    print(f"\nLoading model: {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    print("  Model loaded.")
 
    # --- Load recipes ---
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
 
    with open(args.input) as f:
        recipes = json.load(f)
 
    if args.max_recipes:
        recipes = recipes[: args.max_recipes]
 
    print(f"\nLoaded {len(recipes):,} recipes from {args.input}")
 
    # --- Classify ---
    kept = []
    rejected = []
    batch_size = args.batch_size
 
    for batch_start in range(0, len(recipes), batch_size):
        batch = recipes[batch_start: batch_start + batch_size]
        batch_messages = [
            build_prompt(r["title"], r.get("ner", []))
            for r in batch
        ]
 
        # Apply chat template to each item in the batch
        batch_inputs = [
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            for msgs in batch_messages
        ]
 
        encodings = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
 
        with torch.no_grad():
            outputs = model.generate(
                **encodings,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,         # greedy — deterministic classification
                pad_token_id=tokenizer.eos_token_id,
            )
 
        # Decode only the newly generated tokens
        input_lengths = encodings["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(
            outputs[:, input_lengths:],
            skip_special_tokens=True,
        )
 
        for recipe, response_text in zip(batch, decoded):
            label, reason = parse_response(response_text)
            if label == "KEEP":
                kept.append(recipe)
            else:
                rejected.append({
                    **recipe,
                    "llm_label":  label,
                    "llm_reason": reason,
                })
 
        n_done = min(batch_start + batch_size, len(recipes))
        print(
            f"  [{n_done:>{len(str(len(recipes)))}}/{len(recipes)}] "
            f"kept={len(kept):,}  rejected={len(rejected):,}"
        )
 
    # --- Save results ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.rejected_output.parent.mkdir(parents=True, exist_ok=True)
 
    with open(args.output, "w") as f:
        json.dump(kept, f, indent=2)
 
    with open(args.rejected_output, "w") as f:
        json.dump(rejected, f, indent=2)
 
    total = len(kept) + len(rejected)
    print(f"\n✓ Done. {len(kept):,}/{total:,} recipes kept ({100*len(kept)/max(total,1):.1f}%)")
    print(f"  Kept     → {args.output}")
    print(f"  Rejected → {args.rejected_output}")
 
    # Label breakdown
    from collections import Counter
    label_counts = Counter(r["llm_label"] for r in rejected)
    for label, count in sorted(label_counts.items()):
        print(f"    {label}: {count:,}")
 
 
if __name__ == "__main__":
    main()