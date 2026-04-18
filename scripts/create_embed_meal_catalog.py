# ──────────────────────────────────────────────────────────────────────────────
# STEP 3.5 — Diversity-aware recipe selection
# ──────────────────────────────────────────────────────────────────────────────
 
def select_recipes(
    recipes: list,
    n: int,
    min_ingredient_freq: int = 5,
    decay: float = 1.0,
) -> list:
    """
    Select a diverse, ingredient-balanced subset of N recipes from a larger
    pool using greedy maximum-coverage with log-frequency ingredient weights.
 
    Algorithm:
      1. Compute w(i) = log(1 + f(i)) for each ingredient i, where f(i) is
         how many recipes in the pool contain i. Ingredients appearing in
         fewer than `min_ingredient_freq` recipes get w(i) = 0.
      2. Normalise all weights to [0, 1].
      3. Score each recipe as the mean weight of its ingredients (length-
         normalised to avoid penalising longer recipes).
      4. Greedily pick the highest-scoring recipe, then re-score remaining
         recipes discounting already-covered ingredients:
              w_adjusted(i) = w(i) / (1 + decay * c(i))
         where c(i) is how many times ingredient i appears in the selected
         set so far. Repeat until N recipes are chosen.
 
    Args:
        recipes:              list of recipe dicts (must have "ner" key)
        n:                    number of recipes to select
        min_ingredient_freq:  ingredients rarer than this are ignored (weight=0)
        decay:                coverage decay rate. Higher → more diversity,
                              lower → more popularity bias. Default 1.0.
 
    Returns:
        List of N selected recipe dicts, in selection order.
    """
    import math
    from collections import Counter, defaultdict
 
    if n >= len(recipes):
        print(f"  [select] Requested {n} >= pool size {len(recipes)}, returning all.")
        return recipes
 
    print(f"\n=== STEP 3.5: Selecting {n} recipes from {len(recipes):,} ===")
 
    # --- Build ingredient frequency table ---
    freq: Counter = Counter()
    recipe_ner: list[list[str]] = []
    for r in recipes:
        # Normalise and deduplicate NER items per recipe
        items = list({item.strip().lower() for item in r.get("ner", []) if item.strip()})
        recipe_ner.append(items)
        freq.update(items)
 
    # --- Compute log-frequency weights, zero out rare ingredients ---
    raw_w: dict[str, float] = {}
    for ingr, f in freq.items():
        raw_w[ingr] = math.log(1 + f) if f >= min_ingredient_freq else 0.0
 
    # Normalise to [0, 1]
    max_w = max(raw_w.values()) or 1.0
    w: dict[str, float] = {ingr: v / max_w for ingr, v in raw_w.items()}
 
    def recipe_score(ner_items: list[str], coverage: Counter) -> float:
        if not ner_items:
            return 0.0
        total = sum(
            w.get(ingr, 0.0) / (1.0 + decay * coverage[ingr])
            for ingr in ner_items
        )
        return total / len(ner_items)   # length-normalise
 
    # --- Greedy selection ---
    remaining = list(range(len(recipes)))
    selected_indices: list[int] = []
    coverage: Counter = Counter()
 
    # Initial scores (no coverage yet)
    scores = {idx: recipe_score(recipe_ner[idx], coverage) for idx in remaining}
 
    from tqdm import tqdm as _tqdm
    for step in _tqdm(range(n), desc="  Selecting recipes"):
        # Pick best
        best_idx = max(remaining, key=lambda idx: scores[idx])
        selected_indices.append(best_idx)
        remaining.remove(best_idx)
 
        # Update coverage counts
        coverage.update(recipe_ner[best_idx])
 
        # Re-score only the remaining recipes that share an ingredient with
        # the just-selected recipe (others are unaffected)
        touched_ingrs = set(recipe_ner[best_idx])
        for idx in remaining:
            if touched_ingrs & set(recipe_ner[idx]):
                scores[idx] = recipe_score(recipe_ner[idx], coverage)
 
    selected = [recipes[i] for i in selected_indices]
 
    # --- Report ---
    all_ingrs = {ingr for r in selected for ingr in r.get("ner", [])}
    print(f"  Selected {len(selected):,} recipes covering {len(all_ingrs):,} unique ingredients")
    top_covered = coverage.most_common(10)
    print(f"  Top covered ingredients: {', '.join(f'{i}({c})' for i, c in top_covered)}")
 
    return selected
 
 
# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — CLIP embeddings
# ──────────────────────────────────────────────────────────────────────────────
 
def embed_recipes_with_clip(filtered_recipes: list, output_dir: Path):
    """
    Embed recipe ingredient lists as text using CLIP's text encoder.
 
    Input text format per recipe:
        "A dish with ingredients: egg, cheese, broccoli, olive oil, salt, pepper"
 
    CLIP's text encoder is capped at 77 tokens. Ingredient lists are usually
    short enough to fit. Long lists are truncated automatically by CLIP.
 
    Output: numpy array of shape (N, D) saved as recipe_text_embeddings.npy
            plus a recipe_index.json mapping row → recipe title
    """
    import torch
    import open_clip
    import numpy as np
 
    print("\n=== STEP 4a: Embedding recipes with CLIP text encoder ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
 
    model, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME)
    model = model.to(device)
    model.eval()
 
    texts = []
    index = []
    for r in filtered_recipes:
        ingr_str = ", ".join(r["ner"]) if r["ner"] else ", ".join(r["ingredients"])
        text = f"A dish with ingredients: {ingr_str}"
        texts.append(text)
        index.append({"title": r["title"], "source_url": r["source_url"]})
 
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  Text batches"):
            batch = texts[i : i + BATCH_SIZE]
            tokens = open_clip.tokenize(batch, truncate=True).to(device)
            embs   = model.encode_text(tokens)
            embs   = embs / embs.norm(dim=-1, keepdim=True)   # L2 normalise
            all_embeddings.append(embs.cpu().numpy())
 
    embeddings = __import__("numpy").concatenate(all_embeddings, axis=0)
 
    emb_path   = output_dir / "recipe_text_embeddings.npy"
    index_path = output_dir / "recipe_index.json"
 
    __import__("numpy").save(str(emb_path), embeddings)
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
 
    print(f"  Saved text embeddings → {emb_path}  shape={embeddings.shape}")
    print(f"  Saved recipe index    → {index_path}")
    return embeddings
 
 
def retrieve_recipes(query_image_path: str, output_dir: Path, top_k: int = 5):
    """
    Given a path to a dish image, retrieve the top-k most similar recipes
    from the shared CLIP embedding space, ranked by cosine similarity
    between the image embedding and stored recipe text embeddings.
 
    Example:
        results = retrieve_recipes("my_dish.jpg", output_dir=Path("./out"), top_k=5)
    """
    import torch
    import open_clip
    import numpy as np
    from PIL import Image
 
    emb_path   = output_dir / "recipe_text_embeddings.npy"
    index_path = output_dir / "recipe_index.json"
 
    if not emb_path.exists():
        raise FileNotFoundError("Run embed_recipes_with_clip() first.")
 
    recipe_embeddings = np.load(str(emb_path))        # (N, D)
    with open(index_path) as f:
        recipe_index = json.load(f)
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME)
    model = model.to(device)
    model.eval()
 
    with torch.no_grad():
        image = preprocess(Image.open(query_image_path).convert("RGB"))
        image = image.unsqueeze(0).to(device)          # (1, C, H, W)
        image_emb = model.encode_image(image)
        image_emb = (image_emb / image_emb.norm(dim=-1, keepdim=True)).cpu().numpy()
 
    sims    = (recipe_embeddings @ image_emb.T).squeeze()   # (N,)
    top_idx = sims.argsort()[::-1][:top_k]
 
    results = [{"rank": i+1, "score": float(sims[j]), **recipe_index[j]}
               for i, j in enumerate(top_idx)]
    print(f"\nTop {top_k} recipes for image: '{query_image_path}'")
    for r in results:
        print(f"  #{r['rank']}  score={r['score']:.4f}  title={r['title']}")
    return results