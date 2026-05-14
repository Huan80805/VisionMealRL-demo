from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from tqdm import tqdm

from visionmealrl.embedding import load_openclip_model_and_preprocess, resolve_device
from visionmealrl.multitask.artifacts import load_checkpoint

GENERIC_INGREDIENTS = {
    "black pepper",
    "kosher salt",
    "oil",
    "pepper",
    "salt",
    "sea salt",
    "seasoning",
    "water",
}

INGREDIENT_ALIASES = {
    "bell peppers": "bell pepper",
    "carrots": "carrot",
    "eggs": "egg",
    "extra virgin olive oil": "olive oil",
    "extra-virgin olive oil": "olive oil",
    "green onions": "green onion",
    "onions": "onion",
    "scallions": "green onion",
    "tomatoes": "tomato",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build three-component meal representations for agent training."
    )
    parser.add_argument(
        "--recipe-file",
        type=Path,
        default=Path("data/meal_catalog.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/catalog/three_component/train"),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(
            "artifacts/multitask/open_clip_ViT-B-32_laion2b_s34b_b79k_overhead_rgb/"
            "best_embedding_model.pt"
        ),
    )
    parser.add_argument("--image-root", type=Path, default=Path("data/catalog/images"))
    parser.add_argument("--name-template", default="a food photo of {recipe_name}")
    parser.add_argument("--cuisine-template", default="{style} cuisine")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def parse_listish(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, list):
        return [str(v) for v in parsed]
    return [str(parsed)]


def normalize_ingredient(raw: str) -> str:
    text = raw.lower()
    text = re.sub(r"[^a-z\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = INGREDIENT_ALIASES.get(text, text)
    if text.endswith("es") and text[:-2] in INGREDIENT_ALIASES:
        text = INGREDIENT_ALIASES[text[:-2]]
    elif text.endswith("s") and text[:-1] in INGREDIENT_ALIASES:
        text = INGREDIENT_ALIASES[text[:-1]]
    return text


def parse_ingredient_rows(row: pd.Series) -> dict[str, float]:
    weights: Counter[str] = Counter()
    try:
        ingredients = ast.literal_eval(row["ingredients"])
    except (ValueError, SyntaxError):
        ingredients = []

    for ingredient in ingredients:
        if not isinstance(ingredient, dict):
            continue
        name = normalize_ingredient(str(ingredient.get("food", "")))
        if not name or name in GENERIC_INGREDIENTS:
            continue
        try:
            weight = float(ingredient.get("weight", 0.0))
        except (TypeError, ValueError):
            weight = 0.0
        if weight > 0.0:
            weights[name] += weight

    if weights:
        return dict(weights)

    fallback = Counter()
    for name in parse_listish(row.get("norm_ingredients", "[]")):
        normalized = normalize_ingredient(name)
        if normalized and normalized not in GENERIC_INGREDIENTS:
            fallback[normalized] += 1.0
    return dict(fallback)


def build_ingredient_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    ingredient_rows = [parse_ingredient_rows(row) for _, row in df.iterrows()]
    document_frequency = Counter()
    for row in ingredient_rows:
        document_frequency.update(row.keys())

    vocabulary = sorted(document_frequency)
    vocab_index = {name: idx for idx, name in enumerate(vocabulary)}
    vectors = np.zeros((len(df), len(vocabulary)), dtype=np.float32)
    n_rows = len(df)

    for row_idx, weights in enumerate(ingredient_rows):
        total_weight = sum(weights.values())
        if total_weight <= 0.0:
            continue
        for ingredient, weight in weights.items():
            col_idx = vocab_index[ingredient]
            share = weight / total_weight
            idf = math.log((1 + n_rows) / (1 + document_frequency[ingredient])) + 1.0
            vectors[row_idx, col_idx] = float(share * idf)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    zero_rows = norms[:, 0] <= 1e-8
    if np.any(zero_rows):
        raise ValueError(f"ingredient embedding has {int(zero_rows.sum())} zero rows")
    vectors = vectors / norms
    return vectors.astype(np.float32), vocabulary


def build_manifest(df: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    nutrients = df["total_nutrients"].apply(json.loads)
    servings = df["servings"].astype(float).replace(0, np.nan)
    manifest = pd.DataFrame({
        "catalog_id": [f"catalog_{idx}" for idx in range(len(df))],
        "recipe_name": df["recipe_name"],
        "style": df["cuisine_type"].astype(str),
        "meal_type": df["meal_type"],
        "dish_type": df["dish_type"],
        "image_path": [
            str(image_root / f"catalog_{idx}" / "rgb.png")
            for idx in range(len(df))
        ],
        "total_calories": df["calories"].astype(float) / servings,
        "total_mass": df["total_weight_g"].astype(float) / servings,
        "total_fat": nutrients.apply(lambda x: x["FAT"]["quantity"]) / servings,
        "total_carb": nutrients.apply(lambda x: x["CHOCDF"]["quantity"]) / servings,
        "total_protein": nutrients.apply(lambda x: x["PROCNT"]["quantity"]) / servings,
    })
    if manifest[["total_calories", "total_fat", "total_carb", "total_protein"]].isna().any().any():
        raise ValueError("manifest contains NaN nutrition values")
    return manifest


def embed_texts(
    texts: list[str],
    checkpoint_path: Path,
    batch_size: int,
    device_arg: str,
) -> np.ndarray:
    device = resolve_device(device_arg)
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    model, _ = load_openclip_model_and_preprocess(
        model_name=checkpoint["model_name"],
        pretrained=checkpoint["pretrained"],
        device=device,
    )
    model.load_state_dict(checkpoint["clip_model_state_dict"], strict=False)
    model.eval()
    tokenizer = open_clip.get_tokenizer(checkpoint["model_name"])

    autocast_enabled = device == "cuda"
    autocast_device = "cuda" if device == "cuda" else "cpu"
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc="text embedding"):
            tokens = tokenizer(texts[start : start + batch_size]).to(device)
            with torch.autocast(device_type=autocast_device, enabled=autocast_enabled):
                features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            batches.append(features.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(batches, axis=0)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.recipe_file)
    manifest = build_manifest(df, args.image_root)
    ingredient_embeddings, vocabulary = build_ingredient_embeddings(df)

    name_texts = [
        args.name_template.format(recipe_name=name)
        for name in manifest["recipe_name"].astype(str)
    ]
    cuisine_texts = [
        args.cuisine_template.format(style=style)
        for style in manifest["style"].astype(str)
    ]
    name_embeddings = embed_texts(
        name_texts,
        args.checkpoint_path,
        args.batch_size,
        args.device,
    )
    cuisine_embeddings = embed_texts(
        cuisine_texts,
        args.checkpoint_path,
        args.batch_size,
        args.device,
    )

    embedding_dim = (
        ingredient_embeddings.shape[1]
        + cuisine_embeddings.shape[1]
        + name_embeddings.shape[1]
    )

    manifest.to_csv(args.output_dir / "catalog_manifest.csv", index=False)
    np.save(args.output_dir / "ingredient_embeddings.npy", ingredient_embeddings)
    np.save(args.output_dir / "cuisine_embeddings.npy", cuisine_embeddings)
    np.save(args.output_dir / "name_embeddings.npy", name_embeddings)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source_recipe_file": str(args.recipe_file),
                "checkpoint_path": str(args.checkpoint_path),
                "name_template": args.name_template,
                "cuisine_template": args.cuisine_template,
                "ingredient_vocabulary": vocabulary,
                "ingredient_dim": int(ingredient_embeddings.shape[1]),
                "cuisine_dim": int(cuisine_embeddings.shape[1]),
                "name_dim": int(name_embeddings.shape[1]),
                "embedding_dim": int(embedding_dim),
                "generic_ingredients": sorted(GENERIC_INGREDIENTS),
                "ingredient_aliases": INGREDIENT_ALIASES,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Wrote manifest: {args.output_dir / 'catalog_manifest.csv'}")
    print(f"Ingredient embeddings: {ingredient_embeddings.shape}")
    print(f"Cuisine embeddings: {cuisine_embeddings.shape}")
    print(f"Name embeddings: {name_embeddings.shape}")
    print(f"Total representation dim: {embedding_dim}")


if __name__ == "__main__":
    main()
