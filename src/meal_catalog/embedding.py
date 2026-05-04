"""
Generates the manifests for the recipes, taking the final recipes list in csv form containing all of the original columns 
of recipes-with-nutrition and converting it to the required form of the manifests:

catalog_manifest.csv - catalog_index, name, style, calories, mass, fat, carb, protein, image path

Then, downloads all the images to the appropriate paths
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import os
import json
from tqdm import tqdm
import ast
import requests
from PIL import Image
from io import BytesIO

from visionmealrl.embedding import load_openclip_model_and_preprocess, resolve_device
from visionmealrl.multitask.artifacts import load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the dish manifest using the original data format of recipes-with-nutrition."
    )
    parser.add_argument(
        "--recipe-file", type=Path, default=Path("./data/meal_catalog.csv"),
        help="Path to the final recipe catalog (default: ./data/meal_catalog.csv)"
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("./artifacts/catalog/open_clip_ViT-B-32_laion2b_s34b_b79k_overhead_rgb/finetuned_embeddings/train"),
        help="Directory where the resulting manifest and embeddings are saved (default: ./artifacts/catalog/open_clip_ViT-B-32_laion2b_s34b_b79k_overhead_rgb/finetuned_embeddings/train)"
    )
    parser.add_argument(
        "--image-root", type=Path, default=Path("./data/catalog/images"),
        help="Directory where images for embedding are saved (default: ./data/catalog/images)"
    )
    parser.add_argument(
        "--image-width", type=int, default=1920,
        help="The width that the image should be transformed / padded into."
    )
    parser.add_argument(
        "--image-height", type=int, default=1080,
        help="The height that the image should be transformed + padded into."
    )

    return parser.parse_args()


def create_manifest_download_images(recipes_fp: Path, image_root: Path, image_width: int, image_height: int) -> pd.DataFrame:
    recipes_df = pd.read_csv(recipes_fp)
    recipes_df["daily_values"] = recipes_df["daily_values"].apply(json.loads)
    
    catalog_id = pd.Series([f"catalog_{i}" for i in range(recipes_df.shape[0])])
    names = recipes_df.recipe_name.reset_index(drop=True)
    style = recipes_df.cuisine_type.reset_index(drop=True)
    image_path = []
    servings = recipes_df.servings.reset_index(drop=True)
    total_calories = recipes_df.calories.reset_index(drop=True) / servings
    total_mass = recipes_df.total_weight_g.reset_index(drop=True) / servings
    total_fat = recipes_df.daily_values.apply(lambda x: x["FAT"]["quantity"]).reset_index(drop=True) / servings
    total_carb = recipes_df.daily_values.apply(lambda x: x["CHOCDF"]["quantity"]).reset_index(drop=True) / servings
    total_protein = recipes_df.daily_values.apply(lambda x: x["PROCNT"]["quantity"]).reset_index(drop=True) / servings

    for index, row in tqdm(recipes_df.iterrows(), total=recipes_df.shape[0]):
        image_url = row["image_url"]
        image_fp = Path(os.path.join(image_root, f"catalog_{index}", "rgb.png"))
        image_path.append(image_fp)
        if not image_fp.exists():
            download_resize_image(image_url, image_width, image_height, image_fp)

    manifest = pd.DataFrame({
        "catalog_id": catalog_id,
        "recipe_name": names,
        "style": style,
        "image_path": image_path,
        "total_calories": total_calories,
        "total_mass": total_mass,
        "total_fat": total_fat,
        "total_carb": total_carb,
        "total_protein": total_protein
    })

    return manifest


def write_manifest(manifest: pd.DataFrame, output_fp: Path):
    manifest.to_csv(output_fp, index=False)


def download_resize_image(image_url: str, image_width: int, image_height: int, output_fp: Path):
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(image_url)
    response.raise_for_status()
    
    img = Image.open(BytesIO(response.content)).convert("RGB")
    
    scale = min(image_width / img.width, image_height / img.height)
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    
    padded = Image.new("RGB", (image_width, image_height), (0, 0, 0))
    x = (image_width - img.width) // 2
    y = (image_height - img.height) // 2
    padded.paste(img, (x, y))
    
    padded.save(output_fp)


def embed_catalog_images(manifest: pd.DataFrame, output_root: Path):
    checkpoint_path = Path(
        "artifacts/multitask/open_clip_ViT-B-32_laion2b_s34b_b79k_overhead_rgb/best_embedding_model.pt"
    )
    image_paths = manifest.image_path

    device = resolve_device("auto")
    checkpoint = load_checkpoint(checkpoint_path, device=device)

    clip_model, preprocess = load_openclip_model_and_preprocess(
        model_name=checkpoint["model_name"],
        pretrained=checkpoint["pretrained"],
        device=device,
    )
    clip_model.load_state_dict(checkpoint["clip_model_state_dict"])
    clip_model.eval()

    def load_image_tensor(image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as image:
            return preprocess(image.convert("RGB")).unsqueeze(0).to(device)
        
    image_tensors = torch.cat([load_image_tensor(p) for p in image_paths], dim=0).to(device)

    with torch.inference_mode():
        embeddings = clip_model.encode_image(image_tensors)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32)
    print("Catalog embeddings of shape: ")
    print(embeddings_np.shape)

    output_fp = os.path.join(output_root, "catalog_embeddings.npy")
    np.save(output_fp, embeddings_np)

    print(f"Catalog embeddings saved to: {output_fp}")


def main():
    args = parse_args()
    RECIPE_FILE = args.recipe_file
    OUTPUT_ROOT = args.output_root
    IMAGE_ROOT = args.image_root
    IMAGE_WIDTH = args.image_width
    IMAGE_HEIGHT = args.image_height

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_df = create_manifest_download_images(RECIPE_FILE, IMAGE_ROOT, IMAGE_WIDTH, IMAGE_HEIGHT)
    write_manifest(manifest_df, os.path.join(OUTPUT_ROOT, "catalog_manifest.csv"))
    embed_catalog_images(manifest_df, OUTPUT_ROOT)
    
if __name__ == "__main__":
    main()