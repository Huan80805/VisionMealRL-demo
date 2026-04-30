"""
Generates the manifests for the recipes, taking the final recipes list in csv form containing all of the original columns 
of recipes-with-nutrition and converting it to the required form of the manifests:

catalog_manifest.csv - catalog_index, name, style, calories, mass, fat, carb, protein, image path

Then, downloads all the images to the appropriate paths
"""

import argparse
from pathlib import Path
import pandas as pd
import os
import json
from tqdm import tqdm
import ast
import requests
from PIL import Image
from io import BytesIO

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the dish manifest using the original data format of recipes-with-nutrition."
    )
    parser.add_argument(
        "--recipe-file", type=Path, default=Path("./out/final_recipes.csv"),
        help="Path to the final recipe catalog (default: ./out/final_recipes.csv)"
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
    recipes_df["cuisine_type"] = recipes_df["cuisine_type"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    
    catalog_id = pd.Series([f"catalog_{i}" for i in range(recipes_df.shape[0])])
    names = recipes_df.recipe_name.reset_index(drop=True)
    style = recipes_df.cuisine_type.apply(lambda x: x[0]).reset_index(drop=True)
    total_calories = recipes_df.calories.reset_index(drop=True)
    image_path = []
    total_calories = recipes_df.calories.reset_index(drop=True)
    total_mass = recipes_df.total_weight_g.reset_index(drop=True)
    total_fat = recipes_df.daily_values.apply(lambda x: x["FAT"]["quantity"]).reset_index(drop=True)
    total_carb = recipes_df.daily_values.apply(lambda x: x["CHOCDF"]["quantity"]).reset_index(drop=True)
    total_protein = recipes_df.daily_values.apply(lambda x: x["PROCNT"]["quantity"]).reset_index(drop=True)

    for index, row in tqdm(recipes_df.iterrows(), total=recipes_df.shape[0]):
        image_url = row["image_url"]
        image_fp = Path(os.path.join(image_root, f"catalog_{index}", "rgb.png"))
        image_path.append(image_fp)
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

if __name__ == "__main__":
    main()