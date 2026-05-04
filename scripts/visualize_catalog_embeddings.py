from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project catalogue embeddings to 2-D and color them by style."
    )
    parser.add_argument(
        "--embedding-file",
        type=Path,
        default=Path(
            "artifacts/catalog/open_clip_ViT-B-32_laion2b_s34b_b79k_overhead_rgb/"
            "finetuned_embeddings/train/catalog_embeddings.npy"
        ),
        help="Path to catalog_embeddings.npy.",
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=Path(
            "artifacts/catalog/open_clip_ViT-B-32_laion2b_s34b_b79k_overhead_rgb/"
            "finetuned_embeddings/train/catalog_manifest.csv"
        ),
        help="Path to catalog_manifest.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/catalog/open_clip_ViT-B-32_laion2b_s34b_b79k_overhead_rgb/"
            "finetuned_embeddings/train/visualizations"
        ),
        help="Directory for the projection CSV and plot.",
    )
    parser.add_argument(
        "--method",
        choices=["umap", "tsne", "pca"],
        default="umap",
        help="Dimensionality reduction method.",
    )
    parser.add_argument(
        "--style-column",
        default="style",
        help="Manifest column used for point colors.",
    )
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def configure_local_caches(output_dir: Path) -> None:
    cache_root = output_dir / ".cache"
    numba_cache = cache_root / "numba"
    mpl_cache = cache_root / "matplotlib"
    numba_cache.mkdir(parents=True, exist_ok=True)
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache.resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache.resolve()))


def project_embeddings(args: argparse.Namespace, embeddings):
    if args.method == "umap":
        from umap import UMAP

        reducer = UMAP(
            n_components=2,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.seed,
        )
        return reducer.fit_transform(embeddings)

    if args.method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            metric=args.metric,
            init="pca",
            learning_rate="auto",
            random_state=args.seed,
        )
        return reducer.fit_transform(embeddings)

    from sklearn.decomposition import PCA

    reducer = PCA(n_components=2, random_state=args.seed)
    return reducer.fit_transform(embeddings)


def plot_projection(args: argparse.Namespace, coords, manifest, plot_file: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    styles = manifest[args.style_column].fillna("unknown").astype(str)
    style_names = sorted(styles.unique())
    cmap = plt.get_cmap("tab20", len(style_names))

    fig, ax = plt.subplots(figsize=(11.5, 8.5), constrained_layout=True)
    for index, style_name in enumerate(style_names):
        mask = styles == style_name
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=24,
            alpha=0.78,
            color=cmap(index),
            label=f"{style_name} ({int(np.sum(mask))})",
            linewidths=0,
        )

    ax.set_title(
        f"Catalogue embedding {args.method.upper()} by {args.style_column} "
        f"(n={len(manifest)}, metric={args.metric})"
    )
    ax.set_xlabel(f"{args.method.upper()} 1")
    ax.set_ylabel(f"{args.method.upper()} 2")
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.legend(
        title=args.style_column,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False,
        markerscale=1.3,
        fontsize=8,
    )
    fig.savefig(plot_file, dpi=args.dpi)
    plt.close(fig)


def compute_style_metrics(args: argparse.Namespace, embeddings, coords, manifest) -> dict:
    import numpy as np
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    labels = manifest[args.style_column].fillna("unknown").astype(str).to_numpy()
    unique_labels, encoded = np.unique(labels, return_inverse=True)
    label_counts = {label: int(np.sum(labels == label)) for label in unique_labels}
    metrics: dict[str, object] = {
        "style_count": int(len(unique_labels)),
        "row_count": int(len(labels)),
        "random_style_match_baseline": float(
            sum((count / len(labels)) ** 2 for count in label_counts.values())
        ),
    }

    valid_mask = np.array([label_counts[label] > 1 for label in labels])
    valid_labels = labels[valid_mask]
    valid_unique_labels, valid_encoded = np.unique(valid_labels, return_inverse=True)
    if len(valid_unique_labels) > 1:
        metrics["silhouette_original_space"] = float(
            silhouette_score(embeddings[valid_mask], valid_encoded, metric=args.metric)
        )
        metrics["silhouette_projected_2d"] = float(
            silhouette_score(coords[valid_mask], valid_encoded, metric="euclidean")
        )

    for k in (5, 10):
        n_neighbors = min(k + 1, len(embeddings))
        nearest = NearestNeighbors(n_neighbors=n_neighbors, metric=args.metric)
        nearest.fit(embeddings)
        neighbor_indices = nearest.kneighbors(embeddings, return_distance=False)[:, 1:]
        same_style = labels[neighbor_indices] == labels[:, None]
        metrics[f"same_style_neighbor_rate_at_{k}"] = float(np.mean(same_style))

    return metrics


def gen_duplicate_image_report(manifest) -> dict:
    if "image_path" not in manifest.columns:
        return {}

    groups: dict[str, list[dict[str, str]]] = {}
    for row in manifest.to_dict(orient="records"):
        image_path = Path(row["image_path"])
        if not image_path.exists():
            continue
        image_hash = hashlib.md5(image_path.read_bytes()).hexdigest()
        groups.setdefault(image_hash, []).append(row)

    duplicate_rows = []
    for image_hash, rows in groups.items():
        if len(rows) <= 1:
            continue
        for row in rows:
            duplicate_rows.append(
                {
                    "image_hash": image_hash,
                    "duplicate_group_size": len(rows),
                    "catalog_id": row.get("catalog_id", ""),
                    "recipe_name": row.get("recipe_name", ""),
                    "style": row.get("style", ""),
                    "image_path": row.get("image_path", ""),
                }
            )

    return {
        "unique_image_hashes": int(len(groups)),
        "duplicate_image_groups": int(
            sum(1 for rows in groups.values() if len(rows) > 1)
        ),
        "duplicate_image_rows": int(
            sum(len(rows) for rows in groups.values() if len(rows) > 1)
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_local_caches(args.output_dir)

    import numpy as np
    import pandas as pd

    embeddings = np.load(args.embedding_file)
    manifest = pd.read_csv(args.manifest_file)

    if len(embeddings) != len(manifest):
        raise ValueError(
            "Embedding row count does not match manifest row count: "
            f"{len(embeddings)} != {len(manifest)}"
        )
    if args.style_column not in manifest.columns:
        raise ValueError(
            f"Style column '{args.style_column}' is missing from {args.manifest_file}"
        )

    coords = project_embeddings(args, embeddings)

    stem = (
        f"catalog_embeddings_{args.method}_{args.style_column}"
        f"_seed{args.seed}_neighbors{args.n_neighbors}"
    )
    metrics_file = args.output_dir / f"{stem}_metrics.json"
    plot_file = args.output_dir / f"{stem}.png"

    metrics = compute_style_metrics(args, embeddings, coords, manifest)
    metrics.update(gen_duplicate_image_report(manifest))
    metrics_file.write_text(json.dumps(metrics, indent=2) + "\n")
    plot_projection(args, coords, manifest, plot_file)

    counts = manifest[args.style_column].fillna("unknown").astype(str).value_counts()
    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Loaded manifest rows: {len(manifest)}")
    print(f"Styles: {len(counts)}")
    print(f"Wrote metrics: {metrics_file}")
    print(f"Wrote plot: {plot_file}")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
