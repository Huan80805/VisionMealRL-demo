from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from visionmealrl.classification import train_classifier_main
from visionmealrl.embedding import extract_embeddings_main, slugify_model_name
from visionmealrl.regression import train_regressor_main


def default_run_name(seed: int, top_k: int, ranking_k: int) -> str:
    return f"linear_top{top_k}_at{ranking_k}_seed{seed}"


def build_summary_row(
    args,
    benchmark_root: Path,
    embeddings_root: Path,
    regression_metrics: dict[str, object],
    classification_metrics: dict[str, object],
) -> dict[str, object]:
    regression_test = regression_metrics["test_metrics"]
    classification_test = classification_metrics["test_metrics"]

    row: dict[str, object] = {
        "run_name": benchmark_root.name,
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "image_source": args.image_source,
        "normalize_embeddings": not args.no_normalize,
        "seed": args.seed,
        "ingredient_top_k": args.top_k,
        "ingredient_min_mass": args.ingredient_min_mass,
        "ingredient_min_fraction": args.ingredient_min_fraction,
        "ranking_k": args.ranking_k,
        "embeddings_root": str(embeddings_root),
        "benchmark_root": str(benchmark_root),
        "regression_overall_mae": regression_test["overall_mae"],
        "regression_overall_rmse": regression_test["overall_rmse"],
        "regression_overall_normalized_mae": regression_test["overall_normalized_mae"],
        "classification_threshold": classification_test["threshold"],
        "classification_micro_map": classification_test["micro_map"],
        "classification_macro_map": classification_test["macro_map"],
        "classification_micro_f1": classification_test["micro_f1"],
        "classification_macro_f1": classification_test["macro_f1"],
        f"classification_precision_at_{args.ranking_k}": classification_test[f"precision_at_{args.ranking_k}"],
        f"classification_recall_at_{args.ranking_k}": classification_test[f"recall_at_{args.ranking_k}"],
    }

    for target_name, metrics in regression_test["per_target"].items():
        row[f"{target_name}_mae"] = metrics["mae"]
        row[f"{target_name}_rmse"] = metrics["rmse"]
        row[f"{target_name}_wape"] = metrics["wape"]
        row[f"{target_name}_r2"] = metrics["r2"]

    return row


def write_summary_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def append_summary_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_benchmark_main(args) -> None:
    model_dir_name = slugify_model_name(args.model_name, args.pretrained)
    embeddings_root = args.output_root / "embeddings" / model_dir_name / args.image_source

    run_name = args.run_name or default_run_name(
        seed=args.seed,
        top_k=args.top_k,
        ranking_k=args.ranking_k,
    )
    benchmark_root = args.output_root / "benchmarks" / model_dir_name / args.image_source / run_name
    benchmark_root.mkdir(parents=True, exist_ok=True)

    benchmark_config = {
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "run_name": run_name,
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "image_source": args.image_source,
        "extract_batch_size": args.extract_batch_size,
        "extract_num_workers": args.extract_num_workers,
        "head_batch_size": args.head_batch_size,
        "head_epochs": args.head_epochs,
        "head_learning_rate": args.head_learning_rate,
        "head_weight_decay": args.head_weight_decay,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "device": args.device,
        "normalize_embeddings": not args.no_normalize,
        "top_k": args.top_k,
        "ingredient_min_mass": args.ingredient_min_mass,
        "ingredient_min_fraction": args.ingredient_min_fraction,
        "ranking_k": args.ranking_k,
        "skip_extraction": args.skip_extraction,
    }
    with (benchmark_root / "benchmark_config.json").open("w", encoding="utf-8") as handle:
        json.dump(benchmark_config, handle, indent=2)

    if not args.skip_extraction:
        extract_embeddings_main(
            SimpleNamespace(
                dataset_root=args.dataset_root,
                output_root=args.output_root,
                model_name=args.model_name,
                pretrained=args.pretrained,
                image_source=args.image_source,
                batch_size=args.extract_batch_size,
                num_workers=args.extract_num_workers,
                device=args.device,
                no_normalize=args.no_normalize,
            )
        )

    regression_output_dir = benchmark_root / "regression"
    train_regressor_main(
        SimpleNamespace(
            embeddings_root=embeddings_root,
            output_root=args.output_root,
            output_dir=regression_output_dir,
            head="linear",
            batch_size=args.head_batch_size,
            epochs=args.head_epochs,
            learning_rate=args.head_learning_rate,
            weight_decay=args.head_weight_decay,
            hidden_dim=512,
            dropout=0.0,
            val_fraction=args.val_fraction,
            seed=args.seed,
            device=args.device,
        )
    )

    classification_output_dir = benchmark_root / "classification"
    train_classifier_main(
        SimpleNamespace(
            dataset_root=args.dataset_root,
            embeddings_root=embeddings_root,
            output_root=args.output_root,
            output_dir=classification_output_dir,
            batch_size=args.head_batch_size,
            epochs=args.head_epochs,
            learning_rate=args.head_learning_rate,
            weight_decay=args.head_weight_decay,
            val_fraction=args.val_fraction,
            seed=args.seed,
            device=args.device,
            top_k=args.top_k,
            ingredient_min_mass=args.ingredient_min_mass,
            ingredient_min_fraction=args.ingredient_min_fraction,
            ranking_k=args.ranking_k,
        )
    )

    with (regression_output_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        regression_metrics = json.load(handle)
    with (classification_output_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        classification_metrics = json.load(handle)

    summary_row = build_summary_row(
        args=args,
        benchmark_root=benchmark_root,
        embeddings_root=embeddings_root,
        regression_metrics=regression_metrics,
        classification_metrics=classification_metrics,
    )

    with (benchmark_root / "benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_row, handle, indent=2)
    write_summary_csv(benchmark_root / "benchmark_summary.csv", summary_row)
    append_summary_csv(args.output_root / "benchmarks" / "benchmark_runs.csv", summary_row)
