from __future__ import annotations

import argparse
from pathlib import Path

from visionmealrl.benchmark import run_benchmark_main
from visionmealrl.classification import train_classifier_main
from visionmealrl.embedding import extract_embeddings_main
from visionmealrl.regression import train_regressor_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visionmealrl",
        description="Nutrition5K CLIP embedding and regression pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract-embeddings",
        help="Extract CLIP embeddings for Nutrition5K splits.",
    )
    extract_parser.add_argument("--dataset-root", type=Path, required=True)
    extract_parser.add_argument("--output-root", type=Path, required=True)
    extract_parser.add_argument("--model-name", default="ViT-B-32")
    extract_parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    extract_parser.add_argument(
        "--image-source",
        choices=["overhead_rgb", "side_angles_frames"],
        default="overhead_rgb",
    )
    extract_parser.add_argument("--batch-size", type=int, default=64)
    extract_parser.add_argument("--num-workers", type=int, default=4)
    extract_parser.add_argument("--device", default="auto")
    extract_parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization of CLIP embeddings before saving.",
    )

    train_parser = subparsers.add_parser(
        "train-regressor",
        help="Train a dish-level regressor from extracted embeddings.",
    )
    train_parser.add_argument("--embeddings-root", type=Path, required=True)
    train_parser.add_argument("--output-root", type=Path, required=True)
    train_parser.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--hidden-dim", type=int, default=512)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--val-fraction", type=float, default=0.1)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--device", default="auto")

    classifier_parser = subparsers.add_parser(
        "train-classifier",
        help="Train a dish-level ingredient classifier from extracted embeddings.",
    )
    classifier_parser.add_argument("--dataset-root", type=Path, required=True)
    classifier_parser.add_argument("--embeddings-root", type=Path, required=True)
    classifier_parser.add_argument("--output-root", type=Path, required=True)
    classifier_parser.add_argument("--batch-size", type=int, default=128)
    classifier_parser.add_argument("--epochs", type=int, default=50)
    classifier_parser.add_argument("--learning-rate", type=float, default=1e-3)
    classifier_parser.add_argument("--weight-decay", type=float, default=1e-4)
    classifier_parser.add_argument("--val-fraction", type=float, default=0.1)
    classifier_parser.add_argument("--seed", type=int, default=7)
    classifier_parser.add_argument("--device", default="auto")
    classifier_parser.add_argument("--top-k", type=int, default=100)
    classifier_parser.add_argument("--ingredient-min-mass", type=float, default=5.0)
    classifier_parser.add_argument("--ingredient-min-fraction", type=float, default=0.02)
    classifier_parser.add_argument("--ranking-k", type=int, default=5)

    benchmark_parser = subparsers.add_parser(
        "run-benchmark",
        help="Run the baseline benchmark end to end and persist summary outputs.",
    )
    benchmark_parser.add_argument("--dataset-root", type=Path, required=True)
    benchmark_parser.add_argument("--output-root", type=Path, required=True)
    benchmark_parser.add_argument("--model-name", default="ViT-B-32")
    benchmark_parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    benchmark_parser.add_argument(
        "--image-source",
        choices=["overhead_rgb", "side_angles_frames"],
        default="overhead_rgb",
    )
    benchmark_parser.add_argument("--extract-batch-size", type=int, default=64)
    benchmark_parser.add_argument("--extract-num-workers", type=int, default=4)
    benchmark_parser.add_argument("--head-batch-size", type=int, default=128)
    benchmark_parser.add_argument("--head-epochs", type=int, default=50)
    benchmark_parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    benchmark_parser.add_argument("--head-weight-decay", type=float, default=1e-4)
    benchmark_parser.add_argument("--val-fraction", type=float, default=0.1)
    benchmark_parser.add_argument("--seed", type=int, default=7)
    benchmark_parser.add_argument("--device", default="auto")
    benchmark_parser.add_argument("--top-k", type=int, default=100)
    benchmark_parser.add_argument("--ingredient-min-mass", type=float, default=5.0)
    benchmark_parser.add_argument("--ingredient-min-fraction", type=float, default=0.02)
    benchmark_parser.add_argument("--ranking-k", type=int, default=5)
    benchmark_parser.add_argument("--run-name")
    benchmark_parser.add_argument("--skip-extraction", action="store_true")
    benchmark_parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization of CLIP embeddings before saving.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "extract-embeddings":
        extract_embeddings_main(args)
        return

    if args.command == "train-regressor":
        train_regressor_main(args)
        return

    if args.command == "train-classifier":
        train_classifier_main(args)
        return

    if args.command == "run-benchmark":
        run_benchmark_main(args)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
