from __future__ import annotations

import argparse
from pathlib import Path

from visionmealrl.benchmark import run_benchmark_main
from visionmealrl.multitask import train_multitask_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visionmealrl",
        description="Nutrition5K end-to-end baseline and multitask pipelines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline_parser = subparsers.add_parser(
        "run-baseline",
        help="Run the frozen-embedding baseline end to end and save benchmark outputs.",
    )
    baseline_parser.add_argument("--dataset-root", type=Path, required=True)
    baseline_parser.add_argument("--output-root", type=Path, required=True)
    baseline_parser.add_argument("--model-name", default="ViT-B-32")
    baseline_parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    baseline_parser.add_argument(
        "--image-source",
        choices=["overhead_rgb", "side_angles_frames"],
        default="overhead_rgb",
    )
    baseline_parser.add_argument("--extract-batch-size", type=int, default=64)
    baseline_parser.add_argument("--extract-num-workers", type=int, default=4)
    baseline_parser.add_argument("--head-batch-size", type=int, default=128)
    baseline_parser.add_argument("--head-epochs", type=int, default=50)
    baseline_parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    baseline_parser.add_argument("--head-weight-decay", type=float, default=1e-4)
    baseline_parser.add_argument("--validation-size", type=float, default=0.1)
    baseline_parser.add_argument("--seed", type=int, default=7)
    baseline_parser.add_argument("--device", default="auto")
    baseline_parser.add_argument("--run-name")
    baseline_parser.add_argument("--skip-extraction", action="store_true")

    multitask_parser = subparsers.add_parser(
        "run-multitask",
        help="Run the multitask finetuning pipeline with a shared encoder and selective unfreezing.",
    )
    multitask_parser.add_argument("--dataset-root", type=Path, required=True)
    multitask_parser.add_argument("--output-root", type=Path, required=True)
    multitask_parser.add_argument("--model-name", default="ViT-B-32")
    multitask_parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    multitask_parser.add_argument(
        "--image-source",
        choices=["overhead_rgb", "side_angles_frames"],
        default="overhead_rgb",
    )
    multitask_parser.add_argument("--batch-size", type=int, default=32)
    multitask_parser.add_argument("--num-workers", type=int, default=4)
    multitask_parser.add_argument("--epochs", type=int, default=20)
    multitask_parser.add_argument("--freeze-epochs", type=int, default=3)
    multitask_parser.add_argument("--head-lr", type=float, default=1e-3)
    multitask_parser.add_argument("--encoder-lr", type=float, default=1e-5)
    multitask_parser.add_argument("--weight-decay", type=float, default=1e-4)
    multitask_parser.add_argument("--dropout", type=float, default=0.1)
    multitask_parser.add_argument("--lambda-reg", type=float, default=1.0)
    multitask_parser.add_argument("--lambda-cls", type=float, default=1.0)
    multitask_parser.add_argument("--unfreeze-last-n-blocks", type=int, default=2)
    multitask_parser.add_argument("--unfreeze-projection", action="store_true")
    multitask_parser.add_argument("--validation-size", type=float, default=0.1)
    multitask_parser.add_argument("--seed", type=int, default=7)
    multitask_parser.add_argument("--device", default="auto")
    multitask_parser.add_argument("--run-name")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-multitask":
        train_multitask_main(args)
        return

    if args.command == "run-baseline":
        run_benchmark_main(args)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
