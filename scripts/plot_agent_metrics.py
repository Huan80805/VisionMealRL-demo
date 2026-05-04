from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


METRICS = [
    ("NGA", "Nutrition Goal Attainment"),
    ("EDC", "Episodic Deficit Closure"),
    ("Return", "Episode Return"),
    ("DDS", "Dietary Diversity Score"),
    ("PA", "Preference Alignment"),
]

POLICY_ORDER = ["DQN", "MultiObjectiveGreedy", "HealthGreedy", "Random"]
POLICY_COLORS = {
    "DQN": "#1f77b4",
    "MultiObjectiveGreedy": "#ff7f0e",
    "HealthGreedy": "#2ca02c",
    "Random": "#7f7f7f",
}
POLICY_MARKERS = {
    "DQN": "o",
    "MultiObjectiveGreedy": "s",
    "HealthGreedy": "^",
    "Random": "x",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot agent horizon-scaling metrics from agent_table.csv."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("runs/agent_report_plots/agent_table.csv"),
        help="Path to agent_table.csv.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("runs/agent_report_plots/agent_horizon_metrics_1x5.png"),
        help="Output plot path.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--ci-z",
        type=float,
        default=1.96,
        help="Normal critical value for confidence intervals. Default is 1.96 for 95%% CI.",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=80,
        help="Fallback evaluation episode count when the CSV has no N column.",
    )
    return parser.parse_args()


def configure_local_matplotlib_cache(output_file: Path) -> None:
    cache_dir = output_file.parent / ".cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))


def load_rows(input_csv: Path, default_n_episodes: int) -> list[dict[str, object]]:
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    fieldnames = set(reader.fieldnames or [])
    required = {"Days", "Policy"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{input_csv} missing required columns: {sorted(missing)}")

    metric_columns: dict[str, tuple[str, str | None]] = {}
    for metric, _title in METRICS:
        if f"{metric}_mean" in fieldnames:
            mean_col = f"{metric}_mean"
        elif metric in fieldnames:
            mean_col = metric
        else:
            raise ValueError(
                f"{input_csv} missing required mean column for {metric}; "
                f"expected {metric}_mean or {metric}"
            )

        std_col = f"{metric}_std" if f"{metric}_std" in fieldnames else None
        metric_columns[metric] = (mean_col, std_col)

    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed_row: dict[str, object] = {
            "Days": int(row["Days"]),
            "Policy": row["Policy"],
            "N": int(row["N"]) if "N" in fieldnames and row["N"] else default_n_episodes,
        }
        for metric, _title in METRICS:
            mean_col, std_col = metric_columns[metric]
            parsed_row[metric] = float(row[mean_col])
            parsed_row[f"{metric}_std"] = (
                float(row[std_col]) if std_col is not None and row[std_col] else 0.0
            )
        parsed.append(parsed_row)
    return parsed


def plot_metrics(
    rows: list[dict[str, object]],
    output_file: Path,
    dpi: int,
    ci_z: float,
) -> None:
    import matplotlib.pyplot as plt

    output_file.parent.mkdir(parents=True, exist_ok=True)

    days = sorted({int(row["Days"]) for row in rows})
    policies = [policy for policy in POLICY_ORDER if any(row["Policy"] == policy for row in rows)]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": dpi,
        }
    )

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), constrained_layout=True)
    for ax, (metric, title) in zip(axes, METRICS):
        for policy in policies:
            values = []
            ci_half_widths = []
            for day in days:
                match = next(
                    (
                        row
                        for row in rows
                        if row["Policy"] == policy and int(row["Days"]) == day
                    ),
                    None,
                )
                values.append(float(match[metric]) if match is not None else float("nan"))
                ci_half_widths.append(
                    ci_z * float(match[f"{metric}_std"]) / (float(match["N"]) ** 0.5)
                    if match is not None and float(match["N"]) > 0
                    else float("nan")
                )

            ax.errorbar(
                days,
                values,
                yerr=ci_half_widths,
                marker=POLICY_MARKERS.get(policy, "o"),
                linewidth=2.0,
                markersize=5.5,
                color=POLICY_COLORS.get(policy),
                label=policy,
                capsize=2.5,
                elinewidth=1.0,
                alpha=0.95,
            )

        ax.set_title(title)
        ax.set_xlabel("Days")
        ax.set_xticks(days)
        ax.grid(True, alpha=0.28)
        if metric != "Return":
            ax.set_ylim(bottom=0.0, top=1.05 if metric != "DDS" else 0.62)
        else:
            ax.set_ylim(bottom=0.0)

    axes[0].set_ylabel("Metric value")
    axes[0].legend(loc="lower right", frameon=True)
    fig.suptitle(
        "Horizon Scaling: Action-Scoring DQN vs. Baselines (95% CI)",
        fontsize=14,
    )
    fig.savefig(output_file, bbox_inches="tight")
    print(f"Wrote plot: {output_file}")


def main() -> None:
    args = parse_args()
    configure_local_matplotlib_cache(args.output_file)
    rows = load_rows(args.input_csv, args.n_episodes)
    plot_metrics(rows, args.output_file, args.dpi, args.ci_z)


if __name__ == "__main__":
    main()
