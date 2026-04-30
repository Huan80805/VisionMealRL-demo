#!/usr/bin/env bash
set -euo pipefail

# Sweep DQN training/evaluation over planning horizons and seeds.
#
# Override defaults with environment variables:
#   PYTHON_BIN=~/.venv/dl/bin/python
#   OUTPUT_ROOT=runs/agent_sweep
#   HORIZONS="1 3 7 21"
#   SEEDS="42 43 44"
#   TOTAL_TIMESTEPS=200000
#   POLICIES="dqn health multi random"
#   N_EVAL_SEEDS=2
#   NUTRITION5K_DATA=/path/to/artifacts/.../overhead_rgb/train
#   NUTRITION5K_SUBSAMPLE_SIZE=1000
#
# Example:
#   TOTAL_TIMESTEPS=10000 HORIZONS="1 3" SEEDS="42" scripts/agent_sweep.sh

PYTHON_BIN="${PYTHON_BIN:-$HOME/.venv/dl/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/agent_sweep}"
HORIZONS="${HORIZONS:-1 3 7 21}"
SEEDS="${SEEDS:-42}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"
POLICIES="${POLICIES:-dqn health multi random}"
N_EVAL_SEEDS="${N_EVAL_SEEDS:-2}"
MEALS_PER_DAY="${MEALS_PER_DAY:-3}"
EMBEDDING_DIM="${EMBEDDING_DIM:-512}"

mkdir -p "$OUTPUT_ROOT"

for horizon in $HORIZONS; do
  for seed in $SEEDS; do
    run_dir="$OUTPUT_ROOT/h${horizon}_seed${seed}"

    train_args=(
      -m agent.train
      --num_days "$horizon"
      --meals_per_day "$MEALS_PER_DAY"
      --embedding_dim "$EMBEDDING_DIM"
      --total_timesteps "$TOTAL_TIMESTEPS"
      --seed "$seed"
      --output_dir "$run_dir"
    )

    if [[ -n "${NUTRITION5K_DATA:-}" ]]; then
      train_args+=(--nutrition5k-data "$NUTRITION5K_DATA")
    fi
    if [[ -n "${NUTRITION5K_SUBSAMPLE_SIZE:-}" ]]; then
      train_args+=(--nutrition5k-subsample-size "$NUTRITION5K_SUBSAMPLE_SIZE")
    fi

    echo "==> Training horizon=$horizon seed=$seed"
    "$PYTHON_BIN" "${train_args[@]}"

    eval_args=(
      -m agent.evaluate
      --run_dir "$run_dir"
      --policies $POLICIES
      --n_eval_seeds "$N_EVAL_SEEDS"
      --output_csv "$run_dir/eval_metrics.csv"
    )

    echo "==> Evaluating horizon=$horizon seed=$seed"
    "$PYTHON_BIN" "${eval_args[@]}"
  done
done

summary_csv="$OUTPUT_ROOT/sweep_metrics.csv"
first=1
for metrics_csv in "$OUTPUT_ROOT"/h*_seed*/eval_metrics.csv; do
  if [[ ! -f "$metrics_csv" ]]; then
    continue
  fi
  if [[ "$first" -eq 1 ]]; then
    cat "$metrics_csv" > "$summary_csv"
    first=0
  else
    tail -n +2 "$metrics_csv" >> "$summary_csv"
  fi
done

echo "Sweep summary -> $summary_csv"
