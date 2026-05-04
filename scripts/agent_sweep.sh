#!/usr/bin/env bash
set -euo pipefail

# Sweep DQN training/evaluation over planning horizons and seeds.
#
# Override defaults with environment variables:
#   PYTHON_BIN=~/.venv/dl/bin/python
#   OUTPUT_ROOT=runs/agent_sweep
#   HORIZONS="1 3 7 21"
#   SEEDS="42 43 44"
#   TOTAL_TIMESTEPS=1000000
#   POLICIES="dqn health multi random"
#   W_HEALTH=3.0
#   W_DIVERSITY=0.1
#   W_PREFERENCE=0.1
#   W_BOUNDARY=1.0
#   GAMMA=0.95
#   N_EVAL_SEEDS=2
#   TB_LOG_ROOT=runs/agent_sweep/tensorboard
#   NUTRITION5K_SUBSAMPLE_SIZE=1000
#   CATALOG_DIR=/path/to/catalog
#   NUTRITION5K_DATA=/path/to/nutrition5k_data
#
# Example:
#   TOTAL_TIMESTEPS=10000 HORIZONS="1 3" SEEDS="42" scripts/agent_sweep.sh
CATALOG_DIR="${CATALOG_DIR:-artifacts/catalog/open_clip_ViT-B-32_laion2b_s34b_b79k_overhead_rgb/finetuned_embeddings/train}"
NUTRITION5K_DATA="${NUTRITION5K_DATA:-artifacts/multitask/open_clip_ViT-B-32_laion2b_s34b_b79k_overhead_rgb/finetuned_embeddings/train}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.venv/dl/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/agent_sweep}"
HORIZONS="${HORIZONS:-1 3 7 21}"
SEEDS="${SEEDS:-42}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
POLICIES="${POLICIES:-dqn health multi random}"
N_EVAL_SEEDS="${N_EVAL_SEEDS:-2}"
MEALS_PER_DAY="${MEALS_PER_DAY:-3}"
EMBEDDING_DIM="${EMBEDDING_DIM:-512}"
TB_LOG_ROOT="${TB_LOG_ROOT:-runs/agent_sweep/tensorboard}"
W_HEALTH="${W_HEALTH:-1.0}"
W_DIVERSITY="${W_DIVERSITY:-0.3}"
W_PREFERENCE="${W_PREFERENCE:-0.2}"
W_BOUNDARY="${W_BOUNDARY:-0.5}"
GAMMA="${GAMMA:-0.99}"

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
      --w_health "$W_HEALTH"
      --w_diversity "$W_DIVERSITY"
      --w_preference "$W_PREFERENCE"
      --w_boundary "$W_BOUNDARY"
      --gamma "$GAMMA"
      --nutrition5k-data "$NUTRITION5K_DATA"
      --catalog_dir "$CATALOG_DIR"
    )
    if [[ -n "${NUTRITION5K_SUBSAMPLE_SIZE:-}" ]]; then
      train_args+=(--nutrition5k-subsample-size "$NUTRITION5K_SUBSAMPLE_SIZE")
    fi
    if [[ -n "${TB_LOG_ROOT:-}" ]]; then
      train_args+=(--tb_log_dir "$TB_LOG_ROOT" --tb_log_name "h${horizon}_seed${seed}_dqn")
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
    if [[ -n "${CATALOG_DIR:-}" ]]; then
      eval_args+=(--catalog_dir "$CATALOG_DIR")
    fi

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
