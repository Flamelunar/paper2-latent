#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

: "${CHATANYWHERE_API:?CHATANYWHERE_API is required for simulation and judging.}"

export CUDA_VISIBLE_DEVICES="${PIPELINE_GPU:-1}"
export PYTHONUNBUFFERED=1

export DAILYPERSUASION_PROFILE=small5000
RUN_ROOT="runs/small5000"
mkdir -p "$RUN_ROOT/logs"

# An optional externally launched ISR job can be waited on. Without it, this
# script requires a completed small-profile ISR adapter before proceeding.
if [ -n "${UPSTREAM_PID:-}" ]; then
  tail --pid="$UPSTREAM_PID" -f /dev/null
fi
test -s "$RUN_ROOT/outputs/02_llama31_isr_sft/train_metrics.json"

python 04_Llama31_Ours_ISR_SimDPO_DPO.py simulate \
  --persuader-url local://cuda:0 \
  --persuader-model "$RUN_ROOT/outputs/02_llama31_isr_sft" \
  --user-url https://api.chatanywhere.tech/v1 \
  --user-model gpt-4o-mini \
  --user-api-key-env CHATANYWHERE_API \
  --judge-url https://api.chatanywhere.tech/v1 \
  --judge-model gpt-4o-mini \
  --simulation-rounds 4 \
  --workers 4 2>&1 | tee "$RUN_ROOT/logs/03_simulation.log"

test -s "$RUN_ROOT/data/simulation_preferences_summary.json"

python 04_Llama31_Ours_ISR_SimDPO_DPO.py dpo \
  --isr-checkpoint "$RUN_ROOT/outputs/02_llama31_isr_sft" 2>&1 | tee "$RUN_ROOT/logs/03_dpo.log"

test -s "$RUN_ROOT/outputs/03_llama31_isr_sim_dpo/train_metrics.json"

python evaluate_main_table.py \
  --judge-url https://api.chatanywhere.tech/v1 \
  --judge-model gpt-4o-mini \
  --workers 4 2>&1 | tee "$RUN_ROOT/logs/04_main_evaluation.log"

test -s "$RUN_ROOT/results/main_experiment_summary.json"
test -s "$RUN_ROOT/results/main_experiment_table.md"
