#!/usr/bin/env bash
# Wait for the fixed-persuadee generation to finish, then score all methods.
set -euo pipefail

ROOT="/home1/liujianjian/2-paper-CoLING/0-paper-2024DailyPersuasion/main_experiment_reproduction"
GENERATION="$ROOT/results/persuadee_simulation/03_isr_sft.jsonl"
cd "$ROOT"

while true; do
  rows=0
  if [ -f "$GENERATION" ]; then
    rows=$(wc -l < "$GENERATION")
  fi
  if [ "$rows" -ge 500 ]; then
    active=$(ps -eo args | awk '/python -u tests\/05_Test_Persuader_With_Qwen_Persuadee\.py/ {found=1} END {print found+0}')
    if [ "$active" -eq 0 ]; then
      break
    fi
  fi
  sleep 60
done

# If an older runner instance already entered evaluation, let it own the
# scoring stage instead of starting a second judge process.
if ps -eo args | awk '/python -u tests\/06_Evaluate_Main_Experiment\.py/ {found=1} END {exit !found}'; then
  while ps -eo args | awk '/python -u tests\/06_Evaluate_Main_Experiment\.py/ {found=1} END {exit !found}'; do
    sleep 60
  done
  exit 0
fi

env PYTHONUNBUFFERED=1 python -u tests/06_Evaluate_Main_Experiment.py \
  --judge both \
  --qwen-model /data1/liujianjian/model/Qwen3.5-9B \
  --qwen-device cuda:1 \
  --workers 1 \
  --gpt-url https://api.chatanywhere.tech/v1 \
  --gpt-model gpt-5.6-luna \
  --gpt-workers 8 \
  --max-new-tokens 1024
