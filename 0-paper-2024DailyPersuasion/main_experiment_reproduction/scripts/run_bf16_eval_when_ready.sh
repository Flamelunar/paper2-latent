#!/usr/bin/env bash
# Wait for the BF16 generation runner, then score only with raw Qwen3.5-9B.
set -euo pipefail

ROOT="/home1/liujianjian/2-paper-CoLING/0-paper-2024DailyPersuasion/main_experiment_reproduction"
cd "$ROOT"

GENERATOR_PID="${GENERATOR_PID:?GENERATOR_PID must be set to the BF16 runner PID}"
GENERATION_DIR="$ROOT/results/persuadee_simulation"

while true; do
  rows=0
  if [ -f "$GENERATION_DIR/03_isr_sft.jsonl" ]; then
    rows="$(wc -l < "$GENERATION_DIR/03_isr_sft.jsonl")"
  fi
  active=0
  pgrep -f 'python -u tests/05_Test_Persuader_With_Qwen_Persuadee.py' >/dev/null \
    && active=1 || true
  if [ "$rows" -ge 500 ] && [ "$active" -eq 0 ]; then
    break
  fi
  if ! kill -0 "$GENERATOR_PID" 2>/dev/null && [ "$rows" -lt 500 ]; then
    echo "BF16 generation runner exited with only $rows ISR rows." >&2
    exit 2
  fi
  sleep 60
done

for method in 01_zero_shot 02_dialogue_sft 03_isr_sft; do
  output="$GENERATION_DIR/${method}.jsonl"
  if [ ! -f "$output" ] || [ "$(wc -l < "$output")" -ne 500 ]; then
    echo "Expected 500 rows in $output before evaluation." >&2
    exit 4
  fi
done

env PYTHONUNBUFFERED=1 python -u tests/06_Evaluate_Main_Experiment.py \
  --judge qwen \
  --generation-dir "$GENERATION_DIR" \
  --evaluation-dir "$ROOT/results/persuadee_evaluation_bf16" \
  --qwen-model /data1/liujianjian/model/Qwen3.5-9B \
  --qwen-device cuda:1 \
  --workers 1 \
  --max-new-tokens 1024 \
  --force
