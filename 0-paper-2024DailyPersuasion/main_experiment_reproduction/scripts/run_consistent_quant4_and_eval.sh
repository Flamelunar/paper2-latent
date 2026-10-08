#!/usr/bin/env bash
# Finish the current ISR run, then score the three existing generation files.
set -euo pipefail

ROOT="/home1/liujianjian/2-paper-CoLING/0-paper-2024DailyPersuasion/main_experiment_reproduction"
cd "$ROOT"

GENERATOR_PID="${GENERATOR_PID:-848747}"
GENERATION_DIR="$ROOT/results/persuadee_simulation"
ISR_OUTPUT="$GENERATION_DIR/03_isr_sft.jsonl"

# Wait for the active runner to finish and require a complete ISR output.
while true; do
  rows=0
  [ -f "$ISR_OUTPUT" ] && rows=$(wc -l < "$ISR_OUTPUT")
  active=0
  pgrep -f 'python -u tests/05_Test_Persuader_With_Qwen_Persuadee.py' >/dev/null && active=1 || true
  if [ "$rows" -ge 500 ] && [ "$active" -eq 0 ]; then
    break
  fi
  if ! kill -0 "$GENERATOR_PID" 2>/dev/null && [ "$rows" -lt 500 ]; then
    echo "The active generation runner exited with only $rows ISR rows." >&2
    exit 2
  fi
  sleep 60
done

# Use the already completed 01/02 outputs as-is; no zero-shot regeneration.
for method in 01_zero_shot 02_dialogue_sft 03_isr_sft; do
  output="$GENERATION_DIR/${method}.jsonl"
  if [ ! -f "$output" ] || [ "$(wc -l < "$output")" -ne 500 ]; then
    echo "Expected 500 rows in $output before evaluation." >&2
    exit 4
  fi
done

# The evaluator writes the Qwen-judge table and per-method judge JSONL files.
env PYTHONUNBUFFERED=1 python -u tests/06_Evaluate_Main_Experiment.py \
  --judge qwen \
  --qwen-model /data1/liujianjian/model/Qwen3.5-9B \
  --qwen-device cuda:1 \
  --workers 1 \
  --max-new-tokens 1024 \
  --force
