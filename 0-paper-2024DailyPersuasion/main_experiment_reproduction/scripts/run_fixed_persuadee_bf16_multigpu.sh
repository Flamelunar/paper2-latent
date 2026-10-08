#!/usr/bin/env bash
# Run all three fixed-persuadee methods sequentially with BF16 inference.
set -euo pipefail

ROOT="/home1/liujianjian/2-paper-CoLING/0-paper-2024DailyPersuasion/main_experiment_reproduction"
cd "$ROOT"

COMMON=(
  tests/05_Test_Persuader_With_Qwen_Persuadee.py
  --test-data data/small5000/test.jsonl
  --output-dir results/persuadee_simulation
  --persuader-device cuda:0,cuda:2
  --persuadee-device cuda:1
  --persuader-quantization none
  --rounds 4
  --temperature 0.0
  --max-new-tokens 1024
)

run_method() {
  local method="$1"
  local output="results/persuadee_simulation/${method}.jsonl"

  if [ -f "$output" ] && [ "$(wc -l < "$output")" -eq 500 ]; then
    echo "$method already has 500 rows; skipping regeneration."
    return 0
  fi

  # The shared runner resumes compatible partial output instead of erasing it.
  env PYTHONUNBUFFERED=1 python -u "${COMMON[@]}" --method "$method"
  local rows
  rows="$(wc -l < "$output")"
  if [ "$rows" -ne 500 ]; then
    echo "$method did not produce 500 rows (found $rows)" >&2
    return 2
  fi
}

run_method 01_zero_shot
run_method 02_dialogue_sft
run_method 03_isr_sft
echo "All three BF16 multi-GPU generation outputs are ready."
