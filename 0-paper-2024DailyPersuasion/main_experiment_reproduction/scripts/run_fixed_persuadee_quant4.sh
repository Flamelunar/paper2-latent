#!/usr/bin/env bash
# Run the remaining fixed-persuadee methods sequentially. Evaluation is a
# separate opt-in stage in run_fixed_persuadee_eval_when_ready.sh.
set -u

ROOT="/home1/liujianjian/2-paper-CoLING/0-paper-2024DailyPersuasion/main_experiment_reproduction"
cd "$ROOT"

COMMON=(
  tests/05_Test_Persuader_With_Qwen_Persuadee.py
  --test-data data/small5000/test.jsonl
  --output-dir results/persuadee_simulation
  --persuader-device cuda:2
  --persuadee-device cuda:1
  --persuader-quantization 4bit
  --rounds 4
  --temperature 0.0
  --max-new-tokens 1024
)

if [ -f results/persuadee_simulation/02_dialogue_sft.jsonl ] \
  && [ "$(wc -l < results/persuadee_simulation/02_dialogue_sft.jsonl)" -eq 500 ]; then
  echo "02_dialogue_sft already has 500 rows; skipping regeneration."
else
  # The runner resumes compatible partial JSONL files; do not erase work if a
  # long GPU job is interrupted.
  env PYTHONUNBUFFERED=1 python -u "${COMMON[@]}" --method 02_dialogue_sft
  if [ "$(wc -l < results/persuadee_simulation/02_dialogue_sft.jsonl)" -ne 500 ]; then
    echo "02_dialogue_sft did not produce 500 rows" >&2
    exit 2
  fi
fi

if [ -f results/persuadee_simulation/03_isr_sft.jsonl ] \
  && [ "$(wc -l < results/persuadee_simulation/03_isr_sft.jsonl)" -eq 500 ]; then
  echo "03_isr_sft already has 500 rows; skipping regeneration."
else
  env PYTHONUNBUFFERED=1 python -u "${COMMON[@]}" --method 03_isr_sft
  if [ "$(wc -l < results/persuadee_simulation/03_isr_sft.jsonl)" -ne 500 ]; then
    echo "03_isr_sft did not produce 500 rows" >&2
    exit 3
  fi
fi

# Generation and scoring are separate stages. Keep scoring opt-in so a
# generation-only run does not consume GPU/API quota unexpectedly.
if [ "${RUN_EVAL:-0}" != "1" ]; then
  echo "All three generation outputs are ready; evaluation skipped (set RUN_EVAL=1 to enable)."
  exit 0
fi

env PYTHONUNBUFFERED=1 python -u tests/06_Evaluate_Main_Experiment.py \
  --judge qwen --qwen-device cuda:1 --workers 1 --max-new-tokens 1024
env PYTHONUNBUFFERED=1 python -u tests/06_Evaluate_Main_Experiment.py \
  --judge gpt --gpt-url https://api.chatanywhere.tech/v1 \
  --gpt-model gpt-5.6-luna --gpt-workers 8 --max-new-tokens 1024
