#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/app/hzj/liujianjian/ljj-2-paper
EXP="$ROOT/exp_code_ljj"
GEN="$EXP/artifacts/generation_5rounds/meta_llama31/baseline3_isr_simdpo.jsonl"
LOCK="$EXP/artifacts/.meta_llama31_b3_generation.lock"
LOG="$EXP/artifacts/logs/new_experiments/meta_llama31_baseline3_generation_guard.log"
PY=/app/hzj/anaconda3/envs/ljj/bin/python

mkdir -p "$(dirname "$LOG")"
while :; do
  rows=0
  if [[ -f "$GEN" ]]; then rows=$(wc -l < "$GEN"); fi
  if (( rows == 1000 )); then
    printf '%s guard complete rows=1000\n' "$(date -Is)" >> "$LOG"
    exit 0
  fi
  printf '%s guard sees rows=%s; trying shared generation lock\n' "$(date -Is)" "$rows" >> "$LOG"
  if flock -n "$LOCK" env -u CUDA_VISIBLE_DEVICES PYTHONPATH=/tmp/ljj_train_env \
      "$PY" -u "$EXP/run_baseline_generation.py" --profile paper10k \
      --base-model /app/hzj/models/Meta-Llama-3.1-8B-Instruct \
      --baseline2-model /app/hzj/models/Meta-Llama-3.1-8B-Instruct \
      --baseline3-model /app/hzj/liujianjian/persuader_atapter_save/meta_llama31/baseline3_simdpo \
      --persuadee-model /app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged \
      --persuader-device cuda:0 --persuadee-device cuda:0 \
      --output-dir "$EXP/artifacts/generation_5rounds/meta_llama31" \
      --method baseline3_isr_simdpo --context-batch-size 1 >> "$LOG" 2>&1; then
    rows=0
    if [[ -f "$GEN" ]]; then rows=$(wc -l < "$GEN"); fi
    if (( rows == 1000 )); then
      printf '%s generation completed rows=1000\n' "$(date -Is)" >> "$LOG"
      exit 0
    fi
    printf '%s generation exited without completion rows=%s; retrying\n' "$(date -Is)" "$rows" >> "$LOG"
  else
    printf '%s generation lock is held; waiting\n' "$(date -Is)" >> "$LOG"
  fi
  sleep 60
done
