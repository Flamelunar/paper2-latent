#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/app/hzj/liujianjian/ljj-2-paper
EXP="$ROOT/exp_code_ljj"
PY=/app/hzj/anaconda3/envs/ljj/bin/python
ADAPTER="$ROOT/persuader_atapter_save/meta_llama31/baseline2_dialogue_sft"
LOG="$EXP/artifacts/logs/new_experiments/meta_llama31_baseline1_resume.log"
while [[ ! -f "$ADAPTER/train_metrics.json" ]]; do sleep 60; done
cd "$EXP"
exec "$PY" -u run_baseline_generation.py --profile paper10k \
  --base-model /app/hzj/models/Meta-Llama-3.1-8B-Instruct \
  --baseline2-model /app/hzj/models/Meta-Llama-3.1-8B-Instruct \
  --baseline3-model /app/hzj/models/Meta-Llama-3.1-8B-Instruct \
  --persuadee-model /app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged \
  --persuader-device cuda:0 --persuadee-device cuda:0 \
  --output-dir artifacts/generation_5rounds/meta_llama31 \
  --method baseline1_zero_shot --context-batch-size 2 >> "$LOG" 2>&1
