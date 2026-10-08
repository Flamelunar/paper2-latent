#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/app/hzj/liujianjian/ljj-2-paper
EXP="$ROOT/exp_code_ljj"
GEN="$EXP/artifacts/generation_5rounds/meta_llama31/baseline2_dialogue_sft.jsonl"
while [[ ! -f "$GEN" ]] || [[ $(wc -l < "$GEN") -ne 1000 ]]; do sleep 60; done
cd "$EXP"
exec /app/hzj/anaconda3/envs/ljj/bin/python -u baseline4_ma2p.py \
  --test-data data/evaluation_points.jsonl \
  --generation-dir artifacts/generation_5rounds/meta_llama31 \
  --persuader-model /app/hzj/models/Meta-Llama-3.1-8B-Instruct \
  --persuadee-model /app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged \
  --device cuda:0 >> artifacts/logs/new_experiments/meta_llama31_baseline4_ma2p_resume.log 2>&1
