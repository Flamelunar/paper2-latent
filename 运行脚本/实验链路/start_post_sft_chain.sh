#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/app/hzj/liujianjian/ljj-2-paper
EXP="$ROOT/exp_code_ljj"
while [[ ! -f /app/hzj/liujianjian/persuader_atapter_save/meta_llama31/baseline2_dialogue_sft/train_metrics.json ]]; do sleep 60; done
while [[ ! -f "$EXP/artifacts/generation_5rounds/meta_llama31/baseline1_zero_shot.jsonl" ]] || [[ $(wc -l < "$EXP/artifacts/generation_5rounds/meta_llama31/baseline1_zero_shot.jsonl") -ne 1000 ]]; do sleep 60; done
while [[ ! -f "$EXP/artifacts/generation_5rounds/deepseek_r1_qwen14b/baseline1_zero_shot.jsonl" ]] || [[ $(wc -l < "$EXP/artifacts/generation_5rounds/deepseek_r1_qwen14b/baseline1_zero_shot.jsonl") -ne 1000 ]]; do sleep 60; done
exec "$ROOT/运行脚本/实验链路/run_new_experiment_chain.sh"
