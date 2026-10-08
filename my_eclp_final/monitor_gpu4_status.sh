#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/app/hzj/liujianjian/ljj-2-paper
WORK="$ROOT/eclp_runs_gpu4"
if [[ -f "$WORK/pipeline_status.json" ]]; then
  cat "$WORK/pipeline_status.json"
else
  echo 'pipeline_status.json has not been created yet'
fi
if [[ -f "$WORK/sft_full/train.jsonl" ]]; then
  tail -n 1 "$WORK/sft_full/train.jsonl"
fi
nvidia-smi --id=4 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
