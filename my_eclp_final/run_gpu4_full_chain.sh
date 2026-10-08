#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/app/hzj/liujianjian/ljj-2-paper
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1
export DAILYPERSUASION_PROFILE=paper10k
export ALLOW_JUDGE_EVAL=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT/my_eclp_final:/tmp:/tmp/ljj_train_env"
exec /app/hzj/anaconda3/envs/ljj/bin/python -u "$ROOT/my_eclp_final/run_gpu4_offline_pipeline.py" "$@"
