#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/app/hzj/liujianjian/ljj-2-paper
EXP="$ROOT/exp_code_ljj"
PY=/app/hzj/anaconda3/envs/ljj/bin/python
PYTHONPATH_TRAIN=/tmp/ljj_train_env
PERSUADEE=/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged
ADAPTER_ROOT=/app/hzj/liujianjian/persuader_atapter_save
GEN_ROOT="$EXP/artifacts/generation_5rounds"
LOG="$EXP/artifacts/logs/new_experiments"
mkdir -p "$LOG"

wait_rows() {
  local path=$1
  local expected=$2
  while true; do
    if [[ -f "$path" ]] && [[ $(wc -l < "$path") -eq "$expected" ]]; then
      return 0
    fi
    sleep 60
  done
}

wait_adapter() {
  local path=$1
  while [[ ! -f "$path/adapter_config.json" || ! -f "$path/train_metrics.json" ]]; do
    sleep 60
  done
}

run_b2_generation() {
  local family=$1 base=$2 adapter=$3 out=$4 device=$5
  wait_rows "$GEN_ROOT/$family/baseline1_zero_shot.jsonl" 1000
  wait_adapter "$adapter"
  "$PY" -u "$EXP/run_baseline_generation.py" --profile paper10k \
    --base-model "$base" --baseline2-model "$adapter" --baseline3-model "$base" \
    --persuadee-model "$PERSUADEE" --persuader-device "$device" \
    --persuadee-device "$device" --output-dir "$GEN_ROOT/$family" \
    --method baseline2_dialogue_sft --context-batch-size 2 \
    >> "$LOG/${family}_baseline2_generation.log" 2>&1
}

train_b3_sft() {
  local family=$1 base=$2 device=$3
  local out="$ADAPTER_ROOT/$family/baseline3_isr_sft"
  if [[ ! -f "$out/adapter_config.json" || ! -f "$out/train_metrics.json" ]]; then
    CUDA_VISIBLE_DEVICES="${device##cuda:}" PYTHONPATH="$PYTHONPATH_TRAIN" "$PY" -u \
      "$EXP/baseline3_isr_sft_simdpo.py" --profile paper10k train-isr \
      --model "$base" --output-dir "$ADAPTER_ROOT/$family" \
      --stage-name baseline3_isr_sft --epochs 1 --finetuning-type lora \
      --cutoff-len 2048 --batch-size 1 --gradient-accumulation-steps 16 \
      --learning-rate 1e-4 --save-steps 100 --overwrite-output-dir \
      >> "$LOG/${family}_baseline3_sft.log" 2>&1
  fi
}

run_b3_simulation() {
  local family=$1 base=$2 device=$3
  local out="$ADAPTER_ROOT/$family/baseline3_isr_sft"
  if [[ ! -f "$EXP/data/simulation_preferences.jsonl" ]]; then
    CUDA_VISIBLE_DEVICES="${device##cuda:}" PYTHONPATH="$PYTHONPATH_TRAIN" "$PY" -u \
      "$EXP/baseline3_isr_sft_simdpo.py" --profile paper10k simulate \
      --persuader-url local://cuda:0 --persuader-model "$out" \
      --user-url local://cuda:0 --user-model "$PERSUADEE" \
      --judge-url local://cuda:0 --judge-model /app/hzj/models/Qwen3.5-9B \
      --simulation-rounds 4 --simulation-batch-size 4 --workers 1 --max-retries 6 \
      >> "$LOG/${family}_baseline3_simulation.log" 2>&1
  fi
}

train_b3_dpo() {
  local family=$1 base=$2 device=$3
  local isr="$ADAPTER_ROOT/$family/baseline3_isr_sft"
  local out="$ADAPTER_ROOT/$family/baseline3_simdpo"
  if [[ ! -f "$out/adapter_config.json" || ! -f "$out/train_metrics.json" ]]; then
    CUDA_VISIBLE_DEVICES="${device##cuda:}" PYTHONPATH="$PYTHONPATH_TRAIN" "$PY" -u \
      "$EXP/baseline3_isr_sft_simdpo.py" --profile paper10k dpo \
      --model "$base" --output-dir "$ADAPTER_ROOT/$family" \
      --stage-name baseline3_simdpo --isr-checkpoint "$isr" \
      --epochs 1 --finetuning-type lora --cutoff-len 2048 \
      --batch-size 1 --gradient-accumulation-steps 16 --learning-rate 5e-5 \
      --save-steps 100 --overwrite-output-dir \
      >> "$LOG/${family}_baseline3_dpo.log" 2>&1
  fi
}

run_b3_generation() {
  local family=$1 base=$2 adapter=$3 out=$4 device=$5
  "$PY" -u "$EXP/run_baseline_generation.py" --profile paper10k \
    --base-model "$base" --baseline2-model "$base" --baseline3-model "$adapter" \
    --persuadee-model "$PERSUADEE" --persuader-device "$device" \
    --persuadee-device "$device" --output-dir "$GEN_ROOT/$family" \
    --method baseline3_isr_simdpo --context-batch-size 2 \
    >> "$LOG/${family}_baseline3_generation.log" 2>&1
}

run_b4_generation() {
  local family=$1 base=$2 device=$3
  "$PY" -u "$EXP/baseline4_ma2p.py" --test-data "$EXP/data/evaluation_points.jsonl" \
    --generation-dir "$GEN_ROOT/$family" --persuader-model "$base" \
    --persuadee-model "$PERSUADEE" --device "$device" \
    >> "$LOG/${family}_baseline4_generation.log" 2>&1
}

run_family() {
  local family=$1 base=$2 device=$3
  mkdir -p "$GEN_ROOT/$family" "$ADAPTER_ROOT/$family"
  run_b2_generation "$family" "$base" "$ADAPTER_ROOT/$family/baseline2_dialogue_sft" "$GEN_ROOT/$family" "$device"
  train_b3_sft "$family" "$base" "$device"
  run_b3_simulation "$family" "$base" "$device"
  train_b3_dpo "$family" "$base" "$device"
  run_b3_generation "$family" "$base" "$ADAPTER_ROOT/$family/baseline3_simdpo" "$GEN_ROOT/$family" "$device"
  run_b4_generation "$family" "$base" "$device"
}

run_family meta_llama31 /app/hzj/models/Meta-Llama-3.1-8B-Instruct cuda:0
run_family deepseek_r1_qwen14b /app/hzj/models/DeepSeek-R1-Distill-Qwen-14B cuda:1
