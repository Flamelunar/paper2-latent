#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/app/hzj/liujianjian/ljj-2-paper
EXP="$ROOT/exp_code_ljj"
PY=/app/hzj/anaconda3/envs/ljj/bin/python
GEN="$EXP/artifacts/generation_5rounds"
OUT="$EXP/artifacts/success_judgments"
LOG="$EXP/artifacts/logs/new_experiments"
NORMALIZE="$EXP/normalize_generation_thinking.py"
METHODS=(baseline1_zero_shot baseline2_dialogue_sft baseline3_isr_simdpo baseline4_ma2p baseline5_latentratt)

exec 8>"$EXP/artifacts/.new_judge_supervisor.lock"
if ! flock -n 8; then
  printf 'A new-baselines Judge supervisor is already running.\n' >&2
  exit 73
fi

wait_generation() {
  local family=$1
  for method in "${METHODS[@]}"; do
    while [[ ! -f "$GEN/$family/$method.jsonl" ]] || [[ $(wc -l < "$GEN/$family/$method.jsonl") -ne 1000 ]]; do
      sleep 60
    done
    "$PY" "$NORMALIZE" "$GEN/$family/$method.jsonl"
  done
  "$PY" "$EXP/report_new_baselines_v18.py" --family "$family" --validate-generations-only
}

wait_gpu_free() {
  local gpu=$1 free_mib
  while :; do
    free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    if (( free_mib >= 50000 )); then return; fi
    printf '%s Judge GPU %s: waiting for generation models to exit; free=%s MiB\n' "$(date -Is)" "$gpu" "$free_mib"
    sleep 60
  done
}

run_judge() {
  local family=$1 judge_model=$2 judge_name=$3 prompt_version=$4 device=$5 out_dir=$6
  mkdir -p "$OUT/$out_dir"
  env -u CUDA_VISIBLE_DEVICES ALLOW_JUDGE_EVAL=1 "$PY" -u "$EXP/evaluate_joint_prompt_probe.py" --profile paper10k \
    --full-evaluation --methods "${METHODS[@]}" \
    --generation-dir "$GEN/$family" --output-dir "$OUT/$out_dir" \
    --judge-model "$judge_model" --judge-name "$judge_name" \
    --judge-device "$device" --prompt-version "$prompt_version" \
    --max-tokens 256 --max-parse-retries 3 \
    >> "$LOG/${family}_${judge_name}_v18.log" 2>&1
  "$PY" "$EXP/report_new_baselines_v18.py" --family "$family" --judge-name "$judge_name"
}

run_family() {
  local family=$1 gpu=$2
  wait_generation "$family"
  wait_gpu_free "$gpu"
  printf '%s %s generation validated; starting Qwen3-14B v18 on GPU %s\n' "$(date -Is)" "$family" "$gpu"
  run_judge "$family" /app/hzj/models/Qwen3-14B qwen3_14b v18 "cuda:$gpu" "qwen3_14b_$family"
  wait_gpu_free "$gpu"
  run_judge "$family" /app/hzj/models/Selene-1-Mini-Llama-3.1-8B selene1 v18 "cuda:$gpu" "selene_$family"
}

run_family meta_llama31 0 &
llama_pid=$!
run_family deepseek_r1_qwen14b 1 &
deepseek_pid=$!
status=0
if wait "$llama_pid"; then :; else status=$?; fi
if wait "$deepseek_pid"; then :; else status=$?; fi
exit "$status"
