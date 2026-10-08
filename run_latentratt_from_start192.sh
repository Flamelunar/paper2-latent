#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/app/hzj/liujianjian/ljj-2-paper
EXP="$ROOT/exp_code_ljj"
LATENT="$ROOT/baseline5_latentRATT"
PY=/app/hzj/anaconda3/envs/ljj/bin/python
TRAIN_ENV=/tmp/ljj_train_env
ADAPTER_ROOT=/app/hzj/liujianjian/persuader_atapter_save
GEN_ROOT="$EXP/artifacts/generation_5rounds"
GEN_192_ROOT="$EXP/artifacts/generation_from_start192"
LOG="$EXP/artifacts/logs/new_experiments"
TRAIN_FILE="$EXP/data/isr_sft.jsonl"

exec 8>"$EXP/artifacts/.new_latentratt_from_start192_supervisor.lock"
if ! flock -n 8; then
  printf 'A new-baselines LatentRATT supervisor is already running.\n' >&2
  exit 73
fi

wait_rows() {
  local path=$1
  while [[ ! -f "$path" ]] || [[ $(wc -l < "$path") -ne 192 ]]; do sleep 60; done
}

wait_gpu_free() {
  local gpu=$1 free_mib
  while :; do
    free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    if (( free_mib >= 50000 )); then return; fi
    printf '%s GPU %s: waiting for 50000 MiB free; currently %s MiB\n' "$(date -Is)" "$gpu" "$free_mib"
    sleep 60
  done
}

wait_existing_sft() {
  local output=$1 found
  while :; do
    if ps -eo args= | rg -F "$LATENT/run_latent_ratt.py sft" | rg -F -- "$output" >/dev/null; then found=1; else found=0; fi
    if (( found == 1 )); then printf '%s waiting for existing SFT process: %s\n' "$(date -Is)" "$output"; fi
    (( found == 1 )) || return 0
    sleep 60
  done
}

run_stage() {
  local stage=$1 out=$2 base=$3 gpu=$4 log=$5 initial=${6:-} max_steps=${7:-}
  local selection_text mode checkpoint code
  local -a selection inspect command
  inspect=("$PY" "$LATENT/stage_checkpoint.py" --output "$out" --stage "$stage"
           --model "$base" --train-file "$TRAIN_FILE" --physical-gpu "$gpu")
  if [[ -n "$max_steps" ]]; then inspect+=(--max-steps "$max_steps"); fi
  while :; do
    if ! selection_text=$("${inspect[@]}"); then
      printf '%s refusing incompatible %s checkpoint: %s\n' "$(date -Is)" "$stage" "$out" >&2
      return 1
    fi
    mapfile -t selection <<< "$selection_text"
    mode=${selection[0]}
    checkpoint=${selection[1]:-}
    if [[ "$mode" == complete ]]; then
      printf '%s verified %s completion: %s\n' "$(date -Is)" "$stage" "$out"
      return
    fi
    wait_gpu_free "$gpu"
    command=("$PY" -u "$LATENT/run_latent_ratt.py" "$stage" --model "$base"
             --device cuda:0 --train-file "$TRAIN_FILE" --output "$out" --epochs 1
             --batch-size 2 --gradient-accumulation 16 --save-steps 100)
    if [[ -n "$max_steps" ]]; then command+=(--max-steps "$max_steps" --save-steps 1); fi
    case "$mode" in
      resume) command+=(--checkpoint "$checkpoint" --resume-training-state "$checkpoint/training_state.pt") ;;
      start)
        if [[ "$stage" == rl ]]; then
          if [[ -z "$initial" ]]; then return 1; fi
          command+=(--checkpoint "$initial")
        fi ;;
      *) printf 'Unknown checkpoint mode: %s\n' "$mode" >&2; return 1 ;;
    esac
    printf '%s GPU %s %s: %s %s\n' "$(date -Is)" "$gpu" "$stage" "$mode" "$out"
    if CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$TRAIN_ENV" "${command[@]}" >> "$log" 2>&1; then
      # Recheck the exact data/model/stage and full step count before proceeding.
      continue
    else
      code=$?
      if [[ "$code" -ne 75 ]]; then return "$code"; fi
      printf '%s %s saved an OOM recovery checkpoint; retaining BF16\n' "$(date -Is)" "$stage"
      sleep 60
    fi
  done
}

run_one() {
  local family=$1 base=$2 physical_gpu=$3
  local out="$ADAPTER_ROOT/$family/baseline5_latentratt"
  local smoke="$ADAPTER_ROOT/$family/baseline5_latentratt_smoke"
  mkdir -p "$ADAPTER_ROOT/$family" "$GEN_192_ROOT/$family"
  if [[ "$family" == deepseek_r1_qwen14b ]]; then
    wait_rows "$GEN_192_ROOT/$family/baseline4_ma2p.jsonl"
  fi
  wait_existing_sft "$out/sft"

  # Validate both forward/backward phases on this architecture. A compatibility
  # failure exits this family's worker; it never substitutes another model.
  run_stage sft "$smoke/sft" "$base" "$physical_gpu" "$LOG/${family}_baseline5_smoke.log" "" 1
  run_stage rl "$smoke/rl" "$base" "$physical_gpu" "$LOG/${family}_baseline5_smoke.log" "$smoke/sft" 1

  # Preserve the completed SFT checkpoint separately. The canonical stage root
  # contains only the final LR-GRPO adapter/attention, not an unfinished SFT run.
  run_stage sft "$out/sft" "$base" "$physical_gpu" "$LOG/${family}_baseline5_sft.log"
  run_stage rl "$out" "$base" "$physical_gpu" "$LOG/${family}_baseline5_rl.log" "$out/sft"

  wait_rows "$GEN_192_ROOT/$family/baseline4_ma2p.jsonl"
  local generation="$GEN_192_ROOT/$family/baseline5_latentratt.jsonl"
  if [[ ! -f "$generation" ]] || [[ $(wc -l < "$generation") -ne 192 ]]; then
    CUDA_VISIBLE_DEVICES="$physical_gpu" PYTHONPATH="$EXP:$TRAIN_ENV" "$PY" -u "$ROOT/generate_from_start192.py" --checkpoint "$out" \
      --base-model "$base" --persuader-device cuda:0 --persuadee-device cuda:0 \
      --persuadee-model /app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged \
      --output "$generation" --method-slug baseline5_latentratt \
      --display-name "Baseline 5: LatentRATT" --limit 1000 \
      --min-free-memory-mib 50000 \
      >> "$LOG/${family}_baseline5_generation.log" 2>&1
  fi
}

run_one meta_llama31 /app/hzj/models/Meta-Llama-3.1-8B-Instruct 0 &
llama_pid=$!
run_one deepseek_r1_qwen14b /app/hzj/models/DeepSeek-R1-Distill-Qwen-14B 1 &
deepseek_pid=$!
status=0
if wait "$llama_pid"; then :; else status=$?; fi
if wait "$deepseek_pid"; then :; else status=$?; fi
exit "$status"
