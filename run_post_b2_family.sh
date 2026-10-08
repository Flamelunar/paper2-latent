#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/app/hzj/liujianjian/ljj-2-paper
EXP="$ROOT/exp_code_ljj"
PY=/app/hzj/anaconda3/envs/ljj/bin/python
TRAIN_ENV=/tmp/ljj_train_env
PERSUADEE=/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged
ADAPTER_ROOT=/app/hzj/liujianjian/persuader_atapter_save
GEN_ROOT="$EXP/artifacts/generation_5rounds"
LOG="$EXP/artifacts/logs/new_experiments"

family=${1:?family required}
base=${2:?base model required}
gpu=${3:?gpu index required}

adapter="$ADAPTER_ROOT/$family"
generation="$GEN_ROOT/$family"
preferences="$EXP/artifacts/preference_simulations/$family"
mkdir -p "$preferences"
exec 7>"$EXP/artifacts/.chain_${family}.lock"
if ! flock -n 7; then
  printf 'A %s baseline chain is already running.\n' "$family" >&2
  exit 73
fi

wait_rows() {
  local path=$1
  while [[ ! -f "$path" ]] || [[ $(wc -l < "$path") -ne 1000 ]]; do
    sleep 60
  done
}

wait_rows "$generation/baseline2_dialogue_sft.jsonl"

# DeepSeek-14B plus the Qwen3 judge can exhaust fragmented BF16 memory on
# long preference histories with a batch of eight. Keep Llama throughput,
# but make checkpoint-safe DeepSeek restarts use one history per pass.
simulation_batch_size=8
if [[ "$family" == deepseek_r1_qwen14b ]]; then
  simulation_batch_size=1
fi

if [[ ! -f "$adapter/baseline3_isr_sft/adapter_config.json" || ! -f "$adapter/baseline3_isr_sft/train_metrics.json" ]]; then
  checkpoint=$("$PY" "$EXP/hf_resume_checkpoint.py" "$adapter/baseline3_isr_sft" "$base")
  resume_args=()
  if [[ -n "$checkpoint" ]]; then resume_args=(--resume-from-checkpoint "$checkpoint"); fi
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$TRAIN_ENV" "$PY" -u \
    "$EXP/baseline3_isr_sft_simdpo.py" --profile paper10k train-isr \
    --model "$base" --output-dir "$adapter" --stage-name baseline3_isr_sft \
    --epochs 1 --finetuning-type lora --cutoff-len 2048 --batch-size 1 \
    --gradient-accumulation-steps 16 --learning-rate 1e-4 --save-steps 100 \
    --overwrite-output-dir "${resume_args[@]}" >> "$LOG/${family}_baseline3_sft_chain.log" 2>&1
fi

if [[ ! -f "$adapter/baseline3_simdpo/adapter_config.json" || ! -f "$adapter/baseline3_simdpo/train_metrics.json" ]]; then
  # Each family owns immutable preference artifacts; no shared data overwrite
  # or cross-GPU serialization. A restart reuses completed public-only paths.
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$TRAIN_ENV" "$PY" -u \
      "$EXP/baseline3_isr_sft_simdpo.py" --profile paper10k simulate \
      --persuader-url local://cuda:0 \
      --persuader-model "$adapter/baseline3_isr_sft" \
      --user-url local://cuda:0 --user-model "$PERSUADEE" \
      --judge-url local://cuda:0 --judge-model /app/hzj/models/Qwen3-14B \
      --simulation-rounds 4 --workers 1 --simulation-batch-size "$simulation_batch_size" \
      --preference-directory "$preferences" \
      >> "$LOG/${family}_baseline3_simulation.log" 2>&1
    checkpoint=$("$PY" "$EXP/hf_resume_checkpoint.py" "$adapter/baseline3_simdpo" "$base")
    resume_args=()
    if [[ -n "$checkpoint" ]]; then resume_args=(--resume-from-checkpoint "$checkpoint"); fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$TRAIN_ENV" "$PY" -u \
      "$EXP/baseline3_isr_sft_simdpo.py" --profile paper10k dpo \
      --model "$base" --output-dir "$adapter" --stage-name baseline3_simdpo \
      --isr-checkpoint "$adapter/baseline3_isr_sft" --epochs 1 \
      --train-file "$preferences/simulation_preferences.jsonl" \
      --finetuning-type lora --cutoff-len 2048 --batch-size 1 \
      --gradient-accumulation-steps 16 --learning-rate 5e-5 --save-steps 100 \
      --overwrite-output-dir "${resume_args[@]}" >> "$LOG/${family}_baseline3_dpo_chain.log" 2>&1
fi

cd "$EXP"
# Generation uses physical CUDA ordinals so the recorded device remains the
# requested GPU. Training/simulation above intentionally mask their GPU and
# therefore use local cuda:0; do not combine that mask with cuda:$gpu here.
env -u CUDA_VISIBLE_DEVICES PYTHONPATH="$TRAIN_ENV" "$PY" -u run_baseline_generation.py --profile paper10k \
  --base-model "$base" --baseline2-model "$base" \
  --baseline3-model "$adapter/baseline3_simdpo" --persuadee-model "$PERSUADEE" \
  --persuader-device "cuda:$gpu" --persuadee-device "cuda:$gpu" \
  --output-dir "$generation" --method baseline3_isr_simdpo --context-batch-size 1 \
  >> "$LOG/${family}_baseline3_generation_chain.log" 2>&1

"$PY" -u baseline4_ma2p.py --test-data "$EXP/data/evaluation_points.jsonl" \
  --generation-dir "$generation" --persuader-model "$base" \
  --persuadee-model "$PERSUADEE" --device "cuda:$gpu" \
  >> "$LOG/${family}_baseline4_generation_chain.log" 2>&1
