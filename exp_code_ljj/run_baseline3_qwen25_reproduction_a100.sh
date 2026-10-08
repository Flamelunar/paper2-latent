#!/usr/bin/env bash
set -euo pipefail

ROOT="/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj"
MODEL_DIR="/data1/liujianjian/Persuader"
BASE_MODEL="/app/hzj/models/Qwen2.5-7B-Instruct"
ISR_DIR="$MODEL_DIR/qwen2.5-7B-Instruct-ISR-sft"
DPO_DIR="$MODEL_DIR/qwen2.5-7B-Instruct-ISR-SFT-SimDPO"
PERSUADEE_MODEL="/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged"
PREFERENCE_JUDGE="/app/hzj/models/Qwen3.5-9B"
GEMMA_JUDGE="/app/hzj/models/gemma-4-E4B-it"
LOG_DIR="$MODEL_DIR/logs-baseline3-qwen25-paper10k"

cd "$ROOT"
mkdir -p "$LOG_DIR"

run_sft() {
  local output="$MODEL_DIR/$1" command_name="$2" log_file="$3"
  shift 3
  local latest
  if [[ -s "$output/adapter_config.json" && -s "$output/train_metrics.json" ]]; then
    echo "$(basename "$output") is already complete; keeping the saved adapter."
    return
  fi
  local -a command=(python baseline3_isr_sft_simdpo.py --profile paper10k "$command_name"
    --output-dir "$MODEL_DIR" --stage-name "$(basename "$output")" --epochs 1.0 "$@")
  latest=$(find "$output" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1 || true)
  if [[ -n "$latest" ]]; then
    command+=(--resume-from-checkpoint "$output/$latest")
    echo "Resuming $(basename "$output") from $output/$latest"
  else
    command+=(--overwrite-output-dir)
  fi
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 "${command[@]}" 2>&1 | tee -a "$log_file"
}

run_dpo() {
  local log_file="$LOG_DIR/simdpo.log" latest
  if [[ -s "$DPO_DIR/adapter_config.json" && -s "$DPO_DIR/train_metrics.json" ]]; then
    echo "SimDPO is already complete; keeping the saved adapter."
    return
  fi
  local -a command=(python baseline3_isr_sft_simdpo.py --profile paper10k dpo
    --model "$BASE_MODEL" --output-dir "$MODEL_DIR" --stage-name "$(basename "$DPO_DIR")"
    --isr-checkpoint "$ISR_DIR" --epochs 1.0)
  latest=$(find "$DPO_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1 || true)
  if [[ -n "$latest" ]]; then
    command+=(--resume-from-checkpoint "$DPO_DIR/$latest")
    echo "Resuming SimDPO from $DPO_DIR/$latest"
  else
    command+=(--overwrite-output-dir)
  fi
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 "${command[@]}" 2>&1 | tee -a "$log_file"
}

run_sft qwen2.5-7B-Instruct-ISR-sft train-isr "$LOG_DIR/isr_sft.log" \
  --model "$BASE_MODEL" --train-file "$ROOT/data/isr_sft.jsonl"

env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python baseline3_isr_sft_simdpo.py --profile paper10k simulate \
  --persuader-url local://cuda:2 --persuader-model "$ISR_DIR" \
  --user-url local://cuda:2 --user-model "$PERSUADEE_MODEL" \
  --judge-url local://cuda:2 --judge-model "$PREFERENCE_JUDGE" \
  --simulation-rounds 4 --simulation-batch-size 16 2>&1 | tee -a "$LOG_DIR/simulate.log"

run_dpo

env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python run_baseline_generation.py --profile paper10k \
  --output-dir artifacts/generation_5rounds \
  --base-model "$BASE_MODEL" \
  --baseline2-model "$MODEL_DIR/qwen2.5-7b-instruct-dialogue-sft-paper10k" \
  --baseline3-model "$DPO_DIR" --persuadee-model "$PERSUADEE_MODEL" \
  --persuader-device cuda:2 --persuadee-device cuda:2 \
  --context-batch-size 2 --method baseline3_isr_simdpo 2>&1 | tee -a "$LOG_DIR/generation.log"

run_joint_judge() {
  local judge_name="$1" judge_model="$2" output_dir="$3" prompt_version="$4" log_file="$5"
  env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID ALLOW_JUDGE_EVAL=1 \
    python evaluate_joint_prompt_probe.py --full-evaluation \
    --methods baseline3_isr_simdpo \
    --generation-dir artifacts/generation_5rounds \
    --output-dir "$output_dir" \
    --judge-model "$judge_model" --judge-name "$judge_name" --judge-device cuda:2 \
    --prompt-version "$prompt_version" --max-parse-retries 3 \
    2>&1 | tee -a "$log_file"
}

run_joint_judge gemma4 "$GEMMA_JUDGE" \
  artifacts/success_judgments/gemma4 gemma_v11 "$LOG_DIR/judge_gemma4.log"

python update_baseline3_gemma_table.py
echo "Baseline 3 Qwen2.5 ISR-SFT + SimDPO reproduction finished."
