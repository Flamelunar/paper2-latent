#!/usr/bin/env bash
# Archived Llama 3.1 branch; keep its outputs and scores outside the active table.
set -euo pipefail

ROOT="/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj"
MODEL_DIR="/data1/liujianjian/Persuader"
ISR_DIR="$MODEL_DIR/baseline3_isr_sft_adapter"
DPO_DIR="$MODEL_DIR/baseline3_isr_simdpo_adapter"
BASE_MODEL="/app/hzj/models/Meta-Llama-3.1-8B-Instruct"
PERSUADEE_MODEL="/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged"
QWEN_JUDGE="/app/hzj/models/Qwen3.5-9B"
LLAMA_JUDGE="$BASE_MODEL"
LOG_DIR="$MODEL_DIR/logs-baseline3-paper10k"

cd "$ROOT"
mkdir -p "$LOG_DIR"

wait_for_a100() {
  local stable=0 state used utilization compute_pids
  echo "Waiting for A100 cuda:2 to become idle before model work starts."
  while (( stable < 6 )); do
    state=$(nvidia-smi -i 2 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    used=${state%%,*}
    utilization=${state##*,}
    used=${used//[[:space:]]/}
    utilization=${utilization//[[:space:]]/}
    compute_pids=$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
    if [[ "$used" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] \
      && (( used < 4096 && utilization < 5 )) && [[ -z "$compute_pids" ]]; then
      ((stable += 1))
      echo "A100 idle check $stable/6: ${used} MiB, ${utilization}%"
    else
      stable=0
      echo "A100 busy: ${used} MiB, ${utilization}%; waiting 20 seconds."
    fi
    if (( stable < 6 )); then sleep 20; fi
  done
  echo "A100 is idle; starting BF16 reproduction."
}

run_sft() {
  local stage="$1" command_name="$2" log_file="$3"
  shift 3
  local output="$MODEL_DIR/$stage" latest
  if [[ -s "$output/adapter_config.json" && -s "$output/train_metrics.json" ]]; then
    echo "$stage is already complete; keeping the saved adapter."
    return
  fi
  local -a command=(python baseline3_isr_sft_simdpo.py --profile paper10k "$command_name"
    --output-dir "$MODEL_DIR" --stage-name "$stage" "$@")
  latest=$(find "$output" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1 || true)
  if [[ -n "$latest" ]]; then
    command+=(--resume-from-checkpoint "$output/$latest")
    echo "Resuming $stage from $output/$latest"
  else
    command+=(--overwrite-output-dir)
  fi
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 "${command[@]}" 2>&1 | tee -a "$log_file"
}

run_dpo() {
  local log_file="$LOG_DIR/dpo.log" latest
  if [[ -s "$DPO_DIR/adapter_config.json" && -s "$DPO_DIR/train_metrics.json" ]]; then
    echo "SimDPO is already complete; keeping the saved adapter."
    return
  fi
  local -a command=(python baseline3_isr_sft_simdpo.py --profile paper10k dpo
    --model "$BASE_MODEL" --output-dir "$MODEL_DIR" --stage-name "$(basename "$DPO_DIR")"
    --isr-checkpoint "$ISR_DIR")
  latest=$(find "$DPO_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1 || true)
  if [[ -n "$latest" ]]; then
    command+=(--resume-from-checkpoint "$DPO_DIR/$latest")
    echo "Resuming SimDPO from $DPO_DIR/$latest"
  else
    command+=(--overwrite-output-dir)
  fi
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 "${command[@]}" 2>&1 | tee -a "$log_file"
}

if [[ "${BASELINE3_SKIP_INITIAL_IDLE_WAIT:-0}" == "1" ]]; then
  echo "Skipping the initial idle wait; model-loading stages retain their BF16 capacity checks."
else
  wait_for_a100
fi

run_sft baseline3_isr_sft_adapter train-isr "$LOG_DIR/isr_sft.log" --model "$BASE_MODEL"

env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python baseline3_isr_sft_simdpo.py --profile paper10k simulate \
  --persuader-url local://cuda:2 --persuader-model "$ISR_DIR" \
  --user-url local://cuda:2 --user-model "$PERSUADEE_MODEL" \
  --judge-url local://cuda:2 --judge-model "$QWEN_JUDGE" \
  --simulation-rounds 4 --simulation-batch-size 16 2>&1 | tee -a "$LOG_DIR/simulate.log"

# Training performs its own A100/BF16 and free-memory check before loading
# weights. Do not require the entire device to be idle: an unrelated A100 job
# can safely keep running when the training loader's reserved capacity fits.
echo "Starting SimDPO after the preference set is complete; its BF16 loader will wait for sufficient A100 capacity."
run_dpo

env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python run_baseline_generation.py --profile paper10k \
  --output-dir artifacts/generation_legacy_llama31 \
  --base-model /app/hzj/models/Qwen2.5-7B-Instruct \
  --baseline2-model /data1/liujianjian/Persuader/qwen2.5-7b-instruct-dialogue-sft-paper10k \
  --baseline3-model "$DPO_DIR" --persuadee-model "$PERSUADEE_MODEL" \
  --persuader-device cuda:2 --persuadee-device cuda:2 \
  --context-batch-size 2 \
  2>&1 | tee -a "$LOG_DIR/generation.log"

env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python evaluate_baselines.py --profile paper10k \
  --generation-dir artifacts/generation_legacy_llama31 \
  --judge-model "$QWEN_JUDGE" --judge-device cuda:2 \
  --judge-batch-size 8 \
  --evaluation-dir artifacts/success_judgments/qwen35 \
  2>&1 | tee -a "$LOG_DIR/judge_qwen35.log"

env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python evaluate_baselines.py --profile paper10k \
  --generation-dir artifacts/generation_legacy_llama31 \
  --judge-model "$LLAMA_JUDGE" --judge-device cuda:2 \
  --judge-batch-size 8 \
  --evaluation-dir artifacts/success_judgments/llama31 \
  2>&1 | tee -a "$LOG_DIR/judge_llama31.log"
echo "Archived Llama 3.1 Baseline 3 run finished; active main table was not updated."
