#!/usr/bin/env bash
# Full-corpus SFT + LR-GRPO + fixed 1,000-case generation.
set -euo pipefail

ROOT="/app/hzj/liujianjian/ljj-2-paper/baseline5_latentRATT"
PYTHON_BIN="/app/hzj/anaconda3/envs/ljj/bin/python"
MODEL_ROOT="/app/hzj/liujianjian/qwen2.5-7b-latentRATT-all-59868"
SFT_DIR="$MODEL_ROOT/sft"
RL_DIR="$MODEL_ROOT/rl"
DATA_FILE="/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj/data/isr_sft.jsonl"
LOG_DIR="$ROOT/logs/full_isr_59868"
EXP_ROOT="/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj"
GENERATION_DIR="$EXP_ROOT/artifacts/generation_5rounds"
CHECKPOINT_MEMORY_FREE_MIB=50000
mkdir -p "$LOG_DIR" "$MODEL_ROOT"
cd "$ROOT"
exec > >(tee -a "$LOG_DIR/runner.log") 2>&1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES

"$PYTHON_BIN" -u prepare_full_isr.py 2>&1 | tee -a "$LOG_DIR/data.log"

is_completed() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "run_config.json"
if not p.is_file():
    raise SystemExit(1)
c = json.loads(p.read_text())
raise SystemExit(0 if c.get("train_rows") == 59868 and c.get("completed_steps") else 1)
PY
}

latest_resume_checkpoint() {
  local output="$1"
  [[ -d "$output" ]] || return 0
  find "$output" -mindepth 2 -maxdepth 2 -type f -name training_state.pt -printf '%h\n' 2>/dev/null \
    | sort -V | tail -n 1
}

run_stage() {
  local stage="$1" output="$2" log="$3" base_checkpoint="${4:-}"
  shift 4 || true
  local -a stage_args=("$@")
  local resume_checkpoint=""
  if is_completed "$output"; then
    echo "$(date -Is) $stage already completed at $output; skipping." | tee -a "$log"
    return 0
  fi
  mkdir -p "$output"
  resume_checkpoint="$(latest_resume_checkpoint "$output")"
  while true; do
    local checkpoint="$base_checkpoint"
    [[ -z "$resume_checkpoint" ]] || checkpoint="$resume_checkpoint"
    local -a command=("$PYTHON_BIN" -u run_latent_ratt.py "$stage" --device cuda:2
      --model /app/hzj/models/Qwen2.5-7B-Instruct
      --train-file "$DATA_FILE" --output "$output"
      --epochs 1 --max-prompt-tokens 1536 --max-target-tokens 256
      --save-steps 25 --log-steps 5 "${stage_args[@]}")
    [[ -z "$checkpoint" ]] || command+=(--checkpoint "$checkpoint")
    if [[ -n "$resume_checkpoint" ]]; then
      command+=(--resume-training-state "$resume_checkpoint/training_state.pt")
    fi

    set +e
    "${command[@]}" 2>&1 | tee -a "$log"
    local status=${PIPESTATUS[0]}
    set -e
    if (( status == 0 )); then
      return 0
    elif (( status != 75 )); then
      echo "$(date -Is) $stage exited with status $status; stopping pipeline." | tee -a "$log"
      return "$status"
    fi

    resume_checkpoint="$(sed -n 's/^CUDA_OOM_RECOVERY_CHECKPOINT=//p' "$log" | tail -n 1)"
    if [[ -z "$resume_checkpoint" || ! -f "$resume_checkpoint/training_state.pt" ]]; then
      echo "Missing optimizer/data recovery state after CUDA OOM; stopping." | tee -a "$log"
      return 1
    fi
    echo "$(date -Is) CUDA OOM saved model, optimizer and data cursor to $resume_checkpoint." | tee -a "$log"
    while true; do
      local free_mib
      free_mib="$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
      if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= CHECKPOINT_MEMORY_FREE_MIB )); then
        break
      fi
      echo "$(date -Is) Waiting for A100 memory to recover (free=${free_mib} MiB; need ${CHECKPOINT_MEMORY_FREE_MIB} MiB)." | tee -a "$log"
      sleep 60
    done
  done
}

run_stage sft "$SFT_DIR" "$LOG_DIR/sft.log" "" \
  --gradient-accumulation 16 --rationale-weight 0.2 --validate
run_stage rl "$RL_DIR" "$LOG_DIR/rl.log" "$SFT_DIR" \
  --batch-size 8 --group-size 4 --save-steps 100 --log-steps 10

"$PYTHON_BIN" -u audit_full_run.py 2>&1 | tee -a "$LOG_DIR/audit.log"

"$PYTHON_BIN" -u generate_1000.py --checkpoint "$RL_DIR" \
  --base-model /app/hzj/models/Qwen2.5-7B-Instruct \
  --persuader-device cuda:2 --persuadee-device cuda:2 \
  --persuadee-model /app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged \
  --max-new-tokens 512 \
  --output "$GENERATION_DIR/baseline5_latentRATT.json" \
  2>&1 | tee -a "$LOG_DIR/generation.log"

echo "$(date -Is) Completed all-session SFT, LR-GRPO, and 1,000 five-round generations; evaluation deferred until its prompts are ready." | tee -a "$LOG_DIR/pipeline.log"
