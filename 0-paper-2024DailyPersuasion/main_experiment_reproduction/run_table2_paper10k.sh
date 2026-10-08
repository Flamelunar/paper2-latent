#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home1/liujianjian/anaconda3/envs/ljj/bin/python}"
MODEL="${PERSUADER_BASE_MODEL:-/data1/liujianjian/model/Meta-Llama-3.1-8B-Instruct}"
PERSUADEE="${PERSUADEE_MODEL:-/data1/liujianjian/Persuadee/Qwen3.5-9B-persuadee/merged}"
JUDGE="${JUDGE_MODEL:-/data1/liujianjian/model/Qwen3.5-9B}"
VISIBLE_GPUS="${RUN_GPUS:-2,1}"
PERSUADER_DEVICE="${PERSUADER_DEVICE:-cuda:0}"
PERSUADEE_DEVICE="${PERSUADEE_DEVICE:-cuda:1}"
JUDGE_DEVICE="${JUDGE_DEVICE:-cuda:0}"
RUN_ROOT="$ROOT/runs/paper10k"
RUN_DATA="$RUN_ROOT/data"
RUN_OUTPUTS="$RUN_ROOT/outputs"
RUN_RESULTS="$RUN_ROOT/results"
LOG_ROOT="$RUN_ROOT/logs"

export CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$LOG_ROOT"
source "$ROOT/scripts/paper10k_sft_helpers.sh"

"$PYTHON_BIN" - <<'PY'
import sys
import torch

minimum_free = (64 * 1024**3, 20 * 1024**3)
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit("paper10k pipeline needs two visible CUDA devices")
for index, required in enumerate(minimum_free):
    free, total = torch.cuda.mem_get_info(index)
    print(f"cuda:{index}: free={free / 1024**3:.1f} GiB total={total / 1024**3:.1f} GiB")
    if free < required:
        raise SystemExit(
            f"cuda:{index} has less than {required / 1024**3:.0f} GiB free; "
            "leave existing GPU jobs running and retry when capacity is available"
        )
PY

cd "$ROOT"
"$PYTHON_BIN" prepare_data.py --profile paper10k \
    2>&1 | tee "$LOG_ROOT/prepare_data.log"

paper10k_run_sft "$PYTHON_BIN" "$MODEL" "$RUN_OUTPUTS" \
    "$LOG_ROOT/dialogue_sft.log" "02_Llama31_Dialogue_SFT.py" \
    "01_llama31_dialogue_sft"

paper10k_run_sft "$PYTHON_BIN" "$MODEL" "$RUN_OUTPUTS" \
    "$LOG_ROOT/isr_sft.log" "03_Llama31_ISR_SFT.py" \
    "02_llama31_isr_sft"

"$PYTHON_BIN" 04_Llama31_Ours_ISR_SimDPO_DPO.py --profile paper10k simulate \
    --persuader-url "local://$PERSUADER_DEVICE" \
    --persuader-model "$RUN_OUTPUTS/02_llama31_isr_sft" \
    --user-url "local://$PERSUADEE_DEVICE" --user-model "$PERSUADEE" \
    --judge-url "local://$JUDGE_DEVICE" --judge-model "$JUDGE" \
    --simulation-rounds 4 --workers 1 \
    2>&1 | tee "$LOG_ROOT/simulation.log"

"$PYTHON_BIN" 04_Llama31_Ours_ISR_SimDPO_DPO.py --profile paper10k dpo \
    --model "$MODEL" --epochs 1 --finetuning-type lora \
    --cutoff-len 2048 --batch-size 1 --gradient-accumulation-steps 16 \
    --isr-checkpoint "$RUN_OUTPUTS/02_llama31_isr_sft" \
    2>&1 | tee "$LOG_ROOT/dpo.log"

"$PYTHON_BIN" tests/05_Test_Persuader_With_Qwen_Persuadee.py --profile paper10k \
    --persuader-device "$PERSUADER_DEVICE" --persuadee-device "$PERSUADEE_DEVICE" \
    --rounds 4 \
    2>&1 | tee "$LOG_ROOT/test_persuaders.log"

"$PYTHON_BIN" tests/06_Evaluate_Main_Experiment.py --profile paper10k \
    --judge qwen --qwen-model "$JUDGE" --qwen-device "$JUDGE_DEVICE" --workers 1 \
    2>&1 | tee "$LOG_ROOT/evaluate_table2.log"

echo "Table 2 adaptation finished. Results: $RUN_RESULTS/main_experiment_table.md"
