#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home1/liujianjian/anaconda3/envs/ljj/bin/python}"
MODEL="${PERSUADER_BASE_MODEL:-/data1/liujianjian/model/Meta-Llama-3.1-8B-Instruct}"
VISIBLE_GPU="${PAPER10K_SFT_GPU:-1}"
RUN_ROOT="$ROOT/runs/paper10k"
RUN_OUTPUTS="$RUN_ROOT/outputs"
LOG_ROOT="$RUN_ROOT/logs"

export CUDA_VISIBLE_DEVICES="$VISIBLE_GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$LOG_ROOT"
source "$ROOT/scripts/paper10k_sft_helpers.sh"

"$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("paper10k SFT stage needs exactly one visible CUDA device")
free, total = torch.cuda.mem_get_info(0)
print(f"visible cuda:0: free={free / 1024**3:.1f} GiB total={total / 1024**3:.1f} GiB")
if free < 20 * 1024**3:
    raise SystemExit("paper10k BF16 SFT needs at least 20 GiB free; leave active GPU jobs running")
PY

paper10k_run_sft "$PYTHON_BIN" "$MODEL" "$RUN_OUTPUTS" \
    "$LOG_ROOT/dialogue_sft.log" "02_Llama31_Dialogue_SFT.py" \
    "01_llama31_dialogue_sft"
paper10k_run_sft "$PYTHON_BIN" "$MODEL" "$RUN_OUTPUTS" \
    "$LOG_ROOT/isr_sft.log" "03_Llama31_ISR_SFT.py" \
    "02_llama31_isr_sft"

echo "paper10k SFT stages finished. Continue the full pipeline with run_table2_paper10k.sh when its two-GPU preflight passes."
