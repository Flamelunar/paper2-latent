#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home1/liujianjian/anaconda3/envs/ljj/bin/python}"
TRAIN_PID="${1:?Pass the PID of the active Qwen Dialogue-SFT process}"
STAGE_DIR="/data1/liujianjian/Persuader/qwen2.5-7b-instruct-dialogue-sft-paper10k"
LOG_DIR="$ROOT/runs/qwen25_dialogue_sft/logs"
LOG_FILE="$LOG_DIR/table2_after_sft.log"
SMOKE_DIR="$ROOT/runs/qwen25_dialogue_sft/smoke_5rounds"

mkdir -p "$LOG_DIR"
exec >> "$LOG_FILE" 2>&1
TRAIN_COMMAND="$(ps -p "$TRAIN_PID" -o args= || true)"
if [[ "$TRAIN_COMMAND" != *"train_qwen25_dialogue_sft.py"* || "$TRAIN_COMMAND" != *"Qwen2.5-7B-Instruct"* ]]; then
    echo "PID $TRAIN_PID is not the active Qwen2.5 Dialogue-SFT process" >&2
    exit 1
fi

echo "Waiting for Qwen Dialogue-SFT PID $TRAIN_PID to finish"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
done

if [[ ! -s "$STAGE_DIR/adapter_config.json" || ! -s "$STAGE_DIR/train_metrics.json" ]]; then
    echo "Training ended without a completed adapter and train_metrics.json" >&2
    exit 1
fi

# Use physical CUDA indices expected by the BF16 evaluation scripts.
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false

"$PYTHON_BIN" "$ROOT/run_qwen25_table2_generation.py" --profile paper10k \
    --limit 1 --output-dir "$SMOKE_DIR/generation" --context-batch-size 2
"$PYTHON_BIN" "$ROOT/evaluate_qwen25_table2.py" --profile paper10k \
    --limit 1 --generation-dir "$SMOKE_DIR/generation" --evaluation-dir "$SMOKE_DIR/evaluation"
echo "Qwen BF16 five-round smoke validation completed"

"$PYTHON_BIN" "$ROOT/run_qwen25_table2_generation.py" --profile paper10k
"$PYTHON_BIN" "$ROOT/evaluate_qwen25_table2.py" --profile paper10k
echo "Qwen2.5 Table 2-style evaluation completed"
