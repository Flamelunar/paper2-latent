#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/app/hzj/anaconda3/envs/ljj/bin/python}"
MODEL="${PERSUADER_BASE_MODEL:-/app/hzj/models/Qwen2.5-7B-Instruct}"
GPU="${PAPER10K_SFT_GPU:-1}"
OUTPUT_ROOT="${BASELINE2_OUTPUT_ROOT:-$ROOT/models}"
STAGE_NAME="${BASELINE2_STAGE_NAME:-dialogue_sft_adapter}"
TRAIN_FILE="$ROOT/data/dialogue_sft.jsonl"
TEST_FILE="$ROOT/data/evaluation_points.jsonl"
STAGE_DIR="$OUTPUT_ROOT/$STAGE_NAME"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/dialogue_sft.log"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$LOG_DIR"

"$PYTHON_BIN" - "$ROOT" "$TRAIN_FILE" "$TEST_FILE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root, train_file, test_file = map(Path, sys.argv[1:])
manifest = json.loads((root / "data/manifest.json").read_text())

def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

train = read_jsonl(train_file)
test = read_jsonl(test_file)
train_meta = manifest["files"]["dialogue_sft"]
test_meta = manifest["files"]["evaluation_points"]
for path, rows, expected in ((train_file, train, train_meta), (test_file, test, test_meta)):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(rows) != expected["rows"] or digest != expected["sha256"]:
        raise SystemExit(f"Dataset does not match manifest: {path}")
train_scenarios = {row["metadata"]["scenario_id"] for row in train}
test_scenarios = {row["scenario_id"] for row in test}
overlap = train_scenarios & test_scenarios
if len(train) != 59868 or len(test) != 1000 or overlap:
    raise SystemExit(f"Unexpected paper10k split: train={len(train)}, test={len(test)}, overlap={len(overlap)}")
for index, row in enumerate(train):
    if not isinstance(row.get("history"), list) or not row.get("instruction") or not row.get("output", "").strip():
        raise SystemExit(f"Malformed Dialogue-SFT example at line {index + 1}")
    if any(not isinstance(pair, list) or len(pair) != 2 for pair in row["history"]):
        raise SystemExit(f"Malformed history at line {index + 1}")
print(f"paper10k validation PASS: {len(train)} SFT rows, {len(test)} held-out contexts, no scenario leakage")
PY

if [[ -f "$STAGE_DIR/adapter_config.json" && -f "$STAGE_DIR/train_metrics.json" ]]; then
    "$PYTHON_BIN" - "$MODEL" "$STAGE_DIR/adapter_config.json" <<'PY'
import json
import sys
from pathlib import Path

selected, config_path = sys.argv[1:]
trained = json.loads(Path(config_path).read_text(encoding="utf-8"))["base_model_name_or_path"]
if Path(selected).expanduser().resolve() != Path(trained).expanduser().resolve():
    raise SystemExit("Existing Dialogue-SFT adapter belongs to another base model; set BASELINE2_STAGE_NAME")
PY
    echo "Baseline 2 Dialogue-SFT is already complete: $STAGE_DIR"
    exit 0
fi

"$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("BF16 SFT requires exactly one visible CUDA device")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("The selected CUDA device does not support BF16")
free, total = torch.cuda.mem_get_info(0)
print(f"visible cuda:0: {torch.cuda.get_device_name(0)}, BF16 supported, free={free / 1024**3:.1f} GiB / {total / 1024**3:.1f} GiB")
if free < 20 * 1024**3:
    raise SystemExit("BF16 SFT needs at least 20 GiB free with this launcher")
PY

RESUME_ARGS=()
if [[ -d "$STAGE_DIR" ]]; then
    CHECKPOINT=""
    while IFS= read -r candidate; do
        if [[ -f "$candidate/trainer_state.json" && -f "$candidate/optimizer.pt" && -f "$candidate/scheduler.pt" ]]; then
            CHECKPOINT="$candidate"
            break
        fi
    done < <(find "$STAGE_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -Vr)
    if [[ -n "$CHECKPOINT" ]]; then
        RESUME_ARGS=(--resume-from-checkpoint "$CHECKPOINT")
        echo "Resuming Baseline 2 Dialogue-SFT from $CHECKPOINT"
    elif [[ -n "$(find "$STAGE_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Output exists without a complete Trainer checkpoint: $STAGE_DIR" >&2
        exit 1
    fi
fi

"$PYTHON_BIN" "$ROOT/baseline2_dialogue_sft.py" --profile paper10k \
    --model "$MODEL" --output-dir "$OUTPUT_ROOT" --stage-name "$STAGE_NAME" \
    --train-file "$TRAIN_FILE" --epochs 1 --finetuning-type lora \
    --cutoff-len 2048 --batch-size 1 --gradient-accumulation-steps 16 \
    --learning-rate 1e-4 --save-steps 100 "${RESUME_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
