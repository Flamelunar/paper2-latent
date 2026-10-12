#!/usr/bin/env bash
set -euo pipefail

ROOT="/home1/liujianjian/2-paper-CoLING/exp_code_ljj"
OUT="$ROOT/artifacts/success_judgments/gemma4"
PYTHON="/home1/liujianjian/anaconda3/envs/ljj/bin/python"
MARKER="GEMMA4_FULL_EVAL_V11"

cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export ALLOW_JUDGE_EVAL=1

if [[ -f "$OUT/full_eval_complete" ]]; then
  exit 0
fi

"$PYTHON" -u evaluate_joint_prompt_probe.py \
  --full-evaluation \
  --methods qwen25_dialogue_sft \
  --output-dir artifacts/success_judgments/gemma4 \
  --judge-model /data1/liujianjian/model/gemma-4-E4B-it \
  --judge-name gemma4 \
  --judge-device cuda:1 \
  --prompt-version v11 \
  --max-tokens 384 \
  --max-parse-retries 3 \
  --max-evaluations 32

"$PYTHON" - <<'PY'
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

root = Path("/home1/liujianjian/2-paper-CoLING/exp_code_ljj")
output_dir = root / "artifacts/joint_success_win_full/gemma4_v11"
result_path = output_dir / "qwen25_dialogue_sft.jsonl"
source_path = root / "artifacts/generation_5rounds/qwen25_dialogue_sft.jsonl"
expected_judge = "/data1/liujianjian/model/gemma-4-E4B-it"
expected_revision = "joint_v2_r11"

source = {
    int(row["index"]): row["scenario_id"]
    for row in (json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip())
}
gpu1_rows = [
    json.loads(line)
    for line in result_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
gpu1_rows = [row for row in gpu1_rows if row.get("judge_device") == "cuda:1"]
by_index = {}
for row in gpu1_rows:
    index = int(row["index"])
    if (row.get("judge_model") != expected_judge
            or row.get("method") != "qwen25_dialogue_sft"
            or row.get("prompt_version") != "v11"
            or row.get("prompt_revision") != expected_revision
            or row.get("judge_compute_dtype") != "bfloat16"
            or row.get("judge_quantization") != "none"
            or row.get("max_parse_retries") != 3
            or source.get(index) != row.get("scenario_id")):
        raise ValueError(f"Unexpected or misaligned GPU 1 record at index {index}")
    if index in by_index:
        raise ValueError(f"Duplicate GPU 1 Dialogue-SFT result at index {index}")
    by_index[index] = row

print(f"GPU 1 Dialogue-SFT checkpoint: {len(by_index)}/1000", flush=True)
if set(by_index) != set(range(1000)):
    raise SystemExit(0)

metadata_path = output_dir / "run_metadata.json"
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
metadata["status"] = "completed"
metadata["status_updated_at_utc"] = datetime.now(timezone.utc).isoformat()
metadata["completed_gpu1_rows"] = {
    "qwen25_zero_shot": 1000,
    "qwen25_dialogue_sft": 1000,
}
metadata["completion_source"] = "independent cron checkpoint batches"
fd, temp_name = tempfile.mkstemp(prefix=".run_metadata.", suffix=".tmp", dir=output_dir)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_name, metadata_path)
finally:
    if os.path.exists(temp_name):
        os.unlink(temp_name)

(output_dir / "full_eval_complete").write_text(
    f"Verified all 1,000 GPU 1 Dialogue-SFT indexes at {datetime.now(timezone.utc).isoformat()}\n",
    encoding="utf-8",
)

listed = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
if listed.returncode == 0:
    retained = [line for line in listed.stdout.splitlines() if "GEMMA4_FULL_EVAL_V11" not in line]
    subprocess.run(
        ["crontab", "-"],
        input=("\n".join(retained) + "\n") if retained else "",
        text=True,
        check=True,
    )
print("All 1,000 GPU 1 Dialogue-SFT indexes are complete; temporary cron entry removed.", flush=True)
PY
