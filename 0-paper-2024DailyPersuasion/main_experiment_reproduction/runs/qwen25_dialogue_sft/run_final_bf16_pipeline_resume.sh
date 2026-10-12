#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Received SIGINT; will resume from saved rows if generation is incomplete."' INT
trap 'echo "Received SIGTERM; will resume from saved rows if generation is incomplete."' TERM

ROOT="/home1/liujianjian/2-paper-CoLING/0-paper-2024DailyPersuasion/main_experiment_reproduction"
PYTHON="/home1/liujianjian/anaconda3/envs/ljj/bin/python"
RUNS="${ROOT}/runs/qwen25_dialogue_sft"

cd "${ROOT}"
mkdir -p "${RUNS}/logs"

echo "Resuming Dialogue-SFT generation with BF16; existing compatible rows will be skipped."
INTERRUPTIONS=0
while true; do
  set +e
  "${PYTHON}" -u run_qwen25_table2_generation.py \
    --profile paper10k \
    --method qwen25_dialogue_sft \
    --output-dir "${RUNS}/generation_final_bf16" \
    --persuader-device cuda:1 \
    --persuadee-device cuda:0,cuda:2 \
    --context-batch-size 2 \
    >> "${RUNS}/logs/generation_final_bf16_resume.log" 2>&1
  GENERATION_STATUS=$?
  set -e

  GENERATED_ROWS=$("${PYTHON}" - <<'PY'
import json
from pathlib import Path

path = Path("runs/qwen25_dialogue_sft/generation_final_bf16/qwen25_dialogue_sft.jsonl")
print(sum(bool(line.strip()) for line in path.open(encoding="utf-8")) if path.exists() else 0)
PY
  )
  if [[ "${GENERATION_STATUS}" -eq 0 || "${GENERATED_ROWS}" -eq 1000 ]]; then
    break
  fi
  if [[ "${GENERATION_STATUS}" -ne 130 && "${GENERATION_STATUS}" -ne 143 ]]; then
    echo "Generation failed with exit code ${GENERATION_STATUS} at ${GENERATED_ROWS}/1000 rows." >&2
    exit "${GENERATION_STATUS}"
  fi
  INTERRUPTIONS=$((INTERRUPTIONS + 1))
  if [[ "${INTERRUPTIONS}" -gt 100 ]]; then
    echo "Generation exceeded 100 interrupted resumes at ${GENERATED_ROWS}/1000 rows." >&2
    exit 1
  fi
  echo "Generation interrupted with exit code ${GENERATION_STATUS}; resuming at ${GENERATED_ROWS}/1000 rows (attempt ${INTERRUPTIONS})."
  sleep 3
done

"${PYTHON}" - <<'PY'
import json
from pathlib import Path

path = Path("runs/qwen25_dialogue_sft/generation_final_bf16/qwen25_dialogue_sft.jsonl")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
indices = {int(row["index"]) for row in rows}
if len(rows) != 1000 or indices != set(range(1000)):
    raise SystemExit(f"Expected 1000 unique sequential contexts, found {len(rows)} rows")
if {row.get("rounds") for row in rows} != {5}:
    raise SystemExit("Expected every generated dialogue to contain five rounds")
print("Validated 1000 unique five-round BF16 generation records.")
PY

echo "Starting both BF16 Judge evaluations."
"${PYTHON}" -u evaluate_qwen25_table2.py \
  --profile paper10k \
  --generation-dir "${RUNS}/generation_final_bf16" \
  --evaluation-dir "${RUNS}/evaluation_final_bf16_qwen35" \
  --judge-model /data1/liujianjian/model/Qwen3.5-9B \
  --judge-device cuda:1 \
  --judge-replicas 1 \
  --judge-batch-size 4 \
  --method qwen25_dialogue_sft \
  > "${RUNS}/logs/evaluation_final_bf16_qwen35.log" 2>&1 &
QWEN35_PID=$!

"${PYTHON}" -u evaluate_qwen25_table2.py \
  --profile paper10k \
  --generation-dir "${RUNS}/generation_final_bf16" \
  --evaluation-dir "${RUNS}/evaluation_final_bf16_persugpt" \
  --judge-model /data1/liujianjian/PersuGPT \
  --judge-device cuda:2 \
  --judge-replicas 1 \
  --judge-batch-size 4 \
  --method qwen25_dialogue_sft \
  > "${RUNS}/logs/evaluation_final_bf16_persugpt.log" 2>&1 &
PERSUGPT_PID=$!

QWEN35_STATUS=0
PERSUGPT_STATUS=0
wait "${QWEN35_PID}" || QWEN35_STATUS=$?
wait "${PERSUGPT_PID}" || PERSUGPT_STATUS=$?
if [[ "${QWEN35_STATUS}" -ne 0 || "${PERSUGPT_STATUS}" -ne 0 ]]; then
  echo "Judge failure: Qwen3.5=${QWEN35_STATUS}, PersuGPT=${PERSUGPT_STATUS}" >&2
  exit 1
fi
"${PYTHON}" -u finalize_qwen25_bf16_results.py \
  > "${RUNS}/logs/finalize_qwen25_bf16_results.log" 2>&1
echo "Final BF16 Table 2-style report is ready."
