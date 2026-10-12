#!/usr/bin/env bash
set -u

ROOT="/home1/liujianjian/2-paper-CoLING/0-paper-2024DailyPersuasion/main_experiment_reproduction"
PYTHON="/home1/liujianjian/anaconda3/envs/ljj/bin/python"
RUNS="${ROOT}/runs/qwen25_dialogue_sft"
cd "${ROOT}"

while true; do
  bash "${RUNS}/run_final_bf16_pipeline_resume.sh" &
  PIPELINE_PID=$!
  PIPELINE_PGID=""
  for _ in $(seq 1 50); do
    PIPELINE_PGID=$(ps -o pgid= -p "${PIPELINE_PID}" | tr -d ' ')
    if [[ -n "${PIPELINE_PGID}" ]]; then
      break
    fi
    sleep 0.1
  done
  if [[ -z "${PIPELINE_PGID}" ]]; then
    echo "Could not identify pipeline process group." >&2
    exit 1
  fi

  bash "${RUNS}/watch_final_bf16_pipeline.sh" \
    "${PIPELINE_PID}" "${PIPELINE_PGID}" &
  WATCHDOG_PID=$!
  wait "${PIPELINE_PID}"
  PIPELINE_STATUS=$?
  kill "${WATCHDOG_PID}" 2>/dev/null || true
  wait "${WATCHDOG_PID}" 2>/dev/null || true

  if [[ "${PIPELINE_STATUS}" -eq 0 ]]; then
    if "${PYTHON}" - <<'PY'
import json
from pathlib import Path

root = Path("runs/qwen25_dialogue_sft")
paths = [
    root / "generation_final_bf16/qwen25_dialogue_sft.jsonl",
    root / "evaluation_final_bf16_qwen35/qwen25_dialogue_sft.jsonl",
    root / "evaluation_final_bf16_persugpt/qwen25_dialogue_sft.jsonl",
]
for path in paths:
    if not path.is_file():
        raise SystemExit(1)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1000 or {int(row["index"]) for row in rows} != set(range(1000)):
        raise SystemExit(1)
if not Path("results/qwen25_dialogue_sft_table2_bf16_final.md").is_file():
    raise SystemExit(1)
if not Path("results/qwen25_dialogue_sft_table2_bf16_final.json").is_file():
    raise SystemExit(1)
PY
    then
      echo "Verified complete BF16 generation, both Judge outputs, and final report."
      exit 0
    fi
    echo "Pipeline exited successfully but output coverage is incomplete; resuming."
  elif [[ "${PIPELINE_STATUS}" -eq 130 || "${PIPELINE_STATUS}" -eq 137 || "${PIPELINE_STATUS}" -eq 143 ]]; then
    echo "Pipeline was interrupted with status ${PIPELINE_STATUS}; resuming from saved rows."
  else
    echo "Pipeline failed with status ${PIPELINE_STATUS}; stopping for diagnosis." >&2
    exit "${PIPELINE_STATUS}"
  fi
  sleep 5
done
