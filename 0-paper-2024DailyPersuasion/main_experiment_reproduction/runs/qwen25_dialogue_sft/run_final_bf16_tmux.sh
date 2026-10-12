#!/usr/bin/env bash
set -u

ROOT="/home1/liujianjian/2-paper-CoLING/0-paper-2024DailyPersuasion/main_experiment_reproduction"
cd "${ROOT}"

bash runs/qwen25_dialogue_sft/run_final_bf16_pipeline_resume.sh &
PIPELINE_PID=$!
for _ in $(seq 1 50); do
  PIPELINE_PGID=$(ps -o pgid= -p "${PIPELINE_PID}" | tr -d ' ')
  if [[ -n "${PIPELINE_PGID}" ]]; then
    break
  fi
  sleep 0.1
done

if [[ -z "${PIPELINE_PGID:-}" ]]; then
  echo "Could not identify the BF16 pipeline process group." >&2
  exit 1
fi

bash runs/qwen25_dialogue_sft/watch_final_bf16_pipeline.sh \
  "${PIPELINE_PID}" "${PIPELINE_PGID}" &
WATCHDOG_PID=$!

wait "${PIPELINE_PID}"
PIPELINE_STATUS=$?
kill "${WATCHDOG_PID}" 2>/dev/null || true
wait "${WATCHDOG_PID}" 2>/dev/null || true
exit "${PIPELINE_STATUS}"
