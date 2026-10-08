#!/usr/bin/env bash
set -euo pipefail

ROOT="/app/hzj/liujianjian/ljj-2-paper/baseline5_latentRATT"
EXP_ROOT="/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj"
PYTHON_BIN="/app/hzj/anaconda3/envs/ljj/bin/python"
METHOD="latentRATT_persuasion"
GENERATION_DIR="$EXP_ROOT/artifacts/generation_5rounds"
SELENE="/app/hzj/models/Selene-1-Mini-Llama-3.1-8B"
GEMMA="/app/hzj/models/gemma-4-E4B-it"

if [[ "${LATENTRATT_EVALUATION_READY:-}" != "1" ]]; then
  echo "Evaluation deferred until prompts are ready; set LATENTRATT_EVALUATION_READY=1 to run."
  exit 0
fi

test -s "$GENERATION_DIR/$METHOD.jsonl"

DAILYPERSUASION_PROFILE=paper10k ALLOW_JUDGE_EVAL=1 "$PYTHON_BIN" "$ROOT/evaluate_latent_judges.py" selene-joint \
  --generation-dir "$GENERATION_DIR" \
  --output-dir "$EXP_ROOT/artifacts/success_judgments/selene1_joint_v17_20260928" \
  --full-evaluation --methods "$METHOD" \
  --judge-model "$SELENE" --judge-name selene1 --judge-device cuda:2 \
  --prompt-version v17 --max-tokens 384 --max-parse-retries 3

DAILYPERSUASION_PROFILE=paper10k ALLOW_JUDGE_EVAL=1 "$PYTHON_BIN" "$ROOT/evaluate_latent_judges.py" gemma-joint \
  --generation-dir "$GENERATION_DIR" \
  --output-dir "$EXP_ROOT/artifacts/success_judgments/gemma4" \
  --full-evaluation --methods "$METHOD" \
  --judge-model "$GEMMA" --judge-name gemma4 --judge-device cuda:2 \
  --prompt-version v11 --max-tokens 384 --max-parse-retries 3

"$PYTHON_BIN" "$ROOT/update_latentratt_main_table.py"
