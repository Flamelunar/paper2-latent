#!/usr/bin/env bash
set -euo pipefail

if (( $# != 5 )); then
  echo "Usage: $0 VERSION METHOD_SLUG MODEL_PATH ARCHITECTURE DISPLAY_NAME" >&2
  exit 2
fi

VERSION="$1"
METHOD_SLUG="$2"
MODEL_PATH="$3"
ARCHITECTURE="$4"
DISPLAY_NAME="$5"
ROOT="/app/hzj/liujianjian/ljj-2-paper"
PYTHON_BIN="/app/hzj/anaconda3/envs/ljj/bin/python"
EXP="$ROOT/exp_code_ljj"
GEN_DIR="$EXP/artifacts/generation_500/$METHOD_SLUG"
SEED=42
PROTOCOL="latentRATT_${VERSION}_fixed_first500_paired_seed${SEED}_v2"
mkdir -p "$GEN_DIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES

"$PYTHON_BIN" -u "$ROOT/baseline5_latentRATT/generate_1000.py" \
  --checkpoint "$MODEL_PATH" \
  --persuader-device cuda:2 --persuadee-device cuda:2 \
  --persuadee-model /app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged \
  --output "$GEN_DIR/$METHOD_SLUG.jsonl" \
  --limit 500 --architecture "$ARCHITECTURE" \
  --method-slug "$METHOD_SLUG" --display-name "$DISPLAY_NAME" \
  --protocol "$PROTOCOL" --max-new-tokens 512 --seed "$SEED" \
  --min-free-memory-mib 50000

"$PYTHON_BIN" -u "$ROOT/baseline5_latentRATT/evaluate_variant_500.py" \
  --method-slug "$METHOD_SLUG" --display-name "$DISPLAY_NAME" \
  --model-path "$MODEL_PATH" \
  --generation-dir "$GEN_DIR" \
  --output-dir "$EXP/artifacts/success_judgments/${METHOD_SLUG}_selene_v17" \
  --judge-model /app/hzj/models/Selene-1-Mini-Llama-3.1-8B \
  --judge-name selene1 --judge-device cuda:2 --prompt-version selene_v17

"$PYTHON_BIN" -u "$ROOT/baseline5_latentRATT/evaluate_variant_500.py" \
  --method-slug "$METHOD_SLUG" --display-name "$DISPLAY_NAME" \
  --model-path "$MODEL_PATH" \
  --generation-dir "$GEN_DIR" \
  --output-dir "$EXP/artifacts/success_judgments/${METHOD_SLUG}_gemma_v11" \
  --judge-model /app/hzj/models/gemma-4-E4B-it \
  --judge-name gemma4 --judge-device cuda:2 --prompt-version gemma_v11

if [[ "$VERSION" == "v0" ]]; then
  echo "Completed paired-seed v0 baseline generation and evaluation."
else
  "$PYTHON_BIN" -u "$ROOT/baseline5_latentRATT/summarize_variant_500.py" \
    --method-slug "$METHOD_SLUG" \
    --selene-dir "$EXP/artifacts/success_judgments/${METHOD_SLUG}_selene_v17" \
    --gemma-dir "$EXP/artifacts/success_judgments/${METHOD_SLUG}_gemma_v11" \
    --output "$EXP/results/${METHOD_SLUG}_first500.md"
fi

echo "Completed the first-500 generation and Selene/Gemma evaluation for $METHOD_SLUG."
