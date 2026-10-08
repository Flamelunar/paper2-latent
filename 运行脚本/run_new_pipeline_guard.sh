#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/app/hzj/liujianjian/ljj-2-paper
EXP="$ROOT/exp_code_ljj"
LOG="$EXP/artifacts/logs/new_experiments/pipeline_guard_v2.log"
exec 6>"$EXP/artifacts/.new_pipeline_guard.lock"
if ! flock -n 6; then exit 73; fi

guard() {
  local family=$1 base=$2 gpu=$3 code
  local b4="$EXP/artifacts/generation_5rounds/$family/baseline4_ma2p.jsonl"
  while :; do
    if [[ -f "$b4" ]] && [[ $(wc -l < "$b4") -eq 1000 ]]; then return; fi
    # Anchored argv matching cannot match this guard's own source text.
    if pgrep -f "^bash $ROOT/run_post_b2_family[.]sh $family " >/dev/null; then
      sleep 180
    else
      code=$?
      if [[ "$code" -ne 1 ]]; then return "$code"; fi
      printf '%s restarting checkpoint-safe %s chain on GPU %s\n' "$(date -Is)" "$family" "$gpu" >> "$LOG"
      nohup bash "$ROOT/run_post_b2_family.sh" "$family" "$base" "$gpu" >> "$LOG" 2>&1 &
      sleep 180
    fi
  done
}
guard meta_llama31 /app/hzj/models/Meta-Llama-3.1-8B-Instruct 0 &
llama=$!
guard deepseek_r1_qwen14b /app/hzj/models/DeepSeek-R1-Distill-Qwen-14B 1 &
deepseek=$!
status=0
if wait "$llama"; then :; else status=$?; fi
if wait "$deepseek"; then :; else status=$?; fi
exit "$status"
