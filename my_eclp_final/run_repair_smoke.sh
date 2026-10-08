#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/app/hzj/liujianjian/ljj-2-paper
PYTHON=${ECLP_REPAIR_PYTHON:-/app/hzj/anaconda3/envs/ljj/bin/python}
CODE=${ECLP_REPAIR_CODE:-$ROOT/my_eclp_final}
WORK=$ROOT/eclp_runs_v21_repair
CFG=$CODE/config_repair.json
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${ECLP_REPAIR_GPU:-4}
cd "$ROOT"
mkdir -p "$WORK"

status() {
  "$PYTHON" - "$WORK/RUN_STATUS.json" "$1" "${2:-}" <<'PY'
import json, os, sys, time
from pathlib import Path
path = Path(sys.argv[1])
temp = path.with_suffix('.tmp')
temp.write_text(json.dumps({'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'phase':sys.argv[2], 'detail':sys.argv[3], 'task_effectiveness_verified':False},
    ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
os.replace(temp,path)
PY
  printf '%s\n' "$1: ${2:-}"
}
failed() {
  local code=$?
  trap - ERR
  status FAILED "exit=$code command=$BASH_COMMAND; see repair_smoke.log"
  exit "$code"
}
trap failed ERR

status MECHANISM "A100 BF16: four modes, three seeds, four feedback cycles"
"$PYTHON" "$CODE/check_eclp_mechanism.py" --device cuda:0 \
  --output "$WORK/mechanism_repaired.json"
if ! test -f "$WORK/protocol.lock.json"; then
  status LOCK "Hashing original source, data and three local model checkpoints"
  "$PYTHON" "$CODE/eclp_elpo.py" --config "$CFG" lock
fi
if ! test -d "$WORK/prepared"; then
  status PREPARE "Reading original ISR sessions; checking disjoint scenario partitions"
  "$PYTHON" "$CODE/eclp_elpo.py" --config "$CFG" prepare --output "$WORK/prepared"
fi
status REAL_GPU_SMOKE "Real Qwen2.5 actor: forward/backward, gradients, optimizer, generation; then fixed Persuadee"
"$PYTHON" "$CODE/eclp_elpo.py" --config "$CFG" smoke \
  --prepared "$WORK/prepared" --output "$WORK/gpu_smoke_repaired.json" --sequential-models
status PASS "Code integration verified; persuasion effectiveness still requires controlled experiments"
