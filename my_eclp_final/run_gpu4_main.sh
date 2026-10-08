#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/app/hzj/liujianjian/ljj-2-paper
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1
export DAILYPERSUASION_PROFILE=paper10k
case "${1:-}" in
  full)
    shift
    exec "$ROOT/my_eclp_final/run_gpu4_full_chain.sh" "$@"
    ;;
  status)
    exec "$ROOT/my_eclp_final/monitor_gpu4_status.sh"
    ;;
  *)
    echo 'usage: run_gpu4_main.sh {full|status}' >&2
    exit 2
    ;;
esac
