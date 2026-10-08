#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/app/hzj/liujianjian/ljj-2-paper
while true; do
  "$ROOT/my_eclp_final/monitor_gpu4_status.sh"
  sleep 30
done
