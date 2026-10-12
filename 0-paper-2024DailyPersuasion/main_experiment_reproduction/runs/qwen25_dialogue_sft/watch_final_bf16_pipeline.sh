#!/usr/bin/env bash
set -u

TARGET_PID="${1:?pipeline PID required}"
TARGET_PGID="${2:?pipeline process group required}"

while kill -0 "${TARGET_PID}" 2>/dev/null; do
  if ps -eo pgid=,stat= | awk -v pgid="${TARGET_PGID}" '$1 == pgid && $2 ~ /T/ { stopped=1 } END { exit !stopped }'; then
    printf '%s SIGCONT process group %s\n' "$(date '+%F %T')" "${TARGET_PGID}"
    kill -CONT -- "-${TARGET_PGID}" 2>/dev/null || true
  fi
  sleep 5
done

printf '%s pipeline %s exited\n' "$(date '+%F %T')" "${TARGET_PID}"
