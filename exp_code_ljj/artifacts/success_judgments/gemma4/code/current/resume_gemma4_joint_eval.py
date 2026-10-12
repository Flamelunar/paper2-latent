#!/usr/bin/env python3
"""Supervise checkpoint resumes for the Gemma 4 full joint evaluation."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
PRIMARY = ARTIFACTS / "success_judgments/gemma4"
GENERATION = ARTIFACTS / "generation_5rounds/legacy"
META_PATH = PRIMARY / "run_metadata.json"
JUDGE_MODEL = "/data1/liujianjian/model/gemma-4-E4B-it"
METHODS = ("qwen25_zero_shot", "qwen25_dialogue_sft")
REVISION = "joint_v2_r11"
DEVICE = "cuda:1"
MAX_ATTEMPTS = 100
PAUSE_SENTINEL = Path.home() / ".cache" / "pause_gemma4_joint_eval"
NO_JUDGE_SENTINEL = Path.home() / ".cache" / "no_judge_latentratt_59868"

stop_signal: int | None = None
active_child: subprocess.Popen[bytes] | None = None


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def generation_ids(method: str) -> dict[int, str]:
    rows = read_jsonl(GENERATION / f"{method}.jsonl")
    ids = {int(row["index"]): str(row["scenario_id"]) for row in rows}
    if len(rows) != 1000 or len(ids) != 1000 or set(ids) != set(range(1000)):
        raise ValueError(f"Generation source must have exactly indexes 0..999: {method}")
    return ids


def completed_indexes(method: str, scenario_ids: dict[int, str]) -> set[int]:
    rows = read_jsonl(PRIMARY / f"{method}.jsonl")
    completed: set[int] = set()
    for row in rows:
        if row.get("judge_device") != DEVICE:
            continue
        index = int(row["index"])
        if index not in scenario_ids or row.get("scenario_id") != scenario_ids[index]:
            raise ValueError(f"Generation mismatch in {method} at index {index}")
        if (row.get("method") != method
                or row.get("judge_model") != JUDGE_MODEL
                or row.get("trial") != 1
                or row.get("prompt_version") != "v11"
                or row.get("prompt_revision") != REVISION
                or row.get("judge_compute_dtype") != "bfloat16"
                or row.get("judge_quantization") != "none"
                or row.get("max_parse_retries") != 3):
            raise ValueError(f"Unexpected evaluation metadata in {method} at index {index}")
        completed.add(index)
    return completed


def update_metadata(mutator) -> dict[str, Any]:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    mutator(meta)
    fd, temp_name = tempfile.mkstemp(prefix=".run_metadata.", suffix=".tmp", dir=PRIMARY)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, META_PATH)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return meta


def interrupted_handler(signum: int, _frame: Any) -> None:
    global stop_signal
    stop_signal = signum
    child = active_child
    if child is not None and child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def child_command() -> list[str]:
    return [
        sys.executable, "-u", "evaluate_joint_prompt_probe.py",
        "--full-evaluation", "--methods", *METHODS,
        "--output-dir", "artifacts/success_judgments/gemma4",
        "--judge-model", JUDGE_MODEL,
        "--judge-name", "gemma4",
        "--judge-device", DEVICE,
        "--prompt-version", "v11",
        "--max-tokens", "384",
        "--max-parse-retries", "3",
    ]


def main() -> int:
    global active_child
    if os.environ.get("ALLOW_JUDGE_EVAL") != "1":
        print("Judge evaluation is disabled; set ALLOW_JUDGE_EVAL=1 to resume.", flush=True)
        return 0
    if PAUSE_SENTINEL.exists() or NO_JUDGE_SENTINEL.exists():
        marker = PAUSE_SENTINEL if PAUSE_SENTINEL.exists() else NO_JUDGE_SENTINEL
        print(f"Evaluation paused by {marker}; remove it to resume.", flush=True)
        return 0

    # This supervisor runs detached from the invoking terminal. Ignore signals used
    # by shell/session cleanup so the checkpointed evaluation can continue. SIGUSR2
    # is reserved for an explicit graceful stop; it kills only the active child after
    # its last fully flushed JSONL record.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    signal.signal(signal.SIGUSR2, interrupted_handler)

    scenario_ids = {method: generation_ids(method) for method in METHODS}
    initial_counts = {method: completed_indexes(method, scenario_ids[method]) for method in METHODS}
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    previous_pid = int(metadata["process_pid"])
    previous_cmdline = Path(f"/proc/{previous_pid}/cmdline")
    if previous_cmdline.exists() and b"evaluate_joint_prompt_probe.py" in previous_cmdline.read_bytes():
        raise RuntimeError(f"Evaluator is still active; refusing a second resume: {previous_pid}")

    def mark_supervisor(meta: dict[str, Any]) -> None:
        history = meta.setdefault("process_attempt_history", [])
        if not any(item.get("process_pid") == previous_pid for item in history):
            history.append({
                "process_pid": previous_pid,
                "device": DEVICE,
                "status": "interrupted",
                "stdout_log": meta.get("stdout_log"),
                "confirmed_at_utc": now_utc(),
                "gpu1_dialogue_completed": len(initial_counts["qwen25_dialogue_sft"]),
                "pending_after_exit": {
                    method: 1000 - len(initial_counts[method]) for method in METHODS
                },
            })
        meta["resume_supervisor_pid"] = os.getpid()
        meta["resume_supervisor_started_at_utc"] = now_utc()
        meta["status"] = "resuming"
        meta["status_updated_at_utc"] = now_utc()
        meta.pop("stop_signal", None)

    update_metadata(mark_supervisor)
    print(f"Gemma 4 resume supervisor PID {os.getpid()} on {DEVICE}", flush=True)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if stop_signal is not None:
            break
        counts = {method: completed_indexes(method, scenario_ids[method]) for method in METHODS}
        missing = {method: 1000 - len(indexes) for method, indexes in counts.items()}
        if not any(missing.values()):
            update_metadata(lambda meta: meta.update({
                "status": "completed",
                "status_updated_at_utc": now_utc(),
                "completed_gpu1_rows": {method: len(counts[method]) for method in METHODS},
                "resume_supervisor_pid": os.getpid(),
            }))
            print("All 2,000 GPU 1 evaluation indexes are complete.", flush=True)
            return 0

        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = PRIMARY / f"run_supervised_attempt_{attempt:02d}_{suffix}.log"
        print(f"Attempt {attempt}/{MAX_ATTEMPTS}: saved={sum(map(len, counts.values()))}, "
              f"pending={missing}; log={log_path.name}", flush=True)
        with log_path.open("wb") as log_handle:
            active_child = subprocess.Popen(
                child_command(),
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        child_pid = active_child.pid

        def mark_child(meta: dict[str, Any]) -> None:
            meta["process_pid"] = child_pid
            meta["device"] = DEVICE
            meta["status"] = "running"
            meta["status_updated_at_utc"] = now_utc()
            meta["stdout_log"] = str(log_path.relative_to(ROOT))
            meta["managed_by_resume_supervisor"] = os.getpid()
            meta["managed_exec_session_id"] = None

        update_metadata(mark_child)
        while active_child.poll() is None and stop_signal is None:
            time.sleep(1)
        return_code = active_child.wait()
        active_child = None

        after = {method: completed_indexes(method, scenario_ids[method]) for method in METHODS}
        after_missing = {method: 1000 - len(indexes) for method, indexes in after.items()}
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        is_keyboard_interrupt = "KeyboardInterrupt" in log_text[-12000:]

        def record_attempt(meta: dict[str, Any]) -> None:
            history = meta.setdefault("process_attempt_history", [])
            history.append({
                "process_pid": child_pid,
                "device": DEVICE,
                "return_code": return_code,
                "status": "interrupted" if is_keyboard_interrupt or stop_signal else "exited",
                "stdout_log": str(log_path.relative_to(ROOT)),
                "confirmed_at_utc": now_utc(),
                "gpu1_dialogue_completed": len(after["qwen25_dialogue_sft"]),
                "pending_after_exit": after_missing,
            })
            meta["last_resume_return_code"] = return_code
            meta["last_resume_log"] = str(log_path.relative_to(ROOT))
            meta["status_updated_at_utc"] = now_utc()

        update_metadata(record_attempt)

        if stop_signal is not None:
            update_metadata(lambda meta: meta.update({"status": "interrupted", "stop_signal": stop_signal}))
            print(f"Supervisor stopped on signal {stop_signal}; checkpoint retained.", flush=True)
            return 128 + stop_signal
        if return_code == 0:
            if not any(after_missing.values()):
                update_metadata(lambda meta: meta.update({
                    "status": "completed",
                    "status_updated_at_utc": now_utc(),
                    "completed_gpu1_rows": {method: len(after[method]) for method in METHODS},
                }))
                print("Evaluation process exited successfully; all GPU 1 rows are complete.", flush=True)
                return 0
            update_metadata(lambda meta: meta.update({"status": "error", "incomplete_after_success_exit": after_missing}))
            print(f"Evaluator exited successfully but left missing rows: {after_missing}", flush=True)
            return 1
        if is_keyboard_interrupt:
            print(f"Evaluator {child_pid} received KeyboardInterrupt after saving "
                  f"{len(after['qwen25_dialogue_sft'])} Dialogue-SFT indexes; resuming.", flush=True)
            time.sleep(2)
            continue
        if return_code in {-signal.SIGINT, -signal.SIGTERM, -signal.SIGHUP}:
            print(f"Evaluator {child_pid} exited on signal {-return_code} after saving "
                  f"{len(after['qwen25_dialogue_sft'])} Dialogue-SFT indexes; resuming.", flush=True)
            time.sleep(2)
            continue
        update_metadata(lambda meta: meta.update({"status": "error", "last_error_return_code": return_code}))
        print(f"Evaluator exited with code {return_code}; inspect {log_path}", flush=True)
        return return_code or 1

    update_metadata(lambda meta: meta.update({
        "status": "interrupted",
        "status_updated_at_utc": now_utc(),
        "resume_supervisor_attempt_limit": MAX_ATTEMPTS,
        "stop_signal": stop_signal,
    }))
    print(f"Resume supervisor stopped before completion after {MAX_ATTEMPTS} attempts or a signal.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
