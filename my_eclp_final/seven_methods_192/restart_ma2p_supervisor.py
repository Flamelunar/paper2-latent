"""Restart only this experiment's controller, adopting its surviving workers."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/app/hzj/liujianjian/ljj-2-paper")
WORK = ROOT / "exp_code_ljj/artifacts/seven_methods_192_20261006"
CODE = ROOT / "my_eclp_final/seven_methods_192"
sys.path.insert(0, str(ROOT / "my_eclp_final"))
from experiment_io import append_jsonl, atomic_json, read_json


def active(pid):
    path = Path(f"/proc/{pid}/status")
    if not path.exists():
        return False
    state = next(line for line in path.read_text().splitlines() if line.startswith("State:"))
    return "Z (" not in state


def main():
    status = read_json(WORK / "live_status.json")
    pid = status["supervisor_pid"]
    if active(pid):
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        if str(CODE / "supervise_parallel.py").encode() not in command or str(WORK).encode() not in command:
            raise RuntimeError("Controller identity did not match this experiment")
        os.kill(pid, signal.SIGTERM)
    with (WORK / ".supervisor.lock").open("a") as handle:
        deadline = time.monotonic() + 15
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Original controller did not release its lock")
                time.sleep(.25)
        latest = read_json(WORK / "live_status.json")
        workers = []
        for task in latest["workers"].values():
            if not active(task["pid"]):
                continue
            command = Path(f"/proc/{task['pid']}/cmdline").read_bytes()
            if str(WORK).encode() not in command:
                raise RuntimeError(f"Worker {task['pid']} belongs to another experiment")
            workers.append({key: value for key, value in task.items() if key != "state"})
        atomic_json(WORK / f"archive/ma2p_recovery_supervisor_snapshot_{pid}.json", latest)
        atomic_json(WORK / "adopt_workers.json", workers)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    command = [sys.executable, "-u", str(CODE / "supervise_parallel.py"),
               "--repo-root", str(ROOT), "--output", str(WORK)]
    with (WORK / "supervisor_parallel.log").open("a") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    record = {"previous_supervisor_pid": pid, "supervisor_pid": process.pid,
              "adopted_worker_pids": [task["pid"] for task in workers],
              "stopped_gpu_workers": [], "started_unix": time.time()}
    atomic_json(WORK / "ma2p_recovery_supervisor_restart.json", record)
    append_jsonl(WORK / "ma2p_recovery_supervisor_restart_history.jsonl", record)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
