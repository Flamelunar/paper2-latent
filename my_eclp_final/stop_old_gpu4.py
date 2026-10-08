#!/usr/bin/env python3
"""Stop verified old GPU4 jobs and remove only their old run artifacts."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import time

ROOT = Path("/app/hzj/liujianjian/ljj-2-paper")
JOBS = {
    1026845: b"my_eclp_final/config_repair.json sft",
    862867: b"start_post_sft_chain.sh",
}


def matches(pid: int, needle: bytes) -> bool:
    path = Path("/proc") / str(pid) / "cmdline"
    return path.exists() and needle in path.read_bytes().replace(b"\x00", b" ")


for pid, needle in JOBS.items():
    if matches(pid, needle):
        print(f"TERM verified old PID {pid}", flush=True)
        os.kill(pid, signal.SIGTERM)
for repeat in range(30):
    if not any(matches(pid, needle) for pid, needle in JOBS.items()):
        break
    time.sleep(1)
else:
    raise RuntimeError("Old GPU4 process has not exited; refusing to remove its live outputs")

old = ROOT / "eclp_runs_v21_repair"
for relative in ("sft_full", "sft_full.log"):
    target = old / relative
    if target.is_dir():
        shutil.rmtree(target)
        print(f"Removed old incomplete run directory: {target}", flush=True)
    elif target.is_file():
        target.unlink()
        print(f"Removed old incomplete log: {target}", flush=True)
work = ROOT / "eclp_runs_gpu4"
if work.exists():
    if any(work.iterdir()):
        raise RuntimeError("New GPU4 work directory already contains data; refusing to delete it")
else:
    work.mkdir()
print("Old GPU4 jobs stopped; repaired validation artifacts retained; new GPU4 work ready", flush=True)
