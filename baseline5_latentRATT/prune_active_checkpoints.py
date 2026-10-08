#!/usr/bin/env python3
"""Keep only the latest complete checkpoints while this experiment runs."""
from __future__ import annotations

import argparse
import os
import re
import shutil
import time
from pathlib import Path

NAME = re.compile(r"checkpoint-(\d+)")


def prune(root: Path, keep: int = 3) -> None:
    for stage in ("sft", "rl"):
        folder = root / stage
        if not folder.is_dir():
            continue
        checkpoints = [
            path for path in folder.iterdir()
            if path.is_dir() and NAME.fullmatch(path.name)
            and (path / "training_state.pt").is_file()
            and not (path / "training_state.pt.tmp").exists()
        ]
        checkpoints.sort(key=lambda path: int(NAME.fullmatch(path.name).group(1)))
        for stale in checkpoints[:-keep]:
            shutil.rmtree(stale)
            print(f"removed {stage}/{stale.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-pid", type=int, required=True)
    parser.add_argument("--root", type=Path, default=Path(
        "/app/hzj/liujianjian/qwen2.5-7b-latentRATT-all-59868"))
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    while True:
        prune(args.root)
        try:
            os.kill(args.runner_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(args.interval)
    prune(args.root)
    print("LatentRATT runner ended; final checkpoint retention pass completed.", flush=True)


if __name__ == "__main__":
    main()
