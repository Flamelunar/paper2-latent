#!/usr/bin/env python3
"""Resume the DeepSeek LatentRATT baseline on GPU4 after the active fresh campaign."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/app/hzj/liujianjian/ljj-2-paper")
MODEL = Path("/app/hzj/models/DeepSeek-R1-Distill-Qwen-14B")
SIMULATOR = Path("/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged")
GPU = int(os.environ.get("DEEPSEEK_TRAIN_GPU", "1"))
BASE = Path("/app/hzj/liujianjian/persuader_atapter_save/deepseek_r1_qwen14b/baseline5_latentratt")
OUT = ROOT / "exp_code_ljj/artifacts/deepseek_goal_strategy_192_20261008/deepseek_r1_qwen14b"
QUEUE = OUT / "latentratt_queue"
LOGS = ROOT / "baseline5_latentRATT/logs/repaired192_deepseek_r1_qwen14b"
TRAINER = ROOT / "my_eclp_final/model_family_jobs/train_with_budget.py"
GENERATOR = ROOT / "my_eclp_final/model_family_jobs/generate_latent_family.py"
EVALUATOR = ROOT / "my_eclp_final/seven_methods_192/evaluate.py"
REPORTER = ROOT / "my_eclp_final/model_family_jobs/report_family.py"
DATA = ROOT / "exp_code_ljj/data/isr_sft.jsonl"
SELECTION = ROOT / "exp_code_ljj/data/test_selection_192.json"
POINTS = ROOT / "exp_code_ljj/data/evaluation_points.jsonl"
FRESH_GENERATION = OUT / "generation_run.json"
METHODS = ("zero_shot_repaired", "dialogue_sft", "persugpt", "ma2p_repaired", "our_direct")
JUDGES = ("selene8b", "qwen3_14b")
PYTHON = "/app/hzj/anaconda3/envs/ljj/bin/python"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def update(stage: str, **extra) -> None:
    atomic_json(QUEUE / "status.json", {"updated": time.time(), "gpu": GPU, "stage": stage, **extra})


def active_campaigns() -> list[int]:
    active = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().decode(errors="replace")
        except OSError:
            continue
        if "llama_fresh_campaign.py" in command:
            active.append(pid)
    return active


def gpu_free_mib() -> int:
    raw = subprocess.check_output(
        ["nvidia-smi", f"--id={GPU}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    return int(raw)


def wait_for_gpu(min_free_gib: int, stage: str) -> None:
    while True:
        free = gpu_free_mib()
        if free >= min_free_gib * 1024:
            return
        update(stage, free_mib=free, required_mib=min_free_gib * 1024)
        time.sleep(60)


def run(command: list[str], log_path: Path, *, stage: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    update(stage, command=command, log=str(log_path))
    with log_path.open("ab", buffering=0) as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"{stage} exited {completed.returncode}; see {log_path}")


def verify_active_suite() -> None:
    work = OUT
    missing = []
    for method in METHODS:
        generation = work / "generators" / method / "generation" / f"{method}_summary.json"
        if not generation.exists() or not read_json(generation).get("complete"):
            missing.append(f"generation:{method}")
        for judge in JUDGES:
            summary = work / "judges" / judge / "evaluation" / method / "summary.json"
            if not summary.exists():
                missing.append(f"evaluation:{judge}:{method}")
            else:
                value = read_json(summary)
                if not value.get("complete") or value.get("n") != 192:
                    missing.append(f"evaluation:{judge}:{method}")
    if missing:
        raise RuntimeError("Fresh campaign ended before required methods completed: " + ", ".join(missing))


def latest_training_checkpoint(output: Path, stage: str) -> Path | None:
    candidates = []
    paths = [output / "run_config.json"]
    paths.extend(output.glob("checkpoint-*/run_config.json"))
    paths.extend(output.glob("oom-step-*/run_config.json"))
    for config_path in paths:
        if not config_path.is_file():
            continue
        config = read_json(config_path)
        if (config.get("stage") == stage and config.get("model") == str(MODEL)
                and config.get("train_rows") == 59868 and config.get("train_sha256") == sha256(DATA)):
            state = config_path.parent / "training_state.pt"
            adapter = config_path.parent / "adapter/adapter_config.json"
            attention = config_path.parent / "attention.pt"
            if state.is_file() and adapter.is_file() and attention.is_file():
                candidates.append((int(config.get("completed_steps", -1)), config_path.parent))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def assert_training_config(config: dict, stage: str, *, complete: bool | None = None) -> None:
    if (config.get("stage") != stage or config.get("model") != str(MODEL)
            or config.get("train_rows") != 59868 or config.get("train_sha256") != sha256(DATA)):
        raise RuntimeError(f"LatentRATT {stage} checkpoint does not match the complete DeepSeek ISR run")
    if complete is not None and config.get("training_complete") is not complete:
        raise RuntimeError(f"LatentRATT {stage} checkpoint has unexpected completion state")


def verify_latent_checkpoint(path: Path) -> dict:
    config = read_json(path / "run_config.json")
    if (config.get("stage") != "rl" or config.get("model") != str(MODEL)
            or config.get("training_complete") is not True or config.get("train_rows") != 59868
            or config.get("train_sha256") != sha256(DATA)):
        raise RuntimeError("LatentRATT RL checkpoint does not match the complete DeepSeek ISR run")
    for relative in ("adapter/adapter_config.json", "attention.pt"):
        if not (path / relative).is_file():
            raise RuntimeError(f"Incomplete LatentRATT checkpoint: {relative}")
    return config


def train_latentratt() -> Path:
    train_config = read_json(ROOT / "eclp_runs_deepseek_r1_qwen14b_20261007/config.json")
    if train_config.get("base_model") != str(MODEL):
        raise RuntimeError("The selected DeepSeek backbone differs from the active ECLP family")
    sft_dir = BASE / "sft"
    rl_dir = BASE / "rl"
    if (sft_dir / "run_config.json").exists() and read_json(sft_dir / "run_config.json").get("training_complete") is True:
        assert_training_config(read_json(sft_dir / "run_config.json"), "sft", complete=True)
        sft_checkpoint = sft_dir
    else:
        checkpoint = latest_training_checkpoint(sft_dir, "sft")
        if checkpoint is None:
            sft_config = {"stage":"sft", "model":str(MODEL), "train_rows":59868,
                          "train_sha256":sha256(DATA), "training_complete":False}
        else:
            sft_config = read_json(checkpoint / "run_config.json")
        if sft_config.get("training_complete") is True:
            assert_training_config(sft_config, "sft", complete=True)
            sft_checkpoint = checkpoint
        else:
            assert_training_config(sft_config, "sft", complete=False)
            wait_for_gpu(44 if GPU == 4 else 62, "waiting_for_gpu_before_sft_resume")
            command = [
                PYTHON, "-u", str(TRAINER), "--entry", "latent", "--physical-gpu", str(GPU),
                "--memory-cap-gib", "54", "--", "sft", "--model", str(MODEL),
                "--device", "cuda:0", "--train-file", str(DATA),
                *(["--checkpoint", str(checkpoint), "--resume-training-state", str(checkpoint / "training_state.pt")] if checkpoint is not None else []),
                "--output", str(sft_dir), "--epochs", "1", "--max-prompt-tokens", "1536",
                "--max-target-tokens", "256", "--gradient-accumulation", "16",
                "--batch-size", "2", "--group-size", "4", "--sigma", "1.5",
                "--rationale-weight", "0.2", "--learning-rate", "0.0001",
                "--save-steps", "100", "--log-steps", "10", "--seed", "42",
            ]
            run(command, LOGS / "sft.log", stage=f"resuming_latentratt_sft_on_gpu{GPU}")
            sft_config = read_json(sft_dir / "run_config.json")
            if sft_config.get("training_complete") is not True:
                raise RuntimeError("LatentRATT SFT exited without completing its single full data pass")
            sft_checkpoint = sft_dir

    rl_summary = rl_dir / "run_config.json"
    if rl_summary.exists() and read_json(rl_summary).get("training_complete") is True:
        verify_latent_checkpoint(rl_dir)
    else:
        resume = latest_training_checkpoint(rl_dir, "rl")
        rl_input = sft_checkpoint
        resume_args = []
        if resume is not None:
            resume_config = read_json(resume / "run_config.json")
            assert_training_config(resume_config, "rl", complete=False)
            rl_input = resume
            resume_args = ["--resume-training-state", str(resume / "training_state.pt")]
        wait_for_gpu(44 if GPU == 4 else 62, "waiting_for_gpu_before_latentratt_rl")
        command = [
            PYTHON, "-u", str(TRAINER), "--entry", "latent", "--physical-gpu", str(GPU),
            "--memory-cap-gib", "54", "--", "rl", "--model", str(MODEL),
            "--device", "cuda:0", "--train-file", str(DATA), "--checkpoint", str(rl_input),
            *resume_args,
            "--output", str(rl_dir), "--epochs", "1", "--max-prompt-tokens", "1536",
            "--max-target-tokens", "256", "--gradient-accumulation", "16",
            "--batch-size", "2", "--group-size", "4", "--sigma", "1.5",
            "--rationale-weight", "0.2", "--learning-rate", "0.00001",
            "--save-steps", "100", "--log-steps", "10", "--seed", "42",
        ]
        run(command, LOGS / "rl.log", stage=f"training_latentratt_rl_on_gpu{GPU}")
    verify_latent_checkpoint(rl_dir)
    return rl_dir


if __name__=="__main__":
    import fcntl
    QUEUE.mkdir(parents=True,exist_ok=True)
    with (QUEUE/".training.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        checkpoint=train_latentratt()
        update("training_complete",checkpoint=str(checkpoint))
