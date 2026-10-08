#!/usr/bin/env python3
"""Adopt the current SFT, then offline latent RL, test1000, two Judges, table."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

from experiment_io import (atomic_json, digest, immutable_json, object_hash,
                           read_json, writer_lock)

ROOT = Path(__file__).resolve().parents[1]
CODE = Path(__file__).resolve().parent
WORK = ROOT / "eclp_runs_gpu4"
PYTHON = "/app/hzj/anaconda3/envs/ljj/bin/python"
CONFIG = CODE / "config_gpu4.json"
PLAN = CODE / "offline_training_plan.json"
PREPARED = WORK / "prepared"


def process_identity(pid):
    path = Path("/proc") / str(pid)
    try:
        raw = (path / "stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        if fields[0] in {"Z", "X"}:
            return None
        command = (path / "cmdline").read_bytes().split(b"\0")
        return {"pid": int(pid), "start_ticks": int(fields[19]),
                "command": [item.decode() for item in command if item]}
    except FileNotFoundError:
        return None


class OfflinePipeline:
    def __init__(self, adoption: Path | None):
        self.plan = read_json(PLAN)
        self.adoption = read_json(adoption) if adoption else None
        self.child = None
        self.phase = "starting_offline_pipeline"
        self.started = time.monotonic()
        self.adopted_child = self.adoption.get("sft_child") if self.adoption else None
        os.environ.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="4",
                          PYTHONUNBUFFERED="1", DAILYPERSUASION_PROFILE="paper10k",
                          ALLOW_JUDGE_EVAL="1", TOKENIZERS_PARALLELISM="false")
        os.chdir(ROOT)
        names = ["eclp_elpo.py", "config_gpu4.json", "training_plan.json", "run_main_experiment.py",
                 "offline_latent_rl.py", "offline_training_plan.json",
                 "run_gpu4_offline_pipeline.py", "experiment_io.py", "evaluate_our.py",
                 "update_our_table.py", "final_eval_protocol.py", "final_judge_prompts.json"]
        self.contract = {
            "plan": self.plan, "code_sha256": {name: digest(CODE / name) for name in names},
            "original_pipeline_contract_hash": object_hash(read_json(WORK / "pipeline_contract.json")),
            "protocol_lock_hash": object_hash(read_json(WORK / "protocol.lock.json")),
            "physical_gpu": 4, "training_dialogue_calls": 0,
            "training_judge_calls": 0, "python": PYTHON,
        }
        immutable_json(WORK / "offline_pipeline_contract.json", self.contract)
        diagnostic = read_json(WORK / "offline_gpu_diagnostic.json")
        if (diagnostic.get("status") != "PASS_OFFLINE_GPU_DIAGNOSTIC_NOT_AN_EXPERIMENT_RESULT"
                or diagnostic.get("trainer_sha256") != self.contract["code_sha256"]["offline_latent_rl.py"]):
            raise ValueError("Current offline trainer has no passing GPU mechanism diagnostic")
        signal.signal(signal.SIGTERM, self.interrupt)
        signal.signal(signal.SIGINT, self.interrupt)
        atomic_json(WORK / "offline_pipeline_pid.json", {
            "pid": os.getpid(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "adopted_sft_child": self.adopted_child,
        })

    def interrupt(self, signum, frame):
        # The adopted SFT is deliberately independent of this supervisor.
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait()
        self.status("interrupted", signal=signum)
        raise SystemExit(128 + signum)

    def check_contract(self):
        for name, expected in self.contract["code_sha256"].items():
            if digest(CODE / name) != expected:
                raise RuntimeError(f"Offline pipeline source changed during the run: {name}")

    def status(self, state, **detail):
        fields = {
            "state": state, "phase": self.phase, "pipeline_pid": os.getpid(),
            "child_pid": self.child.pid if self.child is not None and self.child.poll() is None else None,
            "physical_gpu": 4, "training_mode": "offline_latent_rl",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_seconds_this_invocation": time.monotonic() - self.started,
            "online_training_enabled": False,
        }
        if self.phase.startswith("sft"):
            log = WORK / "sft_full/train.jsonl"
            if log.exists():
                import json
                with log.open("rb") as handle:
                    handle.seek(0, 2)
                    end = handle.tell()
                    handle.seek(max(0, end - 12000))
                    lines = handle.read().decode("utf-8", "replace").splitlines()
                rows = []
                for line in lines:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                if rows:
                    last = rows[-1]
                    speed = None
                    if len(rows) >= 2 and last["step"] > rows[0]["step"]:
                        dt = last["seconds_since_start"] - rows[0]["seconds_since_start"]
                        if dt > 0:
                            speed = dt / (last["step"] - rows[0]["step"])
                    fields.update(sft_step=last["step"], sft_target=self.plan["sft_target_steps"],
                                  sft_next_position=last["next_position"])
                    if speed is not None:
                        fields.update(seconds_per_step_recent=speed,
                                      sft_eta_seconds=(self.plan["sft_target_steps"] - last["step"]) * speed)
            if self.adopted_child and process_identity(self.adopted_child["pid"]) == self.adopted_child:
                fields["child_pid"] = self.adopted_child["pid"]
        if self.phase == "offline_rl" and (WORK / "offline_rl/progress.json").exists():
            fields["offline_rl_progress"] = read_json(WORK / "offline_rl/progress.json")
        if self.phase.startswith("judge_"):
            path = WORK / f"test_eval_{self.phase[6:]}/progress.json"
            if path.exists():
                fields["judge_progress"] = read_json(path)
        atomic_json(WORK / "pipeline_status.json", {**fields, **detail})

    def execute(self, name, command, artifacts, validator=None):
        self.phase = name
        self.check_contract()
        marker = WORK / "offline_stages" / f"{name}.json"
        if marker.exists():
            record = read_json(marker)
            if record["contract_hash"] != object_hash(self.contract):
                raise ValueError(f"Offline stage contract differs at {name}")
            for path, expected in record["artifacts"].items():
                if digest(Path(path)) != expected:
                    raise ValueError(f"Offline stage artifact changed: {path}")
            if validator:
                validator()
            return
        recovered = False
        if artifacts and all(p.is_file() for p in artifacts) and validator:
            try:
                validator()
            except (ValueError, RuntimeError, FileNotFoundError):
                pass
            else:
                recovered = True
        start = time.monotonic()
        if not recovered:
            logs = WORK / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            with (logs / f"{name}.log").open("a", encoding="utf-8") as handle:
                handle.write(f"\nSTART {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {command!r}\n")
                handle.flush()
                self.child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                                              stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                while True:
                    self.status("running", command=command, log=str(logs / f"{name}.log"))
                    try:
                        code = self.child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        continue
                    break
                self.child = None
                if code:
                    raise RuntimeError(f"Stage {name} exited {code}; see {logs / (name + '.log')}")
            if validator:
                validator()
        if any(not p.is_file() for p in artifacts):
            raise RuntimeError(f"Stage {name} omitted its committed artifacts")
        atomic_json(marker, {
            "contract_hash": object_hash(self.contract), "command": command,
            "artifacts": {str(p): digest(p) for p in artifacts}, "recovered": recovered,
            "seconds_this_invocation": time.monotonic() - start,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        self.status("running", completed_stage=name)

    def core(self, name, args, artifacts, validator=None):
        self.execute(name, [PYTHON, "-u", str(CODE / "run_main_experiment.py"), "--config", str(CONFIG)]
                     + [str(value) for value in args], artifacts, validator)

    def checkpoint(self, directory, stage, step=None):
        from eclp_elpo import verify_checkpoint
        last = read_json(directory / "last.json")
        path = Path(last["checkpoint"])
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Checkpoint path leaves its experiment directory")
        meta = verify_checkpoint(path, read_json(WORK / "protocol.lock.json"))
        if meta["checkpoint_id"] != last["checkpoint_id"] or meta["stage"] != stage:
            raise ValueError("Wrong checkpoint identity or stage")
        if step is not None and meta["step"] != step:
            raise ValueError(f"Expected checkpoint step {step}, found {meta['step']}")
        return path

    def adopt_sft(self):
        self.phase = "sft_adopted"
        self.status("running", handoff="adopt_current_SFT_without_reset; online_RL_disabled")
        if self.adopted_child:
            while process_identity(self.adopted_child["pid"]) == self.adopted_child:
                self.check_contract()
                self.status("running", handoff="original_SFT_child_running; online_RL_disabled")
                time.sleep(20)
            self.adopted_child = None
        sft = WORK / "sft_full"
        checkpoint = self.checkpoint(sft, "sft")
        step = read_json(checkpoint / "checkpoint.json")["step"]
        if step < self.plan["sft_target_steps"]:
            legacy = read_json(CODE / "training_plan.json")
            self.core("sft_resume_offline_handoff", [
                "sft", "--prepared", PREPARED, "--output", sft,
                "--epochs", legacy["sft_epochs"], "--accumulation", legacy["sft_accumulation"],
                "--lr", legacy["sft_learning_rate"], "--max-steps", legacy["sft_target_steps"],
                "--save-every", legacy["sft_save_every"], "--checkpoint", checkpoint, "--resume",
            ], [sft / "last.json"], lambda: self.checkpoint(sft, "sft", self.plan["sft_target_steps"]))
        return self.checkpoint(sft, "sft", self.plan["sft_target_steps"])

    def run(self):
        checkpoint = self.adopt_sft()
        from eclp_elpo import Config, verified_prepared, verify_lock
        cfg = Config.load(CONFIG)
        lock = verify_lock(cfg, WORK / "protocol.lock.json")
        data = verified_prepared(PREPARED, lock)
        if (data["sft_sessions"] != self.plan["expected_sft_sessions"]
                or data["partitions"] != {"rl_train": 1800, "dev": 200, "test": 1000}):
            raise ValueError("Original corpus or fixed partitions changed")
        output = WORK / "offline_rl"
        def valid_rl():
            meta = read_json(self.checkpoint(output, "rl", self.plan["expected_rl_scenarios"] * self.plan["rl_epochs"]) / "checkpoint.json")
            summary = read_json(output / "summary.json")
            if (summary.get("complete") is not True or summary.get("unique_scenarios") != 1800
                    or summary.get("checkpoint_id") != meta["checkpoint_id"]
                    or summary.get("policy_changed") is not True or summary.get("executor_unchanged") is not True
                    or any(summary.get(key) != 0 for key in ["new_persuadee_calls", "judge_calls", "generated_training_tokens"])):
                raise ValueError("Offline RL completion/provenance gate failed")
        common = [PYTHON, "-u", str(CODE / "offline_latent_rl.py"), "--config", str(CONFIG), "--plan", str(PLAN)]
        self.execute("offline_rl", common + ["train", "--prepared", str(PREPARED),
                                           "--checkpoint", str(checkpoint), "--output", str(output)],
                     [output / "summary.json", output / "last.json"], valid_rl)
        checkpoint = self.checkpoint(output, "rl", self.plan["expected_rl_scenarios"] * self.plan["rl_epochs"])
        validation = WORK / "offline_dev_validation.json"
        def dev_gate():
            value = read_json(validation)
            if (value.get("complete") is not True or value.get("n") != 200
                    or value.get("checkpoint_id") != read_json(checkpoint / "checkpoint.json")["checkpoint_id"]
                    or value.get("proxy_quality_gate_pass") is not True):
                raise ValueError("Offline dev likelihood materially worsened; diagnose before test generation")
        self.execute("offline_dev_validation", common + ["validate", "--prepared", str(PREPARED),
                                                        "--checkpoint", str(checkpoint), "--output", str(validation)],
                     [validation], dev_gate)
        immutable_json(WORK / "final_checkpoint.json", {
            "checkpoint": str(checkpoint), "metadata": read_json(checkpoint / "checkpoint.json"),
            "rule": self.plan["final_checkpoint_rule"], "selected_using_test": False,
            "pipeline_contract_hash": object_hash(self.contract), "training": "offline_continuous_action_RL",
            "reward": self.plan["reward"], "correction_weight": self.plan["correction_weight"],
            "training_persuadee_calls": 0, "training_judge_calls": 0,
        })
        test = WORK / "test_eclp_offline_lpo"
        from evaluate_our import METHOD, verify_generation
        self.core("offline_test_generation", [
            "generate", "--prepared", PREPARED, "--checkpoint", checkpoint, "--split", "test",
            "--limit", 0, "--allow-test", "--method", self.plan["test_method"], "--output", test,
        ], [test / "generation.jsonl", test / "generation_summary.json"], lambda: verify_generation(test, WORK))
        evaluations = {}
        from update_our_table import verify_summary, update_table
        for judge in self.plan["final_judges"]:
            directory = WORK / f"test_eval_{judge}"
            self.execute(f"judge_{judge}", [PYTHON, "-u", str(CODE / "evaluate_our.py"),
                                           "--judge", judge, "--generation", str(test),
                                           "--output", str(directory), "--work", str(WORK)],
                         [directory / "summary.json", directory / f"{METHOD}.jsonl"],
                         lambda directory=directory, judge=judge: verify_summary(directory, test, WORK, judge))
            evaluations[judge] = directory
            self.execute(f"table_{judge}", [PYTHON, str(CODE / "update_our_table.py"),
                                           "--generation", str(test), "--work", str(WORK),
                                           "--selene" if judge == "selene8b" else "--qwen3", str(directory)], [])
        update_table(WORK, test, evaluations, ROOT / "exp_code_ljj/results/main_experiment_table.md")
        self.phase = "complete"
        self.status("complete", final_checkpoint=str(checkpoint), test_scenarios=1000,
                    table=str(ROOT / "exp_code_ljj/results/main_experiment_table.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adopt-from", type=Path,
                        default=WORK / "offline_handoff.json" if (WORK / "offline_handoff.json").is_file() else None)
    args = parser.parse_args()
    with writer_lock(WORK / ".offline_pipeline.lock"):
        pipeline = OfflinePipeline(args.adopt_from)
        try:
            pipeline.run()
        except Exception as error:
            pipeline.status("failed", error=f"{type(error).__name__}: {error}",
                            restart_command=str(CODE / "run_gpu4_full_chain.sh"))
            raise


if __name__ == "__main__":
    main()
