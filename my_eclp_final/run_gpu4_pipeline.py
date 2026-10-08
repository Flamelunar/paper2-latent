#!/usr/bin/env python3
"""Persistent, sequential GPU4 run: repaired SFT -> ELPO -> test -> Judge table."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiment_io import (append_jsonl, atomic_json, digest, immutable_json,
                           object_hash, read_json, read_jsonl, writer_lock)

ROOT = Path(__file__).resolve().parents[1]
CODE = Path(__file__).resolve().parent
PYTHON = "/app/hzj/anaconda3/envs/ljj/bin/python"
CONFIG = CODE / "config_gpu4.json"
WORK = ROOT / "eclp_runs_gpu4"
PREPARED = WORK / "prepared"


class Pipeline:
    def __init__(self):
        self.plan = read_json(CODE / "training_plan.json")
        self.child = None
        self.phase = "starting"
        self.started = time.time()
        os.environ.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="4",
                          PYTHONUNBUFFERED="1", DAILYPERSUASION_PROFILE="paper10k",
                          ALLOW_JUDGE_EVAL="1", TOKENIZERS_PARALLELISM="false")
        os.chdir(ROOT)
        files = ["eclp_elpo.py", "check_eclp_mechanism.py", "run_main_experiment.py",
                 "config_gpu4.json", "training_plan.json", "experiment_io.py",
                 "run_gpu4_pipeline.py", "evaluate_our.py", "update_our_table.py",
                 "final_eval_protocol.py", "final_judge_prompts.json"]
        self.contract = {"plan": self.plan, "python": PYTHON,
                         "code_sha256": {name: digest(CODE / name) for name in files},
                         "physical_gpu": 4, "visible_devices": "4"}
        immutable_json(WORK / "pipeline_contract.json", self.contract)
        signal.signal(signal.SIGTERM, self.interrupt)
        signal.signal(signal.SIGINT, self.interrupt)

    def interrupt(self, signum, frame):
        self.status("interrupted", signal=signum)
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait()
        raise SystemExit(128 + signum)

    def check_contract(self):
        for name, expected in self.contract["code_sha256"].items():
            if digest(CODE / name) != expected:
                raise RuntimeError(f"Pipeline source changed during the run: {name}")

    def progress(self) -> dict:
        path = WORK / "sft_full/train.jsonl"
        result = {}
        if self.phase.startswith("sft") and path.exists():
            # Logs are only a few MB; use complete newline-terminated records
            # while the child may be writing the final record.
            import json
            lines = path.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines):
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue
                result = {"sft_step": last["step"], "sft_target": self.plan["sft_target_steps"],
                          "sft_next_position": last["next_position"],
                          "last_losses": {key: last[key] for key in ("utterance", "observation", "strategy", "reasoning")},
                          "last_grad_norm": last["grad_norm"]}
                if last["step"] >= 2:
                    seconds = last["seconds_since_start"]
                    # A resumed invocation's timer restarts. Do not report a
                    # false total-step ETA from its local time.
                    if self.phase == "sft_full":
                        local_steps = last["step"] - self.sft_resume_step
                        if local_steps > 0:
                            result["seconds_per_step_this_invocation"] = seconds / local_steps
                            result["sft_eta_seconds"] = ((self.plan["sft_target_steps"] - last["step"])
                                                         * seconds / local_steps)
                break
        if self.phase.startswith("judge_"):
            judge = self.phase[len("judge_"):]
            path = WORK / f"test_eval_{judge}/progress.json"
            if path.exists():
                result["judge_progress"] = read_json(path)
        return result

    def status(self, state: str, **detail):
        atomic_json(WORK / "pipeline_status.json", {
            "state": state, "phase": self.phase, "pipeline_pid": os.getpid(),
            "child_pid": self.child.pid if self.child is not None and self.child.poll() is None else None,
            "physical_gpu": 4, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_seconds_this_invocation": time.time() - self.started,
            **self.progress(), **detail})

    def execute(self, name: str, command: list[str], artifacts: list[Path], validate=None):
        self.phase = name
        self.check_contract()
        marker = WORK / "stages" / f"{name}.json"
        if marker.exists():
            saved = read_json(marker)
            if saved.get("pipeline_contract_hash") != object_hash(self.contract):
                raise ValueError(f"Stage contract changed at {name}")
            for path, expected in saved["artifacts"].items():
                if digest(Path(path)) != expected:
                    raise ValueError(f"Completed stage artifact changed: {path}")
            if validate is not None:
                validate()
            print(f"SKIP complete stage {name}", flush=True)
            return
        self.status("running", command=command)
        start = time.time()
        # Artifact creation and the marker commit may be separated by a crash.
        # Recover a completed stage only after its semantic validator succeeds.
        recovered = False
        if artifacts and all(path.is_file() for path in artifacts) and validate is not None:
            try:
                validate()
            except (ValueError, RuntimeError, FileNotFoundError):
                # An earlier stage can legitimately leave last.json behind
                # before a later stage reaches its own completion criterion.
                recovered = False
            else:
                recovered = True
        if not recovered:
            logs = WORK / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            with (logs / f"{name}.log").open("a", encoding="utf-8") as handle:
                handle.write(f"\nSTART {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {command!r}\n")
                handle.flush()
                self.child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=handle,
                                              stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                self.status("running", command=command, log=str(logs / f"{name}.log"))
                while True:
                    try:
                        code = self.child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        self.status("running", command=command, log=str(logs / f"{name}.log"))
                    else:
                        break
                self.child = None
                if code != 0:
                    raise RuntimeError(f"Stage {name} failed with exit {code}; see {logs / (name + '.log')}")
            if validate is not None:
                validate()
        if any(not path.is_file() for path in artifacts):
            raise RuntimeError(f"Stage {name} exited without complete committed artifacts")
        record = {"stage": name, "complete": True, "pipeline_contract_hash": object_hash(self.contract),
                  "command": command, "seconds_this_invocation": time.time() - start,
                  "recovered_completed_artifacts": recovered,
                  "artifacts": {str(path): digest(path) for path in artifacts},
                  "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        atomic_json(marker, record)
        append_jsonl(WORK / "stage_events.jsonl", record)
        self.status("running", completed_stage=name)
        print(f"DONE {name} ({record['seconds_this_invocation']:.1f}s)", flush=True)

    def core(self, name: str, args: list, artifacts: list[Path], validate=None):
        command = [PYTHON, "-u", str(CODE / "run_main_experiment.py"), "--config", str(CONFIG)]
        self.execute(name, command + [str(a) for a in args], artifacts, validate)

    def checkpoint(self, directory: Path, stage: str, exact_step: int | None = None) -> Path:
        from eclp_elpo import verify_checkpoint
        last = read_json(directory / "last.json")
        checkpoint = Path(last["checkpoint"])
        if not checkpoint.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Checkpoint points outside the current experiment")
        meta = verify_checkpoint(checkpoint, read_json(WORK / "protocol.lock.json"))
        if meta["checkpoint_id"] != last["checkpoint_id"] or meta["stage"] != stage:
            raise ValueError("Wrong checkpoint stage or identity")
        if exact_step is not None and meta["step"] != exact_step:
            raise ValueError(f"Expected committed step {exact_step}, found {meta['step']}")
        return checkpoint

    def fresh_output(self, stage: str, complete_file: str) -> Path:
        """Retain interrupted short fits/updates; restart in a new attempt dir."""
        marker = WORK / "stages" / f"{stage}.json"
        if marker.exists():
            files = read_json(marker)["artifacts"]
            return next(Path(path).parent for path in files if Path(path).name == complete_file)
        root = WORK / f"{stage}_attempts"
        attempts = sorted(root.glob("attempt-*")) if root.exists() else []
        if attempts and (attempts[-1] / complete_file).is_file():
            return attempts[-1]
        return root / f"attempt-{len(attempts) + 1:03d}"

    def run(self, until: str | None = None):
        from eclp_elpo import Config, verify_lock, verified_prepared, verify_collection
        cfg = Config.load(CONFIG)
        cfg.validate()
        if (cfg.base_model != "/app/hzj/models/Qwen2.5-7B-Instruct"
                or cfg.work != WORK or cfg.root != ROOT
                or any(getattr(cfg, k) != "cuda:0" for k in ("actor_device", "persuadee_device", "judge_device"))):
            raise ValueError("GPU4 actor/work/config contract differs from the user's request")
        p = self.plan
        self.core("audit", ["audit", "--output", WORK / "source_audit.json"], [WORK / "source_audit.json"],
                  lambda: self.require(not read_json(WORK / "source_audit.json")["blockers"], "Source audit blockers"))
        self.core("lock", ["lock"], [WORK / "protocol.lock.json"],
                  lambda: verify_lock(cfg, WORK / "protocol.lock.json"))
        lock = read_json(WORK / "protocol.lock.json")
        self.core("prepare", ["prepare", "--output", PREPARED], [PREPARED / "manifest.json"],
                  lambda: verified_prepared(PREPARED, lock))
        data = read_json(PREPARED / "manifest.json")
        self.require(data["sft_sessions"] == p["expected_sft_sessions"]
                     and data["sft_scenarios"] == p["expected_sft_scenarios"]
                     and data["partitions"] == {"rl_train": 1800, "dev": 200, "test": 1000}, "Original split changed")
        self.require(math.ceil(data["sft_sessions"] / p["sft_accumulation"]) * p["sft_epochs"] == p["sft_target_steps"],
                     "SFT step target differs from the full original corpus")
        self.core("smoke", ["smoke", "--prepared", PREPARED, "--output", WORK / "gpu_smoke.json"], [WORK / "gpu_smoke.json"],
                  lambda: self.require(read_json(WORK / "gpu_smoke.json").get("status") == "PASS_GPU_SMOKE_NOT_AN_EXPERIMENT_RESULT", "GPU smoke did not pass"))
        if until == "smoke":
            self.status("stopped_at_requested_stage", stage="smoke")
            return
        sft = WORK / "sft_full"
        # Save the first real optimizer update promptly, then continue from its
        # exact optimizer/RNG/data cursor. There is no extra epoch or reset.
        sft_common = ["sft", "--prepared", PREPARED, "--output", sft, "--epochs", p["sft_epochs"],
                      "--accumulation", p["sft_accumulation"], "--lr", p["sft_learning_rate"]]
        bootstrap_marker = WORK / "stages/sft_bootstrap.json"
        if not bootstrap_marker.exists():
            self.core("sft_bootstrap", sft_common + ["--max-steps", 1, "--save-every", 1],
                      [sft / "checkpoint-000001/checkpoint.json"],
                      lambda: self.checkpoint(sft, "sft", 1))
        # last.json changes during subsequent SFT; the bootstrap artifact is a
        # historical cursor and is deliberately not replayed as a final check.
        checkpoint = self.checkpoint(sft, "sft")
        meta = read_json(checkpoint / "checkpoint.json")
        self.sft_resume_step = meta["step"]
        self.core("sft_full", sft_common + ["--max-steps", p["sft_target_steps"], "--save-every", p["sft_save_every"],
                                            "--checkpoint", checkpoint, "--resume"],
                  [sft / "last.json"], lambda: self.checkpoint(sft, "sft", p["sft_target_steps"]))
        checkpoint = self.checkpoint(sft, "sft", p["sft_target_steps"])
        if until == "sft":
            self.status("stopped_at_requested_stage", stage="sft")
            return
        self.dev_stages("sft", checkpoint)
        probe = WORK / "probe_isotropic"
        self.collect_stage("probe_isotropic", checkpoint, probe, p["probe_offset"], p["probe_contexts"], p["probe_horizon"])
        effect = self.fresh_output("effect_fit", "effect.json")
        def valid_effect():
            source_run = read_json(probe / "run.json")
            meta = read_json(effect / "effect.json")
            self.require(meta.get("beats_zero_predictor") is True
                         and meta.get("lock_id") == lock["lock_id"]
                         and meta.get("executor_hash") == source_run["executor_hash"]
                         and meta.get("sha256") == digest(effect / "effect.pt")
                         and meta.get("effect_id") == object_hash({k: v for k, v in meta.items() if k != "effect_id"}),
                         "Effect model failed its fit or provenance gate")
        self.core("effect_fit", ["fit-effect", "--rollouts", probe, "--output", effect,
                                 "--epochs", p["effect_epochs"], "--batch-size", p["effect_batch_size"],
                                 "--lr", p["effect_learning_rate"]],
                  [effect / "effect.json", effect / "effect.pt"], valid_effect)
        holdout = WORK / "probe_holdout"
        self.collect_stage("probe_holdout", checkpoint, holdout, p["holdout_offset"], p["holdout_contexts"], p["probe_horizon"])
        held = WORK / "effect_holdout.json"
        def holdout_gate():
            value = read_json(held)
            self.require(value.get("scenarios_disjoint_from_fitting_and_selection") is True
                         and value.get("scenario_count") == p["holdout_contexts"]
                         and value.get("relative_mse_gain") is not None
                         and value["relative_mse_gain"] > 0,
                         "Effect failed the independent holdout zero-predictor gate; diagnose before ELPO, no silent fallback")
        self.core("effect_holdout", ["check-effect", "--rollouts", holdout, "--effect", effect, "--output", held], [held], holdout_gate)
        for number, offset in enumerate(p["elpo_offsets"]):
            tag = f"elpo_r{number}"
            rollouts = WORK / tag
            scores = WORK / f"{tag}_scores"
            self.collect_stage(f"{tag}_collect", checkpoint, rollouts, offset, p["elpo_contexts"], p["elpo_horizon"], effect)
            self.core(f"{tag}_score", ["score", "--rollouts", rollouts, "--output", scores],
                      [scores / "scores.jsonl", scores / "scoring.json"],
                      lambda scores=scores: self.valid_scores(scores))
            update = self.fresh_output(f"{tag}_update", "update.json")
            self.core(f"{tag}_update", ["update", "--rollouts", rollouts, "--scores", scores, "--checkpoint", checkpoint,
                                        "--output", update, "--epochs", p["elpo_epochs"], "--batch-size", p["elpo_batch_size"],
                                        "--lr", p["elpo_learning_rate"], "--max-unresolved-fraction", 0],
                      [update / "update.json", update / "last.json"], lambda update=update: self.checkpoint(update, "rl"))
            checkpoint = self.checkpoint(update, "rl")
        self.dev_stages("elpo_final", checkpoint)
        frozen = {"checkpoint": str(checkpoint), "metadata": read_json(checkpoint / "checkpoint.json"),
                  "rule": p["final_checkpoint_rule"], "selected_using_test": False,
                  "pipeline_contract_hash": object_hash(self.contract)}
        immutable_json(WORK / "final_checkpoint.json", frozen)
        test = WORK / "test_eclp_elpo"
        from evaluate_our import verify_generation
        self.core("test_generation", ["generate", "--prepared", PREPARED, "--checkpoint", checkpoint,
                                       "--split", "test", "--limit", 0, "--allow-test", "--method", "ECLP-ELPO", "--output", test],
                  [test / "generation.jsonl", test / "generation_summary.json"], lambda: verify_generation(test, WORK))
        evaluations = {}
        for judge in p["final_judges"]:
            output = WORK / f"test_eval_{judge}"
            self.execute(f"judge_{judge}", [PYTHON, "-u", str(CODE / "evaluate_our.py"), "--judge", judge,
                                             "--generation", str(test), "--output", str(output), "--work", str(WORK)],
                         [output / "summary.json", output / "our_eclp_elpo.jsonl"],
                         lambda output=output, judge=judge: self.valid_judgments(output, test, judge))
            evaluations[judge] = output
            # Fill each row as soon as that Judge completes, so its result is
            # available even while the second Judge is still running.
            flag = "--selene" if judge == "selene8b" else "--qwen3"
            self.execute(f"table_{judge}", [PYTHON, str(CODE / "update_our_table.py"), "--generation", str(test),
                                             flag, str(output), "--work", str(WORK)], [])
        from update_our_table import update_table
        update_table(WORK, test, evaluations, ROOT / "exp_code_ljj/results/main_experiment_table.md")
        self.phase = "complete"
        self.status("complete", final_checkpoint=str(checkpoint), test_scenarios=1000,
                    table=str(ROOT / "exp_code_ljj/results/main_experiment_table.md"))
        print("COMPLETE: repaired SFT, 3 ELPO rounds, test1000, Selene8B, Qwen3-14B, both Our table rows", flush=True)

    @staticmethod
    def require(condition: bool, message: str):
        if not condition:
            raise RuntimeError(message)

    def collect_stage(self, name, checkpoint, output, offset, limit, horizon, effect=None):
        from eclp_elpo import verify_collection
        args = ["collect", "--prepared", PREPARED, "--checkpoint", checkpoint, "--output", output,
                "--offset", offset, "--limit", limit, "--horizon", horizon,
                "--candidates", self.plan["candidates"], "--repeats", self.plan["repeats"],
                "--exploration", "effect" if effect else "isotropic", "--reward", "interaction"]
        if effect:
            args.extend(["--effect", effect])
        self.core(name, args, [output / "collection.json"],
                  lambda: verify_collection(output, read_json(WORK / "protocol.lock.json")))

    def dev_stages(self, tag: str, checkpoint: Path):
        output = WORK / f"validate_{tag}"
        self.core(f"dev_{tag}_offline", ["validate", "--prepared", PREPARED, "--checkpoint", checkpoint,
                                        "--output", output, "--limit", self.plan["dev_scenarios"]],
                  [output / "summary.json"], lambda: self.require(read_json(output / "summary.json")["n"] == 200, "DEV count mismatch"))
        generation = WORK / f"dev_{tag}_generation"
        self.core(f"dev_{tag}_generation", ["generate", "--prepared", PREPARED, "--checkpoint", checkpoint,
                                           "--split", "dev", "--limit", 200, "--method", f"ECLP-{tag}", "--output", generation],
                  [generation / "generation_summary.json"],
                  lambda: self.require(read_json(generation / "generation_summary.json")["complete"] is True
                                       and read_json(generation / "generation_summary.json")["n"] == 200, "Incomplete DEV generation"))
        evaluated = WORK / f"dev_{tag}_eval"
        self.core(f"dev_{tag}_eval", ["evaluate", "--generation", generation, "--output", evaluated],
                  [evaluated / "summary.json"], lambda: self.require(read_json(evaluated / "summary.json")["n"] == 200, "Incomplete DEV scoring"))

    def valid_scores(self, scores: Path):
        summary = read_json(scores / "scoring.json")
        self.require(summary.get("complete") is True and summary.get("unresolved") == 0
                     and summary.get("scores_sha256") == digest(scores / "scores.jsonl"),
                     "Unresolved training rewards: repair scoring before PPO")

    @staticmethod
    def valid_judgments(output: Path, generation: Path, judge: str):
        from update_our_table import verify_summary
        verify_summary(output, generation, WORK, judge)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until", choices=("smoke", "sft"))
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    with writer_lock(WORK / ".pipeline.lock"):
        pipeline = Pipeline()
        atomic_json(WORK / "pipeline_pid.json", {"pid": os.getpid(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        try:
            pipeline.run(args.until)
        except Exception as exc:
            pipeline.status("failed", error=f"{type(exc).__name__}: {exc}",
                            restart_command=str(CODE / "run_gpu4_full_chain.sh"))
            raise


if __name__ == "__main__":
    main()
