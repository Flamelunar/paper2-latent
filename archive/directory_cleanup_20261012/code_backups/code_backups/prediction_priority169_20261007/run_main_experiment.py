#!/usr/bin/env python3
"""Emit a full staged experiment plan; execute only with explicit --execute."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any
from evaluation.revision_config import RevisionConfig
from utils.artifacts import write_json, hash_object

LOCAL_ROOT = Path(__file__).resolve().parent


def scoped_repeat_config(cfg: dict[str, Any], repeat_ids: list[int]) -> dict[str, Any]:
    """Return an evaluation config limited to selected, predeclared rollout seeds."""
    available = cfg["rollout_seed_ids"]
    if not repeat_ids or any(type(value) is not int for value in repeat_ids):
        raise ValueError("repeat_ids must contain one or more integer seed IDs")
    if len(set(repeat_ids)) != len(repeat_ids):
        raise ValueError("repeat_ids must be unique")
    if any(value not in available for value in repeat_ids):
        raise ValueError(f"repeat_ids must be selected from the frozen IDs: {available}")
    scoped = deepcopy(cfg)
    scoped["rollout_seed_ids"] = list(repeat_ids)
    return scoped


def schedule(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    cells = []
    pmm_runs = cfg.get("pmm_runs_root", cfg["runs_root"])
    for repeat in cfg["rollout_seed_ids"]:
        for method in cfg["methods"]:
            for backbone, base in list(cfg["backbones"].items()):
                run = Path(cfg["runs_root"]) / backbone
                pmm_run = Path(pmm_runs) / backbone
                checkpoint = {"zero_shot": base, "ma2p": base,
                    "sft": str(run / "structured_strategy_merged"),
                    "grpo_only": str(run / "grpo_only_merged"),
                    "sft_grpo": str(run / "sft_grpo_merged"),
                    "ours": str(pmm_run / "strategy_compact_merged")}[method]
                generation = str(Path(cfg["outputs_root"]) / "generation" / backbone / method / f"repeat-{repeat}")
                judges = {j: str(Path(cfg["outputs_root"]) / "judges" / j / backbone / method / f"repeat-{repeat}") for j in cfg["judges"]}
                cells.append({"method": method, "backbone": backbone, "repeat_id": repeat, "checkpoint": checkpoint,
                              "generation_dir": generation, "judge_dirs": judges, "shared": method == "zero_shot"})
    return cells


def build_plan(cfg: dict[str, Any], config_path: Path, plan_path: Path = Path("artifacts/main_plan.json")) -> dict[str, Any]:
    py, project = cfg["python"], cfg["project_root"]
    jobs = []
    def add(phase: str, name: str, argv: list[str], cuda: str | None = None) -> None:
        jobs.append({"phase": phase, "name": name, "argv": argv, "cwd": project,
                     "env": {"CUDA_VISIBLE_DEVICES": cuda} if cuda is not None else {}})
    add("preflight", "environment-and-data", [py, "scripts/preflight.py", "--config", str(config_path)])
    add("prepare", "adapt-exact-recorded-strategies", [py, "utils/prepare_revision_data.py", "--raw-source", cfg["data"]["raw_source"],
        "--processed-dir", cfg["data"]["processed_dir"], "--eval-source", cfg["data"].get("training_exclusion_eval", cfg["data"]["eval"]), "--output-dir", cfg["data"]["revision_dir"]])
    tr = cfg["training"]
    policy_path = Path(project) / 'configs/grpo_training_policy.json'
    grpo_policy = json.loads(policy_path.read_text(encoding='utf-8')) if policy_path.exists() else {}
    grpo_train_file = cfg['data'].get('grpo_train_file', grpo_policy.get('train_file',
        f"{cfg['data']['revision_dir']}/structured_strategy_train.jsonl"))
    grpo_subset_manifest = cfg['data'].get('grpo_subset_manifest', grpo_policy.get('subset_manifest'))
    grpo_output_suffix = tr.get('grpo_output_suffix', grpo_policy.get('output_suffix', ''))
    grpo_jobs = []
    for backbone, base in cfg["backbones"].items():
        run = Path(cfg["runs_root"]) / backbone
        name = "structured_strategy"
        argv = [py, "train_sft.py", "--model", base, "--task", name,
            "--train-file", f"{cfg['data']['revision_dir']}/structured_strategy_train.jsonl",
            "--validation-file", f"{cfg['data']['revision_dir']}/structured_strategy_validation.jsonl",
            "--output-dir", str(run / f"{name}_adapter"), "--merged-output", str(run / f"{name}_merged"),
            "--epochs", str(tr["sft_epochs"]), "--learning-rate", str(tr["sft_learning_rate"]),
            "--batch-size", str(tr["sft_batch_size"]), "--grad-accum", str(tr["sft_gradient_accumulation"]),
            "--max-length", str(tr["max_length"]), "--lora-rank", str(tr["lora_rank"]),
            "--lora-alpha", str(tr["lora_alpha"]), "--lora-dropout", str(tr["lora_dropout"]),
            "--seed", str(tr["seed"]), "--precision", cfg["inference"]["precision"]]
        add("train", f"{backbone}-{name}-sft", argv, tr["cuda_visible_devices"])
        for variant, init in (("grpo_only", base), ("sft_grpo", str(run / "structured_strategy_merged"))):
            argv = [py, "train_standard_grpo.py", "--init-model", init,
                "--init-mode", "base" if variant == "grpo_only" else "sft",
                "--train-file", grpo_train_file,
                "--output-dir", str(run / f"{variant}_adapter{grpo_output_suffix}"), "--merged-output", str(run / f"{variant}_merged"),
                "--learning-rate", str(tr["grpo_learning_rate"]),
                "--batch-size", str(tr["grpo_batch_size"]), "--grad-accum", str(tr["grpo_gradient_accumulation"]),
                "--num-generations", str(tr["grpo_num_generations"]), "--max-completion-length", "512",
                "--seed", str(tr["seed"]), "--lora-rank", str(tr["lora_rank"]),
                "--lora-alpha", str(tr["lora_alpha"]), "--lora-dropout", str(tr["lora_dropout"]),
                "--precision", cfg["inference"]["precision"]]
            grpo_units = tr.get("grpo_units", "steps")
            if grpo_units == "epochs":
                argv += ["--epochs", str(tr["grpo_epochs"])]
            elif grpo_units == "steps":
                argv += ["--max-steps", str(tr["grpo_steps"])]
            else:
                raise ValueError("training.grpo_units must be 'epochs' or 'steps'")
            grpo_jobs.append((f"{backbone}-{variant}-grpo", argv))
    for name, argv in grpo_jobs:
        add("train", name, argv, tr["cuda_visible_devices"])
    pmm_runs = Path(cfg.get("pmm_runs_root", cfg["runs_root"]))
    for backbone in cfg["backbones"]:
        run = Path(cfg["runs_root"]) / backbone
        pmm_run = pmm_runs / backbone
        argv = [py, "train_strategy_pmm_grpo.py", "--task", "progressive_strategy_wm",
            "--sft-merged", str(run / "structured_strategy_merged"),
            "--train-file", grpo_train_file,
            "--output-dir", str(pmm_run / f"strategy_compact_adapter{grpo_output_suffix}"),
            "--merged-output", str(pmm_run / "strategy_compact_merged"),
            "--batch-size", str(tr["grpo_batch_size"]),
            "--grad-accum", str(tr["grpo_gradient_accumulation"]),
            "--learning-rate", str(tr["grpo_learning_rate"]),
            "--num-generations", str(tr["grpo_num_generations"]),
            "--max-strategy-tokens", "512", "--seed", str(tr["seed"]),
            "--lora-rank", str(tr["lora_rank"]), "--lora-alpha", str(tr["lora_alpha"]),
            "--lora-dropout", str(tr["lora_dropout"]), "--precision", cfg["inference"]["precision"]]
        pmm_units = tr.get("pmm_grpo_units", grpo_policy.get("pmm_units", "epochs"))
        if pmm_units == "epochs":
            argv += ["--epochs", str(tr.get("pmm_grpo_epochs", grpo_policy.get("pmm_epochs", 1)))]
        elif pmm_units == "steps":
            argv += ["--max-steps", str(tr["pmm_grpo_steps"])]
        else:
            raise ValueError("training.pmm_grpo_units must be epochs or steps")
        if grpo_subset_manifest:
            argv += ['--train-subset-manifest', grpo_subset_manifest]
        add("train", f"{backbone}-progressive-pmm", argv, tr["cuda_visible_devices"])
    manifest_args = [py, "make_model_manifest.py", "--output", cfg["model_manifest"]]
    for role, spec in cfg["roles"].items():
        if "path" in spec:
            manifest_args += ["--model", f"{role}={spec['path']}"]
    for backbone, base in cfg["backbones"].items():
        manifest_args += ["--model", f"{backbone}_base={base}"]
        for name in ("structured_strategy", "grpo_only", "sft_grpo"):
            manifest_args += ["--model", f"{backbone}_{name}={cfg['runs_root']}/{backbone}/{name}_merged"]
        pmm_runs = cfg.get("pmm_runs_root", cfg["runs_root"])
        manifest_args += ["--model", f"{backbone}_compact={pmm_runs}/{backbone}/strategy_compact_merged"]
    add("freeze", "model-identities", manifest_args)
    cells = schedule(cfg)
    for cell in cells:
        rc = RevisionConfig.from_experiment(cfg)
        rc.method, rc.world_model, rc.backbone, rc.repeat_id = cell["method"], cell["checkpoint"], cell["backbone"], cell["repeat_id"]
        rc.training_seed = None if rc.method in {"zero_shot", "ma2p"} else tr["seed"]
        rc.wm_analysis_budget = 0 if rc.method == "ours" else 512
        args = [py, "evaluate.py", "--config", str(config_path), "--input", cfg["data"]["eval"],
            "--output-dir", cell["generation_dir"], "--method", rc.method, "--world-model", rc.world_model,
            "--backbone", rc.backbone, "--repeat-id", str(rc.repeat_id),
            "--wm-analysis-budget", str(rc.wm_analysis_budget)]
        add("generate", f"{rc.backbone}-{rc.method}-r{rc.repeat_id}", args, cfg["generation_cuda_visible_devices"])
        for judge, spec in cfg["judges"].items():
            args = [py, "judge_dialogues.py", "--input", cell["generation_dir"] + "/results.jsonl",
                "--output-dir", cell["judge_dirs"][judge], "--judge-id", judge, "--provider", spec["provider"],
                "--model", spec["model"], "--max-output-tokens", str(spec["max_output_tokens"]),
                "--max-input-tokens", str(cfg["inference"]["max_input_tokens"]),
                "--expected-items", str(cfg["data"]["expected_eval_items"])]
            if spec["provider"] == "local":
                args += ["--device", spec.get("device", "auto"), "--precision", cfg["inference"]["precision"]]
                if cfg.get("shared_local_endpoint"):
                    args += ["--shared-local-endpoint", cfg["shared_local_endpoint"]]
                cuda = tr["cuda_visible_devices"]
            else:
                # Judge resolves endpoint/credential environment variables at
                # execution. Printed commands work without shell interpolation.
                args += ["--base-url-env", spec["base_url_env"],
                         "--api-key-env", spec["api_key_env"], "--api-protocol", spec["api_protocol"],
                         "--api-token-parameter", spec["api_token_parameter"]]
                if spec.get("base_url"):
                    args += ["--base-url", spec["base_url"]]
                if spec.get("send_sampling"):
                    args.append("--api-send-sampling")
                if spec.get("send_seed"):
                    args.append("--api-send-seed")
                cuda = None
            add("judge", f"{judge}-{rc.backbone}-{rc.method}-r{rc.repeat_id}", args, cuda)
    add("aggregate", "paired-main-table", [py, "aggregate_experiments.py", "--plan",
        str(cfg.get("plan_path", plan_path)),
        "--eval-source", cfg["data"]["eval"], "--output-dir", cfg["outputs_root"] + "/tables",
        "--bootstrap-replicates", str(cfg["bootstrap_replicates"])])
    return {"protocol_version": cfg["protocol_version"], "config_hash": hash_object(cfg), "configuration": cfg,
        "cells": cells, "jobs": jobs, "counts": {"unique_method_backbone_cells": len({(c['backbone'], c['method']) for c in cells}),
        "display_rows_with_shared_zero_shot": len(cfg['methods']) * len(cfg['backbones']), "generation_jobs": len(cells),
        "planned_episodes": len(cells) * cfg["data"]["expected_eval_items"],
        "judge_jobs": len(cells) * len(cfg["judges"]),
        "sft_jobs": sum(j['name'].endswith('-sft') for j in jobs),
        "grpo_jobs": sum(j['name'].endswith('-grpo') for j in jobs),
        "progressive_pmm_jobs": sum(j['name'].endswith('-progressive-pmm') for j in jobs)},
        "execution_status": "NOT_RUN"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=LOCAL_ROOT / "configs/main_experiment.json")
    p.add_argument("--plan-output", type=Path, default=Path("artifacts/main_plan.json"))
    p.add_argument("--phase", choices=("all", "preflight", "prepare", "train", "freeze", "generate", "judge", "aggregate"), default="all")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--job", default=None, help="Exact job name to run or display")
    a = p.parse_args()
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    remote_config = Path(cfg["project_root"]) / "configs" / a.config.name
    plan = build_plan(cfg, remote_config, a.plan_output)
    write_json(a.plan_output, plan)
    jobs = [j for j in plan["jobs"] if (a.phase == "all" or j["phase"] == a.phase) and (not a.job or j["name"] == a.job)]
    if not jobs:
        raise SystemExit("No matching jobs")
    if not a.execute:
        print(json.dumps({"status": "PLAN_ONLY", "plan": str(a.plan_output), "counts": plan["counts"]}, ensure_ascii=False))
        for j in jobs:
            env = " ".join(f"{k}={shlex.quote(v)}" for k, v in j["env"].items())
            print(f"[{j['phase']}] {j['name']}: {env} {shlex.join(j['argv'])}")
        return
    for j in jobs:
        argv = list(j["argv"])
        env = dict(os.environ)
        env.update(j["env"])
        print(f"Executing {j['name']}", flush=True)
        subprocess.run(argv, cwd=j["cwd"], env=env, check=True)


if __name__ == "__main__":
    main()
