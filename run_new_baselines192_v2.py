#!/usr/bin/env python3
"""Schedule isolated 192-case unified-v2 evaluations on physical GPUs 0/1."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "my_eclp_final"))
from experiment_io import atomic_json, digest, object_hash, read_json, read_jsonl, writer_lock
from evaluate_new_baselines192_v2 import JUDGES, METHOD_LABELS, summarize, validate_record
from matched_eval_protocol_v2 import CONFIG, PROMPT_FILE

FAMILIES = {
    "meta_llama31": (0, "Meta-Llama-3.1-8B-Instruct", "llama31"),
    "deepseek_r1_qwen14b": (1, "DeepSeek-R1-Distill-Qwen-14B", "deepseek14b"),
}
METHODS = [m for m in METHOD_LABELS if m.startswith("baseline")]
EVALUATOR_NAMES = {"evaluate_new_baselines192_v2.py", "evaluate_selected192_v2.py"}
WORK_DIR = "evaluation_from_start192_fullsource_v2"


def free_gib(gpu):
    value = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    return float(value.strip()) / 1024


def pid_matches_output(pid, output):
    try:
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        return (any(Path(argument).name in EVALUATOR_NAMES for argument in arguments)
                and "--output" in arguments and str(output) in arguments)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def find_eval_pid(output):
    result = subprocess.run(["pgrep", "-f", "evaluate_.*192_v2[.]py"],
                            text=True, capture_output=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"pgrep failed while adopting evaluators: {result.stderr.strip()}")
    for line in result.stdout.splitlines():
        if line.isdigit() and pid_matches_output(int(line), output):
            return int(line)
    return None


def has_exact_indexes(path, expected):
    if not path.is_file():
        return False
    try:
        rows = read_jsonl(path)
    except (OSError, ValueError):
        return False
    indexes = [row.get("index") for row in rows]
    return (len(rows) == len(expected) and all(type(index) is int for index in indexes)
            and len(set(indexes)) == len(indexes) and set(indexes) == expected)


def preferred_generation(root, family, method):
    selected = set(read_json(root / "exp_code_ljj/data/test_selection_192.json")["original_indexes"])
    full = root / "exp_code_ljj/artifacts/generation_5rounds" / family / (method + ".jsonl")
    subset = root / "exp_code_ljj/artifacts/generation_from_start192" / family / (method + ".jsonl")
    if has_exact_indexes(full, set(range(1000))):
        return full
    if has_exact_indexes(subset, selected):
        return subset
    return None


def evaluation_outputs(work, family, judge, method):
    assigned_gpu = FAMILIES[family][0]
    base = work / family / judge / method
    other_gpu = 1 - assigned_gpu
    return [(assigned_gpu, base),
            (other_gpu, base.with_name(f"{method}_gpu{other_gpu}"))]


def best_result_output(work, family, judge, method):
    candidates = evaluation_outputs(work, family, judge, method)
    scored = []
    for order, (_, output) in enumerate(candidates):
        summary_path = output / "summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            scored.append((bool(summary.get("complete")),
                           int(summary.get("evaluated_count", 0)), -order, output))
    if scored:
        return max(scored)[-1]
    for _, output in candidates:
        if (output / "run.json").exists():
            return output
    return candidates[0][1]


def runtime_output(work, family, judge, method, minimum_gib, busy_gpus=()):
    busy_gpus = set(busy_gpus)
    candidates = evaluation_outputs(work, family, judge, method)
    for default_gpu, output in candidates:
        run_path = output / "run.json"
        if run_path.exists():
            recorded_gpu = int(read_json(run_path).get("physical_gpu", default_gpu))
            if recorded_gpu not in busy_gpus and free_gib(recorded_gpu) >= minimum_gib:
                return recorded_gpu, output
    for gpu, output in candidates:
        if gpu not in busy_gpus and free_gib(gpu) >= minimum_gib:
            return gpu, output
    return None


def render(root, work, family):
    gpu, model, table_suffix = FAMILIES[family]
    selected = read_json(root / "exp_code_ljj/data/test_selection_192.json")["original_indexes"]
    golden = {r["index"]: r for r in read_jsonl(root / "eclp_runs_gpu4/evaluation_v2/golden/golden.jsonl")}
    selected_golden = {index: golden[index] for index in selected}
    lines = ["## Main experiment: unified v2 (192 scenarios)", "",
             f"Persuader: ({'/app/hzj/models/' + model}). Persuadee: (/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged).",
             f"Prompt / calculation revision: `{CONFIG['revision']}`. Judges and generation use BF16 without quantization or thinking output.",
             "", "固定测试集 **N=192**：原始测试点 `target_turn == 1` 且 `history == []`，使用 [test_selection_192.json](../data/test_selection_192.json) 中的原始索引。完整 1000 条生成存在且通过索引检查时，直接从中抽取这 192 条；否则只使用固定 192 条生成文件。",
             "", "| Judge | Method | Persuader model | Parsed / selected N | Success within 5 rounds (%) | Win@1 (%) | Win-Full (%) | Mean rounds (failure=5) | Pending / unresolved |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for judge in JUDGES:
        for method in METHODS:
            output = best_result_output(work, family, judge, method)
            summary_path = output / "summary.json"
            values = "待评估 / 192 | - | - | - | - | 192"
            if summary_path.exists():
                summary = read_json(summary_path)
                contract = read_json(output / "run.json")
                generation = Path(contract["generation_source"])
                allowed_sources = {
                    (root / "exp_code_ljj/artifacts/generation_5rounds" / family / (method + ".jsonl")).resolve(),
                    (root / "exp_code_ljj/artifacts/generation_from_start192" / family / (method + ".jsonl")).resolve(),
                }
                if generation.resolve() not in allowed_sources:
                    raise ValueError("Evaluation contract uses an unexpected generation source")
                if (contract["revision"] != CONFIG["revision"] or contract["prompt_sha256"] != digest(PROMPT_FILE)
                        or contract["expected_indexes"] != selected or contract["method_slug"] != method
                        or contract["judge"] != judge or contract["generation_sha256"] != digest(generation)
                        or summary["contract_hash"] != object_hash(contract)):
                    raise ValueError("Table result contract mismatch")
                sources = {r["index"]: r for r in read_jsonl(generation)}
                records = read_jsonl(output / "judgments.jsonl")
                if len({r["index"] for r in records}) != len(records) or any(r["index"] not in selected for r in records):
                    raise ValueError("Duplicate or out-of-subset judgment")
                for row in records:
                    validate_record(row, sources[row["index"]], selected_golden[row["index"]], object_hash(contract))
                n = summary["evaluated_count"]
                summary = summarize(records, selected_golden, contract)
                summary["judgments_sha256"] = digest(output / "judgments.jsonl")
                all_requests = [read_json(p) for p in (output / "requests").glob("*.json")
                                if not p.name.endswith(".error.json")]
                summary["total_independent_requests"] = len(all_requests)
                summary["format_retry_count"] = sum(r["attempts"] - 1 for r in all_requests)
                atomic_json(summary_path, summary)
                n = summary["evaluated_count"]
                if n:
                    values = (f"{n}/192 | {summary['success_within5_percent']:.2f} ({summary['success_count']}/{n}) | "
                              f"{summary['win1_percent']:.2f} | {summary['win_full_percent']:.2f} | "
                              f"{summary['mean_rounds_or_cap5']:.3f} | {192-n}")
            label = Path(JUDGES[judge]).name + " (BF16, unified v2)"
            lines.append(f"| {label} | {METHOD_LABELS[method]} | {model} | {values} |")
        lines.append(f"|  | Our | {model} | 待评估 / 192 | - | - | - | - | 192 |")
    lines.extend(["", "成功率为五轮内任一轮接受目标的比例；平均轮数取首次成功轮，失败计5。Win@1 比较首条完整回复；Win-Full 比较全部五轮与原始 golden 全部可用对话；两项均独立匿名换序两次，胜=1、平=0.5、负=0。",
                  "", "指标按完成且全部字段解析成功的场景计算；Parsed/selected 显示实际分母与固定192条，未完成或格式未解决单独列示。小于192的行是部分进度，不作为完整最终结果。严格有限 JSON 解码及最新 v2 重试机制保持不变；技术错误保留日志并停止该任务，不伪造分数。",
                  "", "仅运行 Selene 与 Qwen3-14B；API 评估暂不执行。评估输出：(" + str(work / family) + ")。", ""])
    table = root / "exp_code_ljj/results" / f"main_experiment_table_{table_suffix}.md"
    import tempfile
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=table.parent, delete=False) as f:
        f.write("\n".join(lines))
        temporary = f.name
    os.replace(temporary, table)


def run(args):
    root = args.repo_root.resolve()
    work = root / "exp_code_ljj/artifacts" / WORK_DIR
    work.mkdir(parents=True, exist_ok=True)
    scripts = root / "my_eclp_final"
    for family in FAMILIES:
        render(root, work, family)
    if args.init_only:
        return
    queues = {}
    for family in FAMILIES:
        queues[family] = []
        for judge in JUDGES:
            for method in METHODS:
                outputs = evaluation_outputs(work, family, judge, method)
                if any((output / "summary.json").exists()
                       and read_json(output / "summary.json").get("complete") is True
                       and read_json(output / "summary.json").get("evaluated_count") == 192
                       for _, output in outputs):
                    continue
                queues[family].append((judge, method))
    active, failures, complete, retries = {}, [], [], {}
    for family in FAMILIES:
        for judge, method in list(queues[family]):
            for gpu, output in evaluation_outputs(work, family, judge, method):
                matching = find_eval_pid(output)
                if matching is not None:
                    active[family] = {"adopted_pid": matching, "judge": judge,
                                      "method": method, "output": output, "gpu": gpu}
                    queues[family].remove((judge, method))
                    print(f"Adopted running evaluation PID {matching}: {family}/{judge}/{method} GPU {gpu}", flush=True)
                    break
            if family in active:
                break
    with writer_lock(work / ".runner.lock"):
        while any(queues.values()) or active:
            for family, (gpu, model, suffix) in FAMILIES.items():
                current = active.get(family)
                if current:
                    if current.get("adopted_pid"):
                        alive = pid_matches_output(current["adopted_pid"], current["output"])
                        code = None if alive else 0
                    else:
                        code = current["process"].poll()
                    if code is None:
                        continue
                    if current.get("log") is not None:
                        current["log"].close()
                    del active[family]
                    if code:
                        key = (family, current["judge"], current["method"])
                        retries[key] = retries.get(key, 0) + 1
                        if retries[key] <= 3:
                            queues[family].insert(0, (current["judge"], current["method"]))
                            print(f"Retrying {key} after exit {code} ({retries[key]}/3)", flush=True)
                        else:
                            failures.append({"family": family, "judge": current["judge"], "method": current["method"], "exit_code": code})
                    else:
                        summary_path = current["output"] / "summary.json"
                        saved_summary = read_json(summary_path) if summary_path.exists() else {}
                        if saved_summary.get("complete") is True and saved_summary.get("evaluated_count") == 192:
                            complete.append({"family": family, "judge": current["judge"], "method": current["method"]})
                        else:
                            key = (family, current["judge"], current["method"])
                            retries[key] = retries.get(key, 0) + 1
                            if retries[key] <= 3:
                                queues[family].insert(0, (current["judge"], current["method"]))
                                print(f"Resuming incomplete {key} ({retries[key]}/3)", flush=True)
                            else:
                                failures.append({"family": family, "judge": current["judge"], "method": current["method"], "reason": "incomplete_after_retries"})
                    render(root, work, family)
                if not queues[family]:
                    continue
                ready = None
                for position, (judge, method) in enumerate(queues[family]):
                    minimum = 24 if judge == "selene8b" else 36
                    busy_gpus = {entry["gpu"] for entry in active.values()}
                    target = runtime_output(work, family, judge, method, minimum, busy_gpus)
                    if target is None:
                        continue
                    target_gpu, output = target
                    saved_run = output / "run.json"
                    if saved_run.exists():
                        generation = Path(read_json(saved_run)["generation_source"])
                        evaluator = "evaluate_selected192_v2.py"
                    else:
                        generation = preferred_generation(root, family, method)
                        if generation is None:
                            continue
                        evaluator = "evaluate_selected192_v2.py"
                    matching = find_eval_pid(output)
                    if matching is not None:
                        active[family] = {"adopted_pid": matching, "judge": judge,
                                          "method": method, "output": output, "gpu": target_gpu}
                        queues[family].pop(position)
                        print(f"Adopted running evaluation PID {matching}: {family}/{judge}/{method} GPU {target_gpu}", flush=True)
                        ready = "adopted"
                        break
                    if not generation.exists():
                        continue
                    with generation.open() as f:
                        count = sum(1 for line in f if line.strip())
                    row_indexes = set()
                    with generation.open() as f:
                        for line in f:
                            if line.strip():
                                row_indexes.add(json.loads(line)["index"])
                    selection = read_json(root / "exp_code_ljj/data/test_selection_192.json")["original_indexes"]
                    if count == 1000 or (count == 192 and row_indexes == set(selection)):
                        ready = (position, judge, method, generation, target_gpu, output)
                        break
                if ready is None:
                    continue
                if ready == "adopted":
                    continue
                position, judge, method, generation, target_gpu, output = ready
                logs = work / "logs"
                logs.mkdir(exist_ok=True)
                handle = (logs / f"{family}_{judge}_{method}_gpu{target_gpu}.log").open("a", encoding="utf-8")
                command = [sys.executable, str(scripts / evaluator),
                           "--repo-root", str(root), "--generation", str(generation),
                           "--golden", str(root / "eclp_runs_gpu4/evaluation_v2/golden"),
                           "--output", str(output), "--judge", judge, "--method", method,
                           "--physical-gpu", str(target_gpu), "--batch-size", "2", "--scenario-chunk-size", "8",
                           "--memory-cap-gib", "22" if judge == "selene8b" else "32"]
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(target_gpu), PYTHONUNBUFFERED="1")
                process = subprocess.Popen(command, cwd=root, env=environment, stdout=handle, stderr=subprocess.STDOUT)
                active[family] = {"process": process, "log": handle, "judge": judge,
                                  "method": method, "output": output, "gpu": target_gpu}
                queues[family].pop(position)
                print(f"Started {family}/{judge}/{method} GPU {target_gpu}, fixed N=192", flush=True)
            atomic_json(work / "status.json", {"state": "running", "complete": complete, "failures": failures,
                        "active": {f: {"pid": a.get("adopted_pid", a.get("process").pid if a.get("process") else None),
                                        "judge": a["judge"], "method": a["method"], "gpu": a["gpu"],
                                        "output": str(a["output"])}
                                   for f,a in active.items()},
                        "pending": queues, "updated_at": time.strftime("%FT%T%z")})
            if any(queues.values()) or active:
                time.sleep(30)
        atomic_json(work / "status.json", {"state": "failed" if failures else "complete", "complete": complete, "failures": failures})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--init-only", action="store_true")
    run(parser.parse_args())
