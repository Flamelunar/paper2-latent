#!/usr/bin/env python3
"""Schedule isolated 192-case unified-v2 evaluations on physical GPUs 0/1."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from experiment_io import atomic_json, digest, object_hash, read_json, read_jsonl, writer_lock
from evaluate_new_baselines192_v2 import JUDGES, METHOD_LABELS, validate_record
from matched_eval_protocol_v2 import CONFIG, PROMPT_FILE

FAMILIES = {
    "meta_llama31": (0, "Meta-Llama-3.1-8B-Instruct", "llama31"),
    "deepseek_r1_qwen14b": (1, "DeepSeek-R1-Distill-Qwen-14B", "deepseek14b"),
}
METHODS = [m for m in METHOD_LABELS if m.startswith("baseline")]


def free_gib(gpu):
    value = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    return float(value.strip()) / 1024


def pid_matches_output(pid, output):
    try:
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        return (any(argument.endswith("evaluate_new_baselines192_v2.py") for argument in arguments)
                and "--output" in arguments and str(output) in arguments)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def find_eval_pid(output):
    result = subprocess.run(["pgrep", "-f", "evaluate_new_baselines192_v2[.]py"],
                            text=True, capture_output=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"pgrep failed while adopting evaluators: {result.stderr.strip()}")
    for line in result.stdout.splitlines():
        if line.isdigit() and pid_matches_output(int(line), output):
            return int(line)
    return None


def render(root, work, family):
    gpu, model, table_suffix = FAMILIES[family]
    selected = read_json(root / "exp_code_ljj/data/test_selection_192.json")["original_indexes"]
    golden = {r["index"]: r for r in read_jsonl(root / "eclp_runs_gpu4/evaluation_v2/golden/golden.jsonl")}
    lines = ["## Main experiment: unified v2 (192 scenarios)", "",
             f"Persuader: ({'/app/hzj/models/' + model}). Persuadee: (/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged).",
             f"Prompt / calculation revision: `{CONFIG['revision']}`. Judges and generation use BF16 without quantization or thinking output.",
             "", "固定测试集 **N=192**：原始测试点 `target_turn == 1` 且 `history == []`，使用 [test_selection_192.json](../data/test_selection_192.json) 中的原始索引。全部方法与 Judge 使用相同场景；从已有生成抽取，不重新生成。",
             "", "| Judge | Method | Persuader model | Parsed / selected N | Success within 5 rounds (%) | Win@1 (%) | Win-Full (%) | Mean rounds (failure=5) | Pending / unresolved |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for judge in JUDGES:
        for method in METHODS:
            output = work / family / judge / method
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
                    validate_record(row, sources[row["index"]], golden[row["index"]], object_hash(contract))
                n = summary["evaluated_count"]
                if n != len(records) or summary["expected_count"] != 192:
                    raise ValueError("Summary/judgment count mismatch")
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
    work = root / "exp_code_ljj/artifacts/evaluation_from_start192_v2"
    work.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    for family in FAMILIES:
        render(root, work, family)
    if args.init_only:
        return
    queues = {}
    for family in FAMILIES:
        queues[family] = []
        for judge in JUDGES:
            for method in METHODS:
                summary_path = work / family / judge / method / "summary.json"
                if summary_path.exists():
                    saved_summary = read_json(summary_path)
                    if saved_summary.get("complete") is True and saved_summary.get("evaluated_count") == 192:
                        continue
                queues[family].append((judge, method))
    active, failures, complete = {}, [], []
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
                    current["log"].close()
                    del active[family]
                    if code:
                        failures.append({"family": family, "judge": current["judge"], "method": current["method"], "exit_code": code})
                        queues[family].clear()
                    else:
                        complete.append({"family": family, "judge": current["judge"], "method": current["method"]})
                    render(root, work, family)
                if not queues[family]:
                    continue
                ready = None
                available = free_gib(gpu)
                for position, (judge, method) in enumerate(queues[family]):
                    output = work / family / judge / method
                    saved_run = output / "run.json"
                    if saved_run.exists():
                        generation = Path(read_json(saved_run)["generation_source"])
                    else:
                        generation = root / "exp_code_ljj/artifacts/generation_from_start192" / family / (method + ".jsonl")
                        if not generation.exists():
                            generation = root / "exp_code_ljj/artifacts/generation_5rounds" / family / (method + ".jsonl")
                    matching = find_eval_pid(output)
                    if matching is not None:
                        active[family] = {"adopted_pid": matching, "judge": judge,
                                          "method": method, "output": output}
                        queues[family].pop(position)
                        print(f"Adopted running evaluation PID {matching}: {family}/{judge}/{method}", flush=True)
                        ready = "adopted"
                        break
                    minimum = 24 if judge == "selene8b" else 36
                    if available < minimum or not generation.exists():
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
                        ready = (position, judge, method, generation)
                        break
                if ready is None:
                    continue
                if ready == "adopted":
                    continue
                position, judge, method, generation = ready
                output = work / family / judge / method
                logs = work / "logs"
                logs.mkdir(exist_ok=True)
                handle = (logs / f"{family}_{judge}_{method}.log").open("a", encoding="utf-8")
                saved_run = output / "run.json"
                evaluator = "evaluate_new_baselines192_v2.py" if saved_run.exists() else "evaluate_selected192_v2.py"
                command = [sys.executable, str(scripts / evaluator),
                           "--repo-root", str(root), "--generation", str(generation),
                           "--golden", str(root / "eclp_runs_gpu4/evaluation_v2/golden"),
                           "--output", str(output), "--judge", judge, "--method", method,
                           "--physical-gpu", str(gpu), "--batch-size", "2", "--scenario-chunk-size", "8",
                           "--memory-cap-gib", "22" if judge == "selene8b" else "32"]
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
                process = subprocess.Popen(command, cwd=root, env=environment, stdout=handle, stderr=subprocess.STDOUT)
                active[family] = {"process": process, "log": handle, "judge": judge, "method": method}
                queues[family].pop(position)
                print(f"Started {family}/{judge}/{method} GPU {gpu}, fixed N=192", flush=True)
            for family in FAMILIES:
                render(root, work, family)
            atomic_json(work / "status.json", {"state": "running", "complete": complete, "failures": failures,
                        "active": {f: {"pid": a["process"].pid, "judge": a["judge"], "method": a["method"], "gpu": FAMILIES[f][0]} for f,a in active.items()},
                        "pending": queues, "updated_at": time.strftime("%FT%T%z")})
            if any(queues.values()) or active:
                time.sleep(30)
        atomic_json(work / "status.json", {"state": "failed" if failures else "complete", "complete": complete, "failures": failures})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--init-only", action="store_true")
    run(parser.parse_args())
