#!/usr/bin/env python3
"""Append a parallel v2 table, retaining the original table and legacy scores."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from experiment_io import atomic_json, digest, object_hash, read_json, read_jsonl, writer_lock
from matched_eval_protocol_v2 import CONFIG, PROMPT_FILE
from evaluate_matched_v2 import METHOD_LABELS, summarize, validate_record

START = "<!-- unified-evaluation-v2:start -->"
END = "<!-- unified-evaluation-v2:end -->"
LEGACY_START = "<!-- baseline-five-rounds:start -->"
LEGACY_END = "<!-- baseline-five-rounds:end -->"


def legacy_rows(text: str) -> list[dict]:
    if LEGACY_START not in text or LEGACY_END not in text:
        raise ValueError("Missing original table markers")
    section = text.split(LEGACY_START, 1)[1].split(LEGACY_END, 1)[0]
    result = []
    group = None
    reverse = {label: slug for slug, label in METHOD_LABELS.items()}
    for line in section.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().split("|")[1:-1]]
        if len(cells) != 8 or cells[1] not in reverse:
            continue
        if cells[0]:
            group = cells[0]
        if group is None:
            raise ValueError("Method row has no Judge group")
        if "Selene-1-Mini" in group:
            judge, label = "selene8b", "Selene-1-Mini-Llama-3.1-8B (BF16, unified v2)"
        elif "Gemma 4" in group:
            continue
        elif "Qwen3-14B" in group:
            judge, label = "qwen3_14b", "Qwen3-14B (BF16, unified v2)"
        else:
            raise ValueError(f"Unknown Judge group: {group}")
        result.append({"judge": judge, "judge_label": label, "show_judge": bool(cells[0]),
                       "method": cells[1], "slug": reverse[cells[1]],
                       "actor": cells[2] if cells[2] not in {"", "-", "rl"} else "Qwen2.5-7B-Instruct"})
    identities = [(r["judge"], r["slug"]) for r in result]
    if not result or len(set(identities)) != len(identities):
        raise ValueError("Missing or duplicate rows in original table")
    return result


def verified_complete_summary(directory: Path, judge: str, method: str) -> dict | None:
    path = directory / "summary.json"
    if not path.exists():
        return None
    saved = read_json(path)
    if not saved.get("complete"):
        return None
    contract = read_json(directory / "run.json")
    if (contract.get("revision") != CONFIG["revision"] or contract.get("judge") != judge
            or contract.get("method_slug") != method
            or contract.get("prompt_sha256") != digest(PROMPT_FILE)
            or contract.get("protocol_sha256") != digest(Path(__file__).with_name("matched_eval_protocol_v2.py"))
            or contract.get("evaluator_sha256") != digest(Path(__file__).with_name("evaluate_matched_v2.py"))):
        raise ValueError("V2 result uses another Judge, method, prompt, or calculation version")
    source_path, gold_path = Path(contract["generation_source"]), Path(contract["golden_source"])
    if digest(source_path) != contract["generation_sha256"] or digest(gold_path) != contract["golden_sha256"]:
        raise ValueError("V2 generation or golden changed")
    sources = {r["index"]: r for r in read_jsonl(source_path)}
    gold = {r["index"]: r for r in read_jsonl(gold_path)}
    records_path = directory / "judgments.jsonl"
    records = read_jsonl(records_path)
    if (len(records) != 1000 or len(sources) != 1000 or len(gold) != 1000
            or {r["index"] for r in records} != set(range(1000))):
        raise ValueError("A complete v2 table row requires all 1,000 scenarios")
    for r in records:
        validate_record(r, sources[r["index"]], gold[r["index"]], object_hash(contract))
    recomputed = summarize(records, gold, contract)
    if any(saved.get(k) != value for k, value in recomputed.items()) or saved.get("judgments_sha256") != digest(records_path):
        raise ValueError("V2 saved summary differs from the underlying records")
    # Current main table uses the fixed, output-independent empty-history subset.
    selection_path = directory.parents[3] / "exp_code_ljj/data/test_selection_192.json"
    selection = read_json(selection_path)
    points_path = selection_path.parent / "evaluation_points.jsonl"
    points = read_jsonl(points_path)
    selected = selection["original_indexes"]
    expected = [i for i, point in enumerate(points)
                if point["target_turn"] == 1 and point["history"] == []]
    if (selection["count"] != 192 or selected != expected or len(selected) != 192
            or selection["source_sha256"] != digest(points_path)):
        raise ValueError("The fixed 192-scenario test selection changed")
    by_index = {r["index"]: r for r in records}
    for index in selected:
        if (sources[index]["initial_history"] != [] or gold[index]["initial_history"] != []
                or sources[index]["scenario_id"] != points[index]["scenario_id"]):
            raise ValueError("The selected scenario is not an aligned empty-history test")
    subset = summarize([by_index[i] for i in selected],
                       {i: gold[i] for i in selected}, contract)
    subset.update(test_selection_revision=selection["revision"],
                  original_indexes=selected, original_evaluation_count=1000,
                  original_judgments_sha256=digest(records_path),
                  selection_manifest_sha256=digest(selection_path),
                  reused_saved_judgments=True)
    return subset


def render_section(original: str, work: Path, template_only: bool = False) -> str:
    lines = [START, "## Main experiment: unified v2 evaluation (Win@1 / Win-Full)", "",
             "本表采用最新统一评估协议，使用 Selene-8B 与 Qwen3-14B 两个 Judge，沿用上表的方法顺序、Qwen2.5 劝说模型和固定 192 个无原始前置对话的测试场景。所有方法与 Judge 使用完全相同的场景子集。`-` 表示尚未完成此版本评估；上表旧分数不填入本表。", "",
             f"Prompt / calculation revision: `{CONFIG['revision']}`；测试子集版本：`from_start_192_20261006`。",
             "数据清单：[test_selection_192.json](../data/test_selection_192.json)；测试数据：[evaluation_points_from_start192.jsonl](../data/evaluation_points_from_start192.jsonl)。筛选条件为原测试点 `target_turn == 1` 且 `history == []`，保留原始索引，不重新抽样。今后主实验仅测试这 192 条；训练数据不变。当前分数从已保存的逐场景 v2 判断重算，无新推理。",
             "提示词：[matched_judge_prompts_v2.json](../../my_eclp_final/matched_judge_prompts_v2.json)；协议说明：[EVALUATION_PROTOCOL_V2_ZH.md](../../my_eclp_final/EVALUATION_PROTOCOL_V2_ZH.md)。", "",
             "| Judge | Method | Persuader model | N | Goal acceptance within 5 rounds (%) | Win@1 (%) | Win-Full (%) | Mean rounds to success (failures = 5) | Success format unresolved |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in legacy_rows(original):
        s = None if template_only else verified_complete_summary(work / row["judge"] / row["slug"], row["judge"], row["slug"])
        label = row["judge_label"] if row["show_judge"] else ""
        if s is None:
            values = "待评估 / 192 | - | - | - | - | -"
        else:
            values = (f"192/192 | {s['success_within5_percent']:.2f} ({s['success_count']}/192) | "
                      f"{s['win1_percent']:.2f} | {s['win_full_percent']:.2f} | "
                      f"{s['mean_rounds_or_cap5']:.3f} | {s['format_unresolved_completed_cases']}")
        lines.append(f"| {label} | {row['method']} | {row['actor']} | {values} |")
    lines += ["",
              'Success：逐轮使用截至当前轮的前缀，只判断 Persuadee 是否明确接受精确目标，不提供未来对话或 golden。只允许输出 `{"success":true}` 或 `{"success":false}`。礼貌、兴趣或愿意考虑不自动算成功；不要求已经执行、固定关键词或正式承诺。实施安排与尚待满足的关键接受条件按统一规则区分。五轮内任一轮接受计成功；首次成功轮次用于平均轮数，未成功计 5。第五轮当前接受率另存 summary.json。',
              "Win@1：双方同一测试起点后的第一条完整 Persuader 回复对比，提供共同背景和目标，初始历史为空。",
              "Win-Full：Our 全部已生成五轮对话与同一起点后 golden 的全部可用对话对比，包含双方角色；golden 不截断、不补齐。双方长度可能不同，缺失回复不当成接受或拒绝；因此本指标不是等轮数的 Win@5。原始 golden 与当前模拟对话对象的来源差异也需保留在结果解读中。",
              "两项 Win 指标均执行两次独立匿名 A/B 换序请求，每次只返回 A、B 或 Tie。每次胜=1、平=0.5、负=0，两次取平均，再对相同 192 个场景取平均×100%。顺序冲突与显式平局分别统计。",
              "Baseline 5 的生成文件未记录生成精度或量化设置；本表评估已保存的回复，不推断该组生成时使用 BF16。",
              "所有 Judge 使用相同 rubric、字段、严格 JSON schema 与有限 token 解码约束。格式/技术失败不得转换成假成功或直接剔除后当作完整结果；完整行要求这 192 个固定场景全部完成；格式未解决不得剔除。上方旧协议表为历史 1,000 条结果，不能与本表混作同口径比较。", END, ""]
    return "\n".join(lines)


def replace_section(original: str, section: str) -> str:
    if START in original or END in original:
        if original.count(START) != 1 or original.count(END) != 1:
            raise ValueError("Malformed or duplicated v2 table markers")
        begin = original.index(START)
        end = original.index(END, begin) + len(END)
        return original[:begin] + section.rstrip() + original[end:]
    separator = "" if original.endswith("\n\n") else "\n" if original.endswith("\n") else "\n\n"
    return original + separator + section


def update_table(table: Path, work: Path, template_only: bool = False, dry_run: bool = False) -> dict:
    with writer_lock(table.parent / ".our_table.lock"):
        for attempt in range(5):
            original = table.read_bytes()
            updated = replace_section(original.decode("utf-8"), render_section(original.decode("utf-8"), work, template_only)).encode("utf-8")
            if dry_run:
                return {"dry_run": True, "rendered": updated.decode("utf-8")}
            if updated == original:
                return {"changed": False, "table": str(table), "table_sha256": digest(table)}
            temporary = table.with_name(table.name + f".v2-tmp-{os.getpid()}")
            with temporary.open("wb") as handle:
                handle.write(updated)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, table.stat().st_mode & 0o777)
            if table.read_bytes() != original:
                temporary.unlink()
                continue
            backup = work / "table_update" / f"main_experiment_table.before_v2_{time.time_ns()}.md"
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(original)
            os.replace(temporary, table)
            report = {"changed": True, "table": str(table), "backup": str(backup),
                      "table_sha256": digest(table), "revision": CONFIG["revision"],
                      "template_only": template_only, "model_evaluation_started": False}
            atomic_json(work / "table_update/report.json", report)
            return report
    raise RuntimeError("Original table was modified concurrently; retry rather than overwriting another writer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--template-only", action="store_true", help="Only draw pending rows; do not load any score")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    import json
    root = args.repo_root.resolve()
    print(json.dumps(update_table(root / "exp_code_ljj/results/main_experiment_table.md",
                                 root / "eclp_runs_gpu4/evaluation_v2", args.template_only, args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
