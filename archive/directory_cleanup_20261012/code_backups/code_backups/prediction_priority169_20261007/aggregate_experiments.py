#!/usr/bin/env python3
"""Scene-paired repeat-aware bootstrap and common-source main/domain tables."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
from typing import Any
from evaluation.modern_pipeline import normalize_item
from evaluation.status_policy import parse_failure
from utils.artifacts import last_records, write_json, sha256_file, hash_object
from utils.evaluation_io import read_json_or_jsonl
from paper_table_figures import BACKBONE_LABELS, model_reuse_note

METHODS = ("zero_shot", "ma2p", "sft", "grpo_only", "sft_grpo", "ours")
METHOD_NAMES = {"zero_shot": "Zero-shot", "sft": "SFT", "ma2p": "MA2P", "grpo_only": "GRPO-only", "sft_grpo": "SFT + GRPO", "ours": "PMM (Ours)"}
BACKBONE_NAMES = BACKBONE_LABELS


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    at = (len(ordered) - 1) * fraction
    lo, hi = int(at), min(len(ordered) - 1, int(at) + 1)
    return ordered[lo] + (at - lo) * (ordered[hi] - ordered[lo])


def bootstrap(means: dict[str, float], *, reps: int, seed: int = 2026) -> list[float]:
    # Resample SCENARIOS, retaining all their repeats together.
    values = list(means.values())
    rng = random.Random(seed)
    n = len(values)
    samples = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps)]
    return [quantile(samples, .025), quantile(samples, .975)]


def scenario_means(rows: dict[tuple[str, int], dict[str, Any]], ids: list[str], repeats: list[int]) -> tuple[dict[str, float], int, int]:
    result, unknown, excluded = {}, 0, 0
    for identifier in ids:
        values = []
        for repeat in repeats:
            row = rows.get((identifier, repeat))
            if row and parse_failure(row):
                excluded += 1
                continue
            value = row.get("metrics", {}).get("strict_success") if row and row.get("status") == "completed" else None
            if type(value) is not bool:
                unknown += 1
            else:
                values.append(float(value))
        if values:
            result[identifier] = sum(values) / len(values)
    return result, unknown, excluded


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def comparison_position_metrics(rows: list[dict[str, Any]], expected: int, metric: str = "comparative") -> dict[str, Any]:
    comparisons = [r.get("metrics", {}).get(metric, {}) for r in rows]
    valid = [c for c in comparisons if c.get("status") == "completed"
             and type(c.get("swap_conflict")) is bool and type(c.get("strict_swap_conflict")) is bool
             and isinstance(c.get("order_credits"), list) and len(c["order_credits"]) == 2
             and all(type(v) in (int, float) and v in (0, .5, 1) for v in c["order_credits"])]
    full = len(valid) == expected
    percent = lambda values: 100 * _mean(values) if full and values else None
    return {"valid": len(valid), "expected": expected, "unknown": expected - len(valid),
        "swap_conflict_percent": percent([float(c["swap_conflict"]) for c in valid]),
        "strict_swap_conflict_percent": percent([float(c["strict_swap_conflict"]) for c in valid]),
        "generated_first_credit_percent": percent([c["order_credits"][0] for c in valid]),
        "generated_second_credit_percent": percent([c["order_credits"][1] for c in valid]),
        "unambiguous_generated_win_percent": percent([float(c.get("generated_more_persuasive") is True) for c in valid]),
        "unambiguous_reference_win_percent": percent([float(c.get("reference_more_persuasive") is True) for c in valid])}


def generation_cost_metrics(rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    costs = [r.get("generation_costs", {}) for r in rows]
    valid = [c for c in costs if type(c.get("requests")) is int and isinstance(c.get("by_role"), dict)]
    full = len(valid) == expected
    usage_full = full and all(c.get("usage_complete") is True for c in valid)
    avg = lambda key: _mean([float(c.get(key, 0)) for c in valid]) if full else None
    by_role = {}
    for role in sorted({role for c in valid for role in c["by_role"]}):
        rc = [c["by_role"].get(role, {}) for c in valid]
        by_role[role] = {key: _mean([float(c.get(key, 0)) for c in rc])
                        if full and (usage_full or key not in {"input_tokens", "output_tokens"}) else None
                        for key in ("requests", "input_tokens", "output_tokens", "latency_ms", "invalid_outputs", "factual_rejections")}
    return {"valid": len(valid), "expected": expected, "usage_complete": usage_full,
            "requests_per_dialogue": avg("requests"), "retry_requests_per_dialogue": avg("retry_requests"),
            "request_latency_ms_per_dialogue": avg("request_latency_ms"),
            "input_tokens_per_dialogue": avg("input_tokens") if usage_full else None,
            "output_tokens_per_dialogue": avg("output_tokens") if usage_full else None,
            "by_role_per_dialogue": by_role}


def _persuader_tokens(row: dict[str, Any]) -> int | None:
    reply_tokens = row.get("generation_metrics", {}).get("persuader_reply_output_tokens")
    role_costs = row.get("generation_costs", {}).get("by_role", {}).get("persuader", {})
    requests = role_costs.get("requests")
    failed_requests = role_costs.get("failed_requests")
    if (type(reply_tokens) is not int or reply_tokens < 0
            or type(requests) is not int or requests < 1
            or failed_requests != 0):
        return None
    return reply_tokens


def aggregate(plan: dict[str, Any], evaluation: list[dict[str, Any]], root: Path, *, reps: int = 10000) -> dict[str, Any]:
    items = [normalize_item(row, i)[0] for i, row in enumerate(evaluation)]
    ids = [item["id"] for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate scenario ids")
    repeats = plan["configuration"]["rollout_seed_ids"]
    groups: dict[tuple[str, str, str], dict[tuple[str, int], dict[str, Any]]] = defaultdict(dict)
    sources = {}
    for cell in plan["cells"]:
        for judge, directory in cell["judge_dirs"].items():
            path = root / directory / "results.jsonl"
            if not path.exists():
                continue
            sources[str(path)] = sha256_file(path)
            for row in last_records(read_json_or_jsonl(path)):
                if row.get("judge_id") != judge or row["method"] != cell["method"] or row["backbone"] != cell["backbone"]:
                    raise ValueError("Mismatched method/backbone/judge identity")
                if row["rollout_seed_id"] != cell["repeat_id"] or row["scenario_id"] not in ids:
                    raise ValueError("Unexpected repeat/scenario")
                key = (row["scenario_id"], row["rollout_seed_id"])
                group = groups[(judge, cell["backbone"], cell["method"])]
                if key in group:
                    raise ValueError("Duplicate experiment cell record")
                group[key] = row
    expected_cells = [(judge, backbone, method)
                      for judge in plan["configuration"]["judges"]
                      for backbone in plan["configuration"]["backbones"]
                      for method in METHODS]
    table, comparisons, domain_table, judge_sensitivity = [], [], [], []
    means_by_group = {}
    for key in expected_cells:
        judge, backbone, method = key
        rows = groups.get(key, {})
        means, unknown, excluded = scenario_means(rows, ids, repeats)
        means_by_group[key] = means
        valid_rows = [r for r in rows.values() if not parse_failure(r) and r.get("status") == "completed"
                      and type(r.get("metrics", {}).get("strict_success")) is bool]
        denominator = len(valid_rows)
        successes = sum(r["metrics"]["strict_success"] is True for r in valid_rows)
        complete = unknown == 0
        quality = {}
        for metric in ("persuasiveness", "logical_coherence", "helpfulness"):
            vals = [r.get("metrics", {}).get("quality", {}).get(metric) for r in valid_rows]
            available = [v for v in vals if v is not None]
            quality[metric] = {"mean": sum(available) / len(available) if available else None,
                               "valid": len(available), "expected": denominator}
        success_rounds = [float(r["metrics"]["success_round"]) for r in valid_rows
                          if isinstance(r.get("metrics", {}).get("success_round"), int)]
        comparison_credits = [float(r["metrics"]["comparative"]["win_credit"])
                              for r in valid_rows
                              if isinstance(r.get("metrics", {}).get("comparative", {}).get("win_credit"), (int, float))]
        token_values = [_persuader_tokens(r) for r in valid_rows]
        token_values = [value for value in token_values if value is not None]
        comparison_unknown = denominator - len(comparison_credits)
        token_unknown = denominator - len(token_values)
        metric_complete = denominator > 0 and unknown == 0 and comparison_unknown == 0 and token_unknown == 0
        table.append({"judge": judge, "backbone": backbone, "method": method, "shared": backbone == "shared",
            "scenarios": len(ids), "repeats": len(repeats), "valid_denominator": denominator,
            "valid_scenarios": len(means), "excluded_parse_failures": excluded, "unknown_episodes": unknown,
            "success_percent": 100 * successes / denominator if complete and denominator else None,
            "success_95ci_percent": [100 * v for v in bootstrap(means, reps=reps)] if complete and means else None,
            "success_bounds_percent": ([100 * successes / (denominator + unknown),
                100 * (successes + unknown) / (denominator + unknown)]
                if denominator + unknown else [None, None]), "quality": quality,
            "success_round": {"mean": _mean(success_rounds) if unknown == 0 else None,
                              "valid": len(success_rounds), "expected_successes": successes},
            "reference_win": {"mean_credit_percent": 100 * _mean(comparison_credits)
                              if comparison_unknown == 0 and comparison_credits and unknown == 0 else None,
                              "valid": len(comparison_credits), "expected": denominator, "unknown": comparison_unknown,
                              "position_audit": comparison_position_metrics(valid_rows, denominator)},
            "reference_win1": {
                "mean_credit_percent": 100 * _mean([r["metrics"]["win1"]["win_credit"] for r in valid_rows])
                if denominator > 0 and unknown == 0 and len(valid_rows) == denominator and all(type(r.get("metrics", {}).get("win1", {}).get("win_credit")) in (int, float) for r in valid_rows) else None,
                "valid": sum(isinstance(r.get("metrics", {}).get("win1", {}).get("win_credit"), (int, float)) for r in valid_rows),
                "expected": denominator, "position_audit": comparison_position_metrics(valid_rows, denominator, "win1")},
            "generation_compute_cost": generation_cost_metrics(valid_rows, denominator),
            "persuader_tokens": {"mean_output_per_dialogue": _mean(token_values)
                                 if denominator > 0 and token_unknown == 0 and unknown == 0 else None,
                                 "valid": len(token_values), "expected": denominator, "unknown": token_unknown},
            "metric_complete": metric_complete})
        for domain in sorted({item["domain"] for item in items}):
            subset = [item["id"] for item in items if item["domain"] == domain]
            domain_means, domain_unknown, domain_excluded = scenario_means(rows, subset, repeats)
            domain_table.append({"judge": judge, "backbone": backbone, "method": method, "domain": domain,
                "scenarios": len(subset),
                "valid_denominator": sum(1 for r in rows.values() if r.get("scenario_id") in set(subset)
                    and not parse_failure(r) and r.get("status") == "completed"
                    and type(r.get("metrics", {}).get("strict_success")) is bool),
                "valid_scenarios": len(domain_means),
                "excluded_parse_failures": domain_excluded, "unknown_episodes": domain_unknown,
                "success_percent": 100 * sum(domain_means.values()) / len(domain_means)
                    if domain_unknown == 0 and domain_means else None})
    judges = list(plan["configuration"]["judges"])
    for judge in judges:
        for backbone in plan["configuration"]["backbones"]:
            ours = means_by_group.get((judge, backbone, "ours"), {})
            for control in ("zero_shot", "sft", "ma2p", "grpo_only", "sft_grpo"):
                other_backbone = backbone
                other = means_by_group.get((judge, other_backbone, control), {})
                paired_ids = sorted(set(ours) & set(other))
                diffs = {i: ours[i] - other[i] for i in paired_ids}
                comparisons.append({"judge": judge, "backbone": backbone, "comparison": f"ours-vs-{control}",
                    "paired_scenarios": len(diffs), "difference_percentage_points": 100 * sum(diffs.values()) / len(diffs) if diffs else None,
                    "paired_95ci_percentage_points": [100 * v for v in bootstrap(diffs, reps=reps)] if diffs else None})
    if len(judges) == 2:
        a, b = judges
        for backbone, method in sorted({(c["backbone"], c["method"]) for c in plan["cells"]}):
            left, right = groups.get((a, backbone, method), {}), groups.get((b, backbone, method), {})
            pairs = []
            for scenario in ids:
                for repeat in repeats:
                    l, r = left.get((scenario, repeat)), right.get((scenario, repeat))
                    if (l and r and not parse_failure(l) and not parse_failure(r)
                            and l.get("status") == "completed" and r.get("status") == "completed"
                            and type(l.get("metrics", {}).get("strict_success")) is bool
                            and type(r.get("metrics", {}).get("strict_success")) is bool):
                        if l["dialogue_sha256"] != r["dialogue_sha256"]:
                            raise ValueError("Judge sensitivity compared different dialogues")
                        x, y = l["metrics"]["strict_success"], r["metrics"]["strict_success"]
                        if type(x) is bool and type(y) is bool:
                            pairs.append((x, y))
            agreement = sum(x == y for x, y in pairs) / len(pairs) if pairs else None
            pa = sum(x for x, _ in pairs) / len(pairs) if pairs else 0
            pb = sum(y for _, y in pairs) / len(pairs) if pairs else 0
            chance = pa * pb + (1 - pa) * (1 - pb)
            kappa = (agreement - chance) / (1 - chance) if agreement is not None and chance < 1 else None
            judge_sensitivity.append({"backbone": backbone, "method": method, "judges": judges,
                "paired_episodes": len(pairs), "agreement": agreement, "cohen_kappa": kappa,
                "disagreements": sum(x != y for x, y in pairs), "expected": len(ids) * len(repeats),
                "excluded_or_unpaired": len(ids) * len(repeats) - len(pairs)})
    # When complete, compute ALL rows and domain rows from exactly the same
    # source records and assert weighted-domain equality (not rounded means).
    for row in table:
        parts = [r for r in domain_table if all(r[k] == row[k] for k in ("judge", "backbone", "method"))]
        if row["success_percent"] is not None:
            weighted_n = sum(p["valid_denominator"] for p in parts)
            weighted = sum((p["success_percent"] or 0) * p["valid_denominator"] for p in parts) / weighted_n
            if weighted_n != row["valid_denominator"]:
                raise AssertionError("Overall/domain valid denominators differ")
            if abs(weighted - row["success_percent"]) > 1e-8:
                raise AssertionError("Overall/domain inconsistency")
    deferred = [{"id": judge, "label": "GPT6-Luna", "status": "deferred"}
                for judge in ("gpt6_luna",) if judge not in judges]
    return {"status": "complete" if all(r["metric_complete"] for r in table) else "incomplete",
            "main_table": table, "domain_table": domain_table, "paired_comparisons": comparisons,
            "judge_sensitivity": judge_sensitivity, "source_hashes": sources,
            "deferred_evaluators": deferred,
            "bootstrap_unit": "scenario with all repeats retained", "bootstrap_replicates": reps}


def write_paper_tables(report: dict[str, Any], plan: dict[str, Any], output: Path) -> None:
    """Publish audited paper tables; never display an incomplete score."""
    judges = list(plan["configuration"]["judges"])
    backbones = list(plan["configuration"]["backbones"])
    configuration = plan["configuration"]
    scenario_count = configuration.get("data", {}).get("expected_eval_items", 500)
    repeat_count = len(configuration.get("rollout_seed_ids", [0]))
    repeat_label = ("one full interaction rollout" if repeat_count == 1
                    else f"{repeat_count} full interaction rollouts")
    latex_break = chr(92) * 2
    rows = {(r["judge"], r["backbone"], r["method"]): r for r in report["main_table"]}
    for judge in judges:
        judge_label = {"qwen3_14b": "Qwen3-14B", "gpt6_luna": "GPT6-Luna"}.get(judge, judge)
        role_note = model_reuse_note(plan["configuration"], judge)
        latex_role_note = role_note.replace("_", r"\_")
        def get(backbone: str, method: str) -> dict[str, Any] | None:
            return rows.get((judge, backbone, method))

        def success_cell(row: dict[str, Any] | None, *, latex: bool, bold: bool = False) -> str:
            if row is None:
                return r"\textemdash" if latex else "— (missing)"
            if row["success_percent"] is None or row.get("unknown_episodes", 0) != 0:
                missing = row["unknown_episodes"]
                return (rf"\textemdash{{\scriptsize ({missing} missing)}}" if latex
                        else f"— ({missing} missing)")
            score = f"{row['success_percent']:.1f}"
            interval = row["success_95ci_percent"]
            ci = f"[{interval[0]:.1f}, {interval[1]:.1f}]"
            if latex:
                value = rf"\textbf{{{score}}}" if bold else score
                return rf"{value} {{\scriptsize {ci}}}"
            return f"**{score}** {ci}" if bold else f"{score} {ci}"

        markdown = [f"# {judge}: strict success (%)", "",
                    "| World-model backbone | " + " | ".join(METHOD_NAMES[m] for m in METHODS) + " |",
                    "| --- | " + " | ".join("---:" for _ in METHODS) + " |"]
        latex_rows = []
        quality_rows = []
        for backbone in backbones:
            current = {method: get(backbone, method) for method in METHODS}
            complete = [r["success_percent"] for r in current.values() if r is not None and r["success_percent"] is not None]
            best = max(complete) if len(complete) == len(METHODS) else None
            name = BACKBONE_NAMES.get(backbone, backbone)
            markdown.append("| " + name + " | " + " | ".join(
                success_cell(current[m], latex=False, bold=best is not None and
                             current[m] is not None and current[m]["success_percent"] == best)
                for m in METHODS) + " |")
            latex_rows.append(name + " & " + " & ".join(
                success_cell(current[m], latex=True, bold=best is not None and
                             current[m] is not None and current[m]["success_percent"] == best)
                for m in METHODS) + " " + latex_break)
            for index, method in enumerate(METHODS):
                row = current[method]
                method_label = METHOD_NAMES[method].replace("_", r"\_")
                success = success_cell(row, latex=True)
                qualities = []
                for metric in ("persuasiveness", "logical_coherence", "helpfulness"):
                    data = row["quality"][metric] if row else None
                    qualities.append(f"{data['mean']:.2f}" if data and data["mean"] is not None
                                     and data["valid"] == data["expected"] else r"\textemdash")
                quality_rows.append((name if index == 0 else "") + " & " + method_label + " & " +
                                    success + " & " + " & ".join(qualities) + " " + latex_break)
            quality_rows.append(r"\addlinespace[2pt]")

        markdown.extend(["", "Each entry gives the strict-success percentage and its scenario-level bootstrap 95% CI.",
                         "The Zero-shot is evaluated separately for each backbone. Missing episodes remain unknown.",
                         role_note, ""])
        compact = "\n".join([r"\begin{table*}[t]", r"\centering", r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            rf"\caption{{Strict persuasion success under the {judge_label} judge. Each method is evaluated on {scenario_count} scenarios with {repeat_label}. Brackets show scenario-paired bootstrap 95\% intervals.}}",
            rf"\label{{tab:{judge}_main_success}}", r"\begin{tabular}{lcccccc}", r"\toprule",
            "World-model backbone & Zero-shot & SFT & MA2P & GRPO-only & SFT + GRPO & PMM (Ours) " + latex_break,
            r"\midrule", *latex_rows, r"\bottomrule", r"\end{tabular}",
            r"\par\smallskip {\footnotesize Zero-shot is evaluated separately for each backbone. Bold marks the highest score within each backbone, including ties. A dash means the cell has unknown or missing episodes.}",
            (r"\par\smallskip {\footnotesize " + latex_role_note + "}") if role_note else "",
            r"\end{table*}", ""])
        quality = "\n".join([r"\begin{table*}[t]", r"\centering", r"\small",
            r"\setlength{\tabcolsep}{5pt}",
            rf"\caption{{Persuasion success and dialogue quality judged by {judge_label}. Quality scores are on a 1--10 scale and are shown only for cells with all expected ratings.}}",
            rf"\label{{tab:{judge}_main_scores}}", r"\begin{tabular}{llcccc}", r"\toprule",
            r"Backbone & Method & Success (\%) [95\% CI] & Persuasiveness & Coherence & Helpfulness " + latex_break,
            r"\midrule", *quality_rows, r"\bottomrule", r"\end{tabular}",
            r"\par\smallskip {\footnotesize Zero-shot is evaluated separately for each backbone. Missing outcomes or ratings are shown as a dash.}",
            (r"\par\smallskip {\footnotesize " + latex_role_note + "}") if role_note else "",
            r"\end{table*}", ""])
        prefix = "" if len(judges) == 1 else judge + "_"
        (output / f"{prefix}main_success_table.md").write_text("\n".join(markdown), encoding="utf-8")
        (output / f"{prefix}main_success_table.tex").write_text(compact, encoding="utf-8")
        (output / f"{prefix}main_score_table.tex").write_text(quality, encoding="utf-8")


    # Requested compact table: three metrics per judge and one token-cost column.
    qwen = "qwen3_14b"
    gpt = "gpt6_luna"
    rows = {(r["judge"], r["backbone"], r["method"]): r for r in report["main_table"]}
    def get(judge: str, backbone: str, method: str) -> dict[str, Any] | None:
        return rows.get((judge, backbone, method))
    def number(value: Any, digits: int = 1) -> str:
        return "--" if value is None else f"{value:.{digits}f}"
    def judge_values(judge: str, backbone: str, method: str) -> list[str]:
        if judge not in judges:
            return ["-- (deferred)"] * 3
        row = get(judge, backbone, method)
        success = row.get("success_percent") if row and row.get("unknown_episodes", 0) == 0 else None
        win_data = row.get("reference_win", {}) if row else {}
        win = (win_data.get("mean_credit_percent")
               if win_data.get("valid") == win_data.get("expected") and win_data.get("unknown") == 0 else None)
        rounds = row.get("success_round", {}).get("mean") if row and row.get("unknown_episodes", 0) == 0 else None
        return [number(success), number(win), number(rounds, 2)]

    compact_rows: list[list[str]] = []
    for backbone in backbones:
        for method in METHODS:
            row = get(qwen, backbone, method)
            qwen_values = judge_values(qwen, backbone, method)
            gpt_values = judge_values(gpt, backbone, method)
            token_data = row.get("persuader_tokens", {}) if row else {}
            tokens = (token_data.get("mean_output_per_dialogue")
                      if token_data.get("valid") == token_data.get("expected")
                      and token_data.get("unknown") == 0 else None)
            coverage = (f"{row['valid_denominator']}/{row['excluded_parse_failures']}/{row['unknown_episodes']}"
                        if row else "--")
            compact_rows.append([BACKBONE_NAMES.get(backbone, backbone), METHOD_NAMES[method],
                                 coverage, *qwen_values, *gpt_values, number(tokens, 0)])
    headers = ["Backbone", "Method", "Valid / parse-excluded / unresolved",
               "Qwen3-14B success %", "Qwen3-14B win-Full vs. reference %",
               "Qwen3-14B avg success round", "GPT6-Luna success %", "GPT6-Luna win-Full vs. reference %",
               "GPT6-Luna avg success round", "Persuader tokens / dialogue"]
    markdown = ["# Main persuasion metrics", "", "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---", "---", "---:"] + ["---:"] * 7) + " |"]
    markdown.extend("| " + " | ".join(row) + " |" for row in compact_rows)
    markdown.extend(["", "Success is the strict success rate. Win vs. reference is the mean two-order comparison credit; each ordering gives 1 for a win, 0.5 for a tie, and 0 for a loss; the two credits are averaged. Swap-conflict rates are reported in comparison_position_bias.csv.",
                     "Average success round is computed over successful dialogues only. Persuader tokens/dialogue sums output tokens for all Persuader replies in one complete dialogue.",
                     "GPT6-Luna was evaluated on the same recorded dialogues." if gpt in judges else
                     "GPT6-Luna columns are intentionally deferred and were not called.", ""])
    win_headers = ["Backbone", "Method", "Judge", "Win@1 %", "Win-Full %", "Win@1 swap conflict %", "Win-Full swap conflict %"]
    win_rows = []
    for row in report["main_table"]:
        first = row.get("reference_win1", {})
        full = row.get("reference_win", {})
        win_rows.append([BACKBONE_NAMES.get(row["backbone"], row["backbone"]), METHOD_NAMES[row["method"]], row["judge"],
            number(first.get("mean_credit_percent")), number(full.get("mean_credit_percent")),
            number(first.get("position_audit", {}).get("swap_conflict_percent")), number(full.get("position_audit", {}).get("swap_conflict_percent"))])
    (output / "win_metrics.md").write_text("\n".join(["| " + " | ".join(win_headers) + " |", "| " + " | ".join(["---"] * len(win_headers)) + " |"] + ["| " + " | ".join(r) + " |" for r in win_rows]) + "\n", encoding="utf-8")
    with (output / "win_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(win_headers)
        writer.writerows(win_rows)
    (output / "main_metric_table.md").write_text("\n".join(markdown), encoding="utf-8")
    with (output / "main_metric_table.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(compact_rows)
    latex_rows = [" & ".join(row) + r" \\" for row in compact_rows]
    latex = "\n".join([r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{Main persuasion metrics on {scenario_count} held-out scenarios with {repeat_label}.}}",
        r"\label{tab:main_metric_groups}", r"\begin{tabular}{llcrrrrrrr}", r"\toprule",
        r"Backbone & Method & N valid/excl./unres. & \multicolumn{3}{c}{Qwen3-14B} & \multicolumn{3}{c}{GPT6-Luna} & Persuader \\",
        r" & & & Success (\%) & Win vs. ref. (\%) & Avg. round & Success (\%) & Win vs. ref. (\%) & Avg. round & tokens/dialogue \\",
        r"\midrule", *latex_rows, r"\bottomrule", r"\end{tabular}",
        r"\par\smallskip {\footnotesize N valid/excl./unres. reports valid dialogues, parse-excluded dialogues, and unresolved technical failures. Qwen3-14B columns are evaluated. " +
        ("GPT6-Luna was evaluated on the same recorded dialogues. " if gpt in judges else "GPT6-Luna is deferred. ") +
        r"Win vs. reference uses two comparison orders; each tie = 0.5; average the two order credits. Avg. round uses successful dialogues only. Persuader tokens sum output tokens for all Persuader replies per complete dialogue.}",
        r"\end{table*}", ""])
    (output / "main_metric_table.tex").write_text(latex, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--eval-source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    a = p.parse_args()
    if a.bootstrap_replicates < 1:
        raise SystemExit("bootstrap-replicates must be positive")
    plan = json.loads(a.plan.read_text(encoding="utf-8"))
    report = aggregate(plan, read_json_or_jsonl(a.eval_source), Path(plan["configuration"]["project_root"]), reps=a.bootstrap_replicates)
    report["plan_sha256"] = sha256_file(a.plan)
    report["eval_sha256"] = sha256_file(a.eval_source)
    report["comparison_position_bias"] = [{"judge": r["judge"], "backbone": r["backbone"], "method": r["method"],
        **r["reference_win"]["position_audit"]} for r in report["main_table"]]
    report["method_compute_costs"] = [{"judge": r["judge"], "backbone": r["backbone"], "method": r["method"],
        **r["generation_compute_cost"]} for r in report["main_table"]]
    write_json(a.output_dir / "main_tables.json", report)
    write_json(a.output_dir / "metrics.json", report)
    for name in ("main_table", "domain_table", "paired_comparisons", "judge_sensitivity", "comparison_position_bias", "method_compute_costs"):
        rows = report[name]
        if rows:
            with (a.output_dir / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in rows)
    if (a.output_dir / "main_table.csv").is_file():
        (a.output_dir / "aggregate_results.csv").write_text(
            (a.output_dir / "main_table.csv").read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
    write_paper_tables(report, plan, a.output_dir)
    from paper_table_figures import render_paper_figures
    render_paper_figures(report, plan, a.output_dir)
    if (a.output_dir / "main_metric_table.tex").is_file():
        (a.output_dir / "main_results.tex").write_text(
            (a.output_dir / "main_metric_table.tex").read_text(encoding="utf-8"), encoding="utf-8")
    for extension in ("png", "pdf", "svg"):
        source = a.output_dir / f"main_metric_table.{extension}"
        alias = a.output_dir / f"main_results.{extension}"
        if source.is_file():
            alias.write_bytes(source.read_bytes())
    print(report["status"])
    if report["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
