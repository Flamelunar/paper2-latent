"""Validate complete new-family v18 judgments and update only their table cells."""

import argparse
import hashlib
import json
import os
import re
import tempfile
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERSUADEE = "/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged"
ADAPTER_ROOT = Path("/app/hzj/liujianjian/persuader_atapter_save")
METHODS = {
    "baseline1_zero_shot": "Baseline 1: Zero-shot",
    "baseline2_dialogue_sft": "Baseline 2: Dialogue-SFT",
    "baseline3_isr_simdpo": "Baseline 3: ISR-SFT + SimDPO",
    "baseline4_ma2p": "Baseline 4: MA2P",
    "baseline5_latentratt": "Baseline 5: LatentRATT",
}
FAMILIES = {
    "meta_llama31": ("/app/hzj/models/Meta-Llama-3.1-8B-Instruct", "main_experiment_table_llama31.md"),
    "deepseek_r1_qwen14b": ("/app/hzj/models/DeepSeek-R1-Distill-Qwen-14B", "main_experiment_table_deepseek14b.md"),
}
JUDGES = {
    "qwen3_14b": ("/app/hzj/models/Qwen3-14B", "qwen3_14b", "Qwen3-14B"),
    "selene1": ("/app/hzj/models/Selene-1-Mini-Llama-3.1-8B", "selene", "Selene-1-Mini-Llama-3.1-8B"),
}
THINK = re.compile(r"</?think\b", re.I)


def load_rows(path, expected_rows=1000):
    with path.open(encoding="utf-8") as handle:
        rows = []
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line {path}:{number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Non-object JSONL line {path}:{number}")
            rows.append(row)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} records: {path}, found {len(rows)}")
    return rows


def generated_text_has_thinking(value):
    if isinstance(value, str):
        return bool(THINK.search(value))
    if isinstance(value, dict):
        return any(generated_text_has_thinking(item) for item in value.values())
    if isinstance(value, list):
        return any(generated_text_has_thinking(item) for item in value)
    return False


def validate_generations(rows, points, *, method, expected_model):
    if len(rows) != 1000 or len(points) != 1000:
        raise ValueError("Full new-family reporting requires the fixed 1,000 test cases")
    for index, (row, point) in enumerate(zip(rows, points)):
        if type(row.get("index")) is not int or row["index"] != index:
            raise ValueError(f"Generation indices/order mismatch for {method}")
        if (row.get("method") != method or row.get("persuader_model") != expected_model
                or row.get("persuadee_model") != PERSUADEE):
            raise ValueError(f"Generation method/model mismatch: {method} index {index}")
        if any(row.get(key) != point.get(key) for key in ("scenario_id", "source_id", "session_id", "target_turn", "scenario", "reference")):
            raise ValueError(f"Generation differs from fixed test case: {method} index {index}")
        if row.get("initial_history") != point.get("history"):
            raise ValueError(f"Initial dialogue differs from fixed test case: {method} index {index}")
        expected_decode = {"rounds": 5, "temperature": .95, "top_p": .7, "top_k": 50, "max_new_tokens": 512}
        if any(row.get(key) != value for key, value in expected_decode.items()):
            raise ValueError(f"Generation sampling protocol mismatch: {method} index {index}")
        turns = row.get("trajectory")
        if not isinstance(turns, list) or len(turns) != 10:
            raise ValueError(f"Incomplete five-round generation: {method} index {index}")
        for position, turn in enumerate(turns):
            if (not isinstance(turn, dict) or turn.get("round") != position // 2 + 1
                    or turn.get("role") != ("persuader" if position % 2 == 0 else "persuadee")
                    or not isinstance(turn.get("content"), str) or not turn["content"].strip()):
                raise ValueError(f"Invalid public turn: {method} index {index}")
        if generated_text_has_thinking(turns) or generated_text_has_thinking(row.get("final_history", [])):
            raise ValueError(f"Thinking leaked into generated views: {method} index {index}")
        history = row.get("final_history")
        expected_history = [{"role": turn["role"], "content": turn["content"]}
                            for turn in point["history"] + turns]
        if (not isinstance(history, list) or any(not isinstance(turn, dict) for turn in history)
                or [{"role": turn.get("role"), "content": turn.get("content")} for turn in history] != expected_history):
            raise ValueError(f"Final dialogue differs from public trajectory: {method} index {index}")


def summarize_rows(rows, generations, *, method, judge_model, expected_rows=1000):
    if len(rows) != expected_rows or len(generations) != expected_rows:
        raise ValueError("Incomplete judgments or generations")
    parsed, unresolved, success_total, win_total, round_total, calls = 0, 0, 0, 0.0, 0, 0
    for index, (row, generation) in enumerate(zip(rows, generations)):
        if type(row.get("index")) is not int or row["index"] != index:
            raise ValueError("Judgment indices must be exactly 0..N-1 in fixed order")
        if (row.get("method") != method or row.get("trial") != 1
                or row.get("scenario_id") != generation["scenario_id"]
                or row.get("model") != generation["persuader_model"]):
            raise ValueError(f"Judgment/source mismatch at index {index}")
        expected = {"judge_model": judge_model, "prompt_version": "v18",
                    "prompt_revision": "joint_strict_v18_json_schema",
                    "judge_compute_dtype": "bfloat16", "judge_quantization": "none",
                    "max_parse_retries": 3}
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Judge protocol mismatch at index {index}")
        outputs = row.get("raw_outputs")
        if (not isinstance(outputs, list) or not 1 <= len(outputs) <= 4
                or any(not isinstance(value, str) for value in outputs)
                or row.get("model_calls") != len(outputs)
                or row.get("raw_output") != outputs[-1]
                or row.get("retry_outputs") != outputs[1:]):
            raise ValueError(f"Invalid retry provenance at index {index}")
        calls += len(outputs)
        if row.get("parse_error") is not None:
            if row.get("verdict") is not None or len(outputs) != 4:
                raise ValueError(f"Unresolved verdict/retry mismatch at index {index}")
            unresolved += 1
            continue
        raw = json.loads(outputs[-1])
        if not isinstance(raw, dict) or set(raw) != {"round_success", "win_ab", "win_ba"}:
            raise ValueError(f"Successful raw response violates v18 schema at index {index}")
        rounds = raw["round_success"]
        if (not isinstance(rounds, list) or len(rounds) != 5 or any(type(value) is not bool for value in rounds)
                or any(rounds[position] and not all(rounds[position:]) for position in range(5))
                or raw["win_ab"] not in ("A", "B") or raw["win_ba"] not in ("A", "B")):
            raise ValueError(f"Invalid native-bool/monotonic/win verdict at index {index}")
        verdict = row.get("verdict")
        if not isinstance(verdict, dict) or any(verdict.get(key) != value for key, value in raw.items()):
            raise ValueError(f"Stored verdict differs from raw response at index {index}")
        first = next((position + 1 for position, value in enumerate(rounds) if value), None)
        win = ((raw["win_ab"] == "A") + (raw["win_ba"] == "B")) / 2
        if (verdict.get("first_success_round") != first
                or verdict.get("success_within_5_rounds") != (first is not None)
                or verdict.get("win_credit") != win):
            raise ValueError(f"Derived verdict is inconsistent at index {index}")
        parsed += 1
        success_total += first is not None
        win_total += win
        round_total += first if first is not None else 5
    return {"total": expected_rows, "parsed": parsed, "unresolved": unresolved,
            "strict_acceptance_percent": 100 * success_total / parsed if parsed else None,
            "win_rate_percent": 100 * win_total / parsed if parsed else None,
            "mean_rounds": round_total / parsed if parsed else None,
            "parse_rate_percent": 100 * parsed / expected_rows,
            "model_calls": calls, "extra_retry_calls": calls - expected_rows,
            "persuader_model": generations[0]["persuader_model"]}


def update_table(text, summaries, *, judge_prefix):
    reverse = {label: slug for slug, label in METHODS.items()}
    current_judge, updated, output = "", set(), []
    for line in text.splitlines(keepends=True):
        if not line.lstrip().startswith("|"):
            output.append(line)
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 8:
            output.append(line)
            continue
        if cells[0]:
            current_judge = cells[0]
        method = reverse.get(cells[1])
        if not current_judge.startswith(judge_prefix) or method not in summaries:
            output.append(line)
            continue
        item = summaries[method]
        cells[2] = item["persuader_model"]
        cells[3] = str(item["parsed"])
        cells[4] = "-" if item["strict_acceptance_percent"] is None else f'{item["strict_acceptance_percent"]:.2f}'
        cells[5] = "-" if item["win_rate_percent"] is None else f'{item["win_rate_percent"]:.2f}'
        cells[6] = "-" if item["mean_rounds"] is None else f'{item["mean_rounds"]:.3f}'
        cells[7] = str(item["unresolved"])
        if method in updated:
            raise ValueError("Duplicate baseline row in table")
        updated.add(method)
        output.append("| " + " | ".join(cells) + " |\n")
    if updated != set(summaries):
        raise ValueError(f"Table missing required {judge_prefix} method rows")
    return "".join(output)


def atomic_write(path, text, *, expected_original=None):
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix="." + path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        if expected_original is not None and path.read_text(encoding="utf-8") != expected_original:
            raise ValueError(f"Table changed during reporting; refusing to overwrite user edits: {path}")
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(FAMILIES), required=True)
    parser.add_argument("--judge-name", choices=tuple(JUDGES))
    parser.add_argument("--validate-generations-only", action="store_true",
                        help="Check all five fixed test files before spending GPU time judging")
    args = parser.parse_args()
    if not args.validate_generations_only and args.judge_name is None:
        parser.error("--judge-name is required when reporting judgments")
    base_model, table_name = FAMILIES[args.family]
    if not args.validate_generations_only:
        judge_model, judge_slug, judge_prefix = JUDGES[args.judge_name]
    exp = ROOT / "exp_code_ljj"
    test_path = exp / "data/evaluation_points.jsonl"
    manifest = json.loads((exp / "data/manifest.json").read_text(encoding="utf-8"))
    test_digest = hashlib.sha256(test_path.read_bytes()).hexdigest()
    if test_digest != manifest["files"]["evaluation_points"]["sha256"]:
        raise ValueError("Fixed evaluation split checksum mismatch")
    points = load_rows(test_path)
    adapter = ADAPTER_ROOT / args.family
    expected_models = dict.fromkeys(METHODS, base_model)
    expected_models.update(baseline2_dialogue_sft=str(adapter / "baseline2_dialogue_sft"),
                           baseline3_isr_simdpo=str(adapter / "baseline3_simdpo"),
                           baseline5_latentratt=str(adapter / "baseline5_latentratt"))
    summaries = {}
    for method in METHODS:
        generation_path = exp / "artifacts/generation_5rounds" / args.family / (method + ".jsonl")
        generations = load_rows(generation_path)
        validate_generations(generations, points, method=method, expected_model=expected_models[method])
        if args.validate_generations_only:
            summaries[method] = {"rows": len(generations), "generation_path": str(generation_path),
                                 "generation_sha256": hashlib.sha256(generation_path.read_bytes()).hexdigest()}
            continue
        judge_path = exp / "artifacts/success_judgments" / (judge_slug + "_" + args.family) / (method + ".jsonl")
        summaries[method] = summarize_rows(load_rows(judge_path), generations, method=method, judge_model=judge_model)
        summaries[method].update({
            "generation_path": str(generation_path), "judge_path": str(judge_path),
            "generation_sha256": hashlib.sha256(generation_path.read_bytes()).hexdigest(),
            "judge_sha256": hashlib.sha256(judge_path.read_bytes()).hexdigest(),
        })
    if args.validate_generations_only:
        print(json.dumps({"family": args.family, "validated": "5 x 1000 fixed five-round generations",
                          "test_sha256": test_digest, "methods": summaries}, ensure_ascii=False), flush=True)
        return
    table = exp / "results" / table_name
    original_text = table.read_text(encoding="utf-8")
    text = update_table(original_text, summaries, judge_prefix=judge_prefix)
    report = {"family": args.family, "judge_model": judge_model, "prompt_version": "v18",
              "prompt_revision": "joint_strict_v18_json_schema", "test_sha256": test_digest,
              "metrics_use_parsed_cases_only": True, "methods": summaries}
    summary_path = exp / "results" / f"summary_v18_{args.family}_{args.judge_name}.json"
    atomic_write(summary_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    atomic_write(table, text, expected_original=original_text)
    print(f"Validated 5 x 1,000 judgments; updated {table}; summary {summary_path}", flush=True)


if __name__ == "__main__":
    main()
