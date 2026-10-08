#!/usr/bin/env python3
"""Replace only the two Our rows in the existing Qwen2.5 result table."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from evaluate_our import ACTOR, JUDGES, METHOD, ROOT, summarize, verify_generation
from experiment_io import atomic_json, digest, object_hash, read_json, read_jsonl, writer_lock


def verify_summary(directory: Path, generation: Path, work: Path, judge: str) -> dict:
    contract = read_json(directory / "run.json")
    if (contract.get("judge") != judge or contract.get("judge_model") != JUDGES[judge]["model"]
            or contract.get("prompt_version") != JUDGES[judge]["version"]
            or contract.get("prompt_revision") != JUDGES[judge]["revision"]
            or contract.get("method") != METHOD or contract.get("actor_base_model") != ACTOR
            or contract.get("physical_gpu") != 4 or contract.get("dtype") != "bfloat16"
            or contract.get("quantization") != "none" or contract.get("max_parse_retries") != 3):
        raise ValueError(f"Wrong evaluation contract in {directory}")
    if contract.get("generation_sha256") != digest(generation / "generation.jsonl"):
        raise ValueError("Summary refers to another generation source")
    source = directory / f"{METHOD}.jsonl"
    summary = summarize(read_jsonl(source), verify_generation(generation, work), judge, contract)
    saved = read_json(directory / "summary.json")
    if any(saved.get(k) != value for k, value in summary.items()):
        raise ValueError(f"Saved summary differs from recomputed judgments in {directory}")
    if saved.get("judgments_sha256") != digest(source) or saved.get("generation_sha256") != contract["generation_sha256"]:
        raise ValueError("Judgment hashes changed after summary")
    if not summary["valid_count"]:
        raise ValueError("No jointly parsed cases; do not fill the table with fabricated zeroes")
    return saved


def row_text(summary: dict) -> str:
    n, successes = summary["valid_count"], summary["success_count"]
    return (f"|  | Our | Qwen2.5-7B-Instruct | {n}/1000 | "
            f"{summary['success_percent']:.2f} ({successes}/{n}) | {summary['win_rate']:.2f} | "
            f"{summary['mean_rounds_or_cap5']:.3f} | {summary['unresolved_cases']} |")


def replace_our_rows(text: str, replacements: dict[str, str]) -> str:
    lines = text.splitlines(keepends=True)
    group = None
    found = {name: 0 for name in replacements}
    patched = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [part.strip() for part in stripped.split("|")[1:-1]]
            if len(cells) == 8:
                if cells[0] in {p["table_label"] for p in JUDGES.values()} or (cells[0] and "(BF16" in cells[0]):
                    group = cells[0]
                if cells[1] == "Our" and group in replacements:
                    found[group] += 1
                    patched.append(replacements[group] + ("\n" if line.endswith("\n") else ""))
                    continue
        patched.append(line)
    if any(count != 1 for count in found.values()):
        raise ValueError(f"Expected one existing Our row in each requested Judge group; found {found}")
    # A line-for-line replacement preserves every other score and paragraph.
    if len(lines) != len(patched):
        raise RuntimeError("Table row count changed")
    return "".join(patched)


def update_table(work: Path, generation: Path, directories: dict[str, Path], table: Path, dry_run: bool = False) -> dict:
    expected = ROOT / "exp_code_ljj/results/main_experiment_table.md"
    if table.resolve() != expected.resolve():
        raise ValueError("This updater only writes the Qwen2.5 table requested by the user")
    summaries = {judge: verify_summary(directory, generation, work, judge) for judge, directory in directories.items()}
    replacements = {JUDGES[judge]["table_label"]: row_text(summary) for judge, summary in summaries.items()}
    with writer_lock(table.parent / ".our_table.lock"):
        for attempt in range(5):
            original = table.read_bytes()
            updated = replace_our_rows(original.decode("utf-8"), replacements).encode("utf-8")
            if dry_run:
                return {"dry_run": True, "rows": replacements, "source_table_sha256": digest(table)}
            if updated == original:
                return {"changed": False, "table": str(table), "rows": replacements, "table_sha256": digest(table)}
            temporary = table.with_name(table.name + f".our-tmp-{os.getpid()}")
            with temporary.open("wb") as handle:
                handle.write(updated)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, table.stat().st_mode & 0o777)
            # Read latest table immediately before replace: other baseline
            # updaters do not share this lock, so a detected edit is retried.
            if table.read_bytes() != original:
                temporary.unlink()
                continue
            backup = work / "table_update" / f"main_experiment_table.before_{time.time_ns()}.md"
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(original)
            os.replace(temporary, table)
            result = {"changed": True, "table": str(table), "rows": replacements,
                      "table_sha256": digest(table), "backup": str(backup),
                      "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      "summaries": summaries, "generation_sha256": digest(generation / "generation.jsonl")}
            atomic_json(work / "table_update" / "report.json", result)
            if table.read_bytes() == updated:
                return result
        raise RuntimeError("Table changed concurrently; retained Our summaries, retry the update after the other writer finishes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT / "eclp_runs_gpu4")
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--selene", type=Path)
    parser.add_argument("--qwen3", type=Path)
    parser.add_argument("--table", type=Path, default=ROOT / "exp_code_ljj/results/main_experiment_table.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    directories = {k: p.resolve() for k, p in {"selene8b": args.selene, "qwen3_14b": args.qwen3}.items() if p is not None}
    if not directories:
        parser.error("At least one complete Our evaluation is required")
    result = update_table(args.work.resolve(), args.generation.resolve(), directories, args.table.resolve(), args.dry_run)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
