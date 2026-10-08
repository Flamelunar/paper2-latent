#!/usr/bin/env python3
"""Generate five-round dialogues for any implemented baseline on the fixed test set."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from repro.profile import add_profile_argument, configure_profile_from_argv

PROFILE = configure_profile_from_argv()
from repro.constants import ROOT
from repro.baseline_data import sha256
from repro.gpu_capacity import wait_for_bf16_capacity
from repro.persuadee_simulation import PersuaderMethod, run_simulation

BASE_MODEL = "/app/hzj/models/Qwen2.5-7B-Instruct"
SFT_MODEL = "/data1/liujianjian/Persuader/qwen2.5-7b-instruct-dialogue-sft-paper10k"
SIMDPO_MODEL = "/data1/liujianjian/Persuader/baseline3_isr_simdpo_adapter"
PERSUADEE_MODEL = "/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged"
TEST_DATA = ROOT / "data/evaluation_points.jsonl"
OUTPUT_DIR = ROOT / "artifacts/generation_5rounds"
PROTOCOL = "paper10k_common_prompt_fixed_qwen35_persuadee_5rounds_bf16_v1"
METHOD_NAMES = ("baseline1_zero_shot", "baseline2_dialogue_sft", "baseline3_isr_simdpo")
ARCHIVED_QWEN25_PROTOCOL = "qwen25_table2_fixed_qwen35_persuadee_5rounds_bf16_contextbatch2_v1"
ARCHIVED_GENERATIONS = {
    METHOD_NAMES[0]: (ROOT / "artifacts/generation_5rounds/legacy/qwen25_zero_shot.jsonl",
                      "qwen25_zero_shot"),
    METHOD_NAMES[1]: (ROOT / "artifacts/generation_5rounds/legacy/qwen25_dialogue_sft.jsonl",
                      "qwen25_dialogue_sft"),
}


def methods_for_args(args: argparse.Namespace) -> tuple[PersuaderMethod, ...]:
    def base_for(model_path: str, fallback: str) -> str:
        config_path = Path(model_path) / "adapter_config.json"
        if config_path.is_file():
            return str(json.loads(config_path.read_text(encoding="utf-8"))[
                "base_model_name_or_path"
            ])
        return fallback

    baseline2_base = base_for(args.baseline2_model, args.base_model)
    baseline3_base = base_for(args.baseline3_model, args.base_model)
    return (
        PersuaderMethod(METHOD_NAMES[0], f"Baseline 1: {Path(args.base_model).name} Zero-shot",
                        args.base_model, False),
        PersuaderMethod(METHOD_NAMES[1], f"Baseline 2: {Path(baseline2_base).name} Dialogue-SFT",
                        args.baseline2_model, False),
        PersuaderMethod(METHOD_NAMES[2], f"Baseline 3: {Path(baseline3_base).name} ISR-SFT + SimDPO",
                        args.baseline3_model, True),
    )


def check_adapter_base(method: PersuaderMethod, base_model: str) -> None:
    config_path = Path(method.model_path) / "adapter_config.json"
    if not config_path.is_file():
        return
    trained_base = json.loads(config_path.read_text(encoding="utf-8"))["base_model_name_or_path"]
    if Path(trained_base).expanduser().resolve() != Path(base_model).expanduser().resolve():
        raise SystemExit(f"{method.slug} adapter belongs to {trained_base}; selected base is {base_model}")


def check_bf16_capacity(model_path: str, device_spec: str) -> None:
    import torch
    from repro.simulation import LocalTransformersEndpoint

    device_ids = LocalTransformersEndpoint._parse_devices(device_spec)
    for device_id in device_ids:
        with torch.cuda.device(device_id):
            if not torch.cuda.is_bf16_supported():
                raise SystemExit(f"cuda:{device_id} does not support BF16")
            if "A100" not in torch.cuda.get_device_name(device_id):
                raise SystemExit(f"Model inference is restricted to A100; cuda:{device_id} is {torch.cuda.get_device_name(device_id)}")
    wait_for_bf16_capacity(
        [model_path], device_ids, reserve_gib=2,
        description=f"Inference {model_path}",
    )


def check_shared_bf16_capacity(model_paths: list[str], device_specs: list[str]) -> None:
    """Preflight resident persuader and persuadee weights when they share a GPU."""
    import torch
    from repro.simulation import LocalTransformersEndpoint

    devices = sorted({
        device for spec in device_specs
        for device in LocalTransformersEndpoint._parse_devices(spec)
    })
    for device_id in devices:
        with torch.cuda.device(device_id):
            if not torch.cuda.is_bf16_supported():
                raise SystemExit(f"cuda:{device_id} does not support BF16")
            if "A100" not in torch.cuda.get_device_name(device_id):
                raise SystemExit(f"Model inference is restricted to A100; cuda:{device_id} is {torch.cuda.get_device_name(device_id)}")
    wait_for_bf16_capacity(
        model_paths, devices, reserve_gib=8,
        description="Shared-device inference",
    )


def reuse_archived_generation(
    args: argparse.Namespace, slug: str, *, report: bool = True,
) -> Path | None:
    """Reuse a verified original Qwen2.5 trajectory under the shared protocol."""
    if slug == METHOD_NAMES[0]:
        from repro.zero_shot_contract import validate_zero_shot_model
        validate_zero_shot_model(args.base_model)
    if args.method is not None or args.limit is not None:
        return None
    if slug not in ARCHIVED_GENERATIONS:
        raise ValueError(f"No archived generation is registered for {slug}")
    source_path, source_method = ARCHIVED_GENERATIONS[slug]
    if slug == METHOD_NAMES[1] and Path(args.baseline2_model).is_dir():
        return None
    if not source_path.is_file():
        return None
    expected_persuader = args.base_model if slug == METHOD_NAMES[0] else args.baseline2_model
    method_number = 1 if slug == METHOD_NAMES[0] else 2
    method_label = "Zero-shot" if method_number == 1 else "Dialogue-SFT"
    display_name = f"Baseline {method_number}: {Path(args.base_model).name} {method_label}"
    expected_settings = {
        "rounds": 5,
        "temperature": 0.95,
        "top_p": 0.7,
        "top_k": 50,
        "max_new_tokens": 512,
        "context_batch_size": 2,
    }
    actual_settings = {
        "rounds": args.rounds,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "context_batch_size": args.context_batch_size,
    }
    if actual_settings != expected_settings:
        raise ValueError(
            f"The archived Baseline {method_number} trajectory can only be reused with its original "
            f"shared generation settings: expected {expected_settings}, got {actual_settings}"
        )

    with source_path.open(encoding="utf-8") as handle:
        source_rows = [json.loads(line) for line in handle if line.strip()]
    with args.test_data.open(encoding="utf-8") as handle:
        test_points = [json.loads(line) for line in handle if line.strip()]
    if len(source_rows) != 1000 or len(test_points) != 1000:
        raise ValueError(f"Archived Baseline {method_number} reuse requires exactly 1,000 contexts")
    source_by_index = {int(row.get("index", -1)): row for row in source_rows}
    if len(source_by_index) != len(source_rows) or set(source_by_index) != set(range(1000)):
        raise ValueError(f"Archived Baseline {method_number} trajectory has missing or duplicate indices")

    normalized_rows: list[dict[str, object]] = []
    for index, point in enumerate(test_points):
        row = source_by_index[index]
        expected_context = (
            row.get("scenario_id") == point.get("scenario_id")
            and row.get("source_id") == point.get("source_id")
            and row.get("session_id") == point.get("session_id")
            and row.get("reference") == point.get("reference")
            and row.get("initial_history") == point.get("history")
        )
        if not expected_context:
            raise ValueError(f"Archived Baseline {method_number} context differs from the fixed test set at index {index}")
        expected_metadata = (
            row.get("protocol") == ARCHIVED_QWEN25_PROTOCOL
            and row.get("method") == source_method
            and row.get("persuader_model") == expected_persuader
            and row.get("persuadee_model") == args.persuadee_model
            and row.get("rounds") == expected_settings["rounds"]
            and row.get("temperature") == expected_settings["temperature"]
            and row.get("top_p") == expected_settings["top_p"]
            and row.get("top_k") == expected_settings["top_k"]
            and row.get("max_new_tokens") == expected_settings["max_new_tokens"]
            and row.get("context_batch_size") == expected_settings["context_batch_size"]
        )
        if not expected_metadata:
            raise ValueError(f"Archived Baseline {method_number} generation settings mismatch at index {index}")
        normalized = dict(row)
        normalized["protocol"] = PROTOCOL
        normalized["method"] = slug
        normalized["method_name"] = display_name
        normalized_rows.append(normalized)

    output = args.output_dir / f"{slug}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_output = True
    if output.exists() and not args.force:
        with output.open(encoding="utf-8") as handle:
            existing_rows = [json.loads(line) for line in handle if line.strip()]
        if existing_rows == normalized_rows:
            write_output = False
        else:
            raise ValueError(
                f"Existing Baseline {method_number} generation differs from the verified archive: "
                f"{output}; use --force to replace it"
            )
    if write_output:
        temporary = output.with_name(output.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in normalized_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(output)

    config_path = args.output_dir / "run_config_baseline3.json"
    if not config_path.is_file():
        # Older shared runs used the generic filename; keep that archive readable.
        config_path = args.output_dir / "run_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        sources = config.setdefault("reused_generation_sources", {})
        sources[slug] = {
                "source": str(source_path),
                "source_protocol": ARCHIVED_QWEN25_PROTOCOL,
                "target_protocol": PROTOCOL,
                "validation": "All 1,000 indices, context fields, model paths, persuadee, and decoding settings matched.",
        }
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report:
        print(f"Reused verified archived Baseline {method_number} generation: {output}")
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_profile_argument(parser)
    parser.add_argument("--test-data", type=Path, default=TEST_DATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--baseline2-model", default=SFT_MODEL)
    parser.add_argument("--baseline3-model", default=SIMDPO_MODEL)
    parser.add_argument("--persuadee-model", default=PERSUADEE_MODEL)
    parser.add_argument("--persuader-device", default="cuda:2")
    parser.add_argument("--persuadee-device", default="cuda:2")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--context-batch-size", type=int, default=2)
    parser.add_argument("--method", choices=METHOD_NAMES, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if PROFILE != "paper10k":
        raise SystemExit("Baseline generation requires the paper10k held-out split")
    if args.rounds != 5:
        raise SystemExit("This reproduction protocol fixes five Persuader-Persuadee rounds")
    if args.context_batch_size < 1 or args.context_batch_size > 4:
        raise SystemExit("--context-batch-size must be between 1 and 4")
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    if sha256(args.test_data) != manifest["files"]["evaluation_points"]["sha256"]:
        raise SystemExit("All baselines must use the fixed paper10k evaluation_points.jsonl")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    reused_archives: dict[str, Path] = {}
    if args.method is None and args.limit is None:
        for slug in METHOD_NAMES[:2]:
            archived_path = reuse_archived_generation(args, slug)
            if archived_path is not None:
                reused_archives[slug] = archived_path

    methods = methods_for_args(args)
    selected_methods = tuple(
        method for method in methods
        if method.slug not in reused_archives
        and (method.slug == args.method or (args.method is None and Path(method.model_path).is_dir()))
    )
    if not selected_methods:
        raise SystemExit("No selected Persuader model is available")
    for method in selected_methods:
        if not Path(method.model_path).is_dir():
            raise SystemExit(f"Missing Persuader model: {method.model_path}")
        if method.slug == METHOD_NAMES[0]:
            from repro.zero_shot_contract import validate_zero_shot_model
            validate_zero_shot_model(method.model_path)
        adapter_config = Path(method.model_path) / "adapter_config.json"
        expected_base = args.base_model
        if adapter_config.is_file() and method.slug != METHOD_NAMES[0]:
            expected_base = str(json.loads(adapter_config.read_text(encoding="utf-8"))[
                "base_model_name_or_path"
            ])
        check_adapter_base(method, expected_base)
    if not Path(args.persuadee_model).is_dir():
        raise SystemExit(f"Missing Persuadee model: {args.persuadee_model}")

    from repro.simulation import LocalTransformersEndpoint
    shared_devices = (
        set(LocalTransformersEndpoint._parse_devices(args.persuader_device))
        & set(LocalTransformersEndpoint._parse_devices(args.persuadee_device))
    )
    if shared_devices:
        for method in selected_methods:
            check_shared_bf16_capacity(
                [method.model_path, args.persuadee_model],
                [args.persuader_device, args.persuadee_device],
            )
    else:
        for method in selected_methods:
            check_bf16_capacity(method.model_path, args.persuader_device)
        check_bf16_capacity(args.persuadee_model, args.persuadee_device)
    paths = run_simulation(
        methods=selected_methods,
        test_data=args.test_data,
        output_dir=args.output_dir,
        persuadee_model=args.persuadee_model,
        persuader_device=args.persuader_device,
        persuadee_device=args.persuadee_device,
        persuader_quantization="none",
        rounds=args.rounds,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        force=args.force,
        top_p=args.top_p,
        top_k=args.top_k,
        context_batch_size=args.context_batch_size,
        protocol=PROTOCOL,
    )
    for slug in reused_archives:
        # run_simulation writes run_config.json; add archive provenance afterward.
        archived_path = reuse_archived_generation(args, slug, report=False)
        if archived_path is not None:
            paths.append(archived_path)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
