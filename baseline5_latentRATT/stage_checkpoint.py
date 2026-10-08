"""Inspect a LatentRATT stage without loading models or touching checkpoints."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from numerical_checks import validate_statistics

CHECKPOINT = re.compile(r"(?:checkpoint|oom-step)-(\d+)")


def usable_weights(path, model):
    adapter = path / "adapter"
    required = (path / "attention.pt", path / "training_state.pt", adapter / "adapter_config.json")
    if not all(file.is_file() and file.stat().st_size > 0 for file in required):
        return False
    if not any((adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        return False
    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if Path(adapter_config["base_model_name_or_path"]).resolve() != Path(model).resolve():
        raise ValueError(f"Checkpoint belongs to another base model: {path}")
    return True


def inspect_stage(output, *, stage, model, train_sha256, max_steps=None, physical_gpu=None):
    expected_steps = max_steps or (3742 if stage == "sft" else 29934)

    def read_config(path):
        config_path = path / "run_config.json"
        if not config_path.is_file():
            return None
        config = json.loads(config_path.read_text(encoding="utf-8"))
        matches = (
            config.get("stage") == stage and config.get("model") == model
            and config.get("train_sha256") == train_sha256
            and config.get("train_rows") == 59868 and config.get("train_scenarios") == 10000
            and config.get("epochs") == 1 and config.get("max_steps") == max_steps
            and config.get("batch_size") == 2 and config.get("gradient_accumulation") == 16
            and config.get("compute_dtype") == "bfloat16" and config.get("quantization") == "none"
            and (physical_gpu is None or config.get("cuda_visible_devices") == str(physical_gpu))
        )
        if not matches:
            raise ValueError(f"Incompatible {stage} stage provenance: {path}")
        validate_statistics(config)
        return config

    config = read_config(output)
    if config is not None and usable_weights(output, model):
        full_completed = config.get("training_complete") is True if max_steps is None else True
        if config.get("completed_steps") == expected_steps and full_completed:
            return "complete", output
        raise ValueError(f"Stage root is not a completed {stage} run: {output}")

    candidates = []
    partial_checkpoint = False
    if output.is_dir():
        for path in output.iterdir():
            match = CHECKPOINT.fullmatch(path.name)
            if not path.is_dir() or not match:
                continue
            partial_checkpoint = True
            config = read_config(path)
            if config is not None and usable_weights(path, model):
                step = int(match.group(1))
                if config.get("completed_steps", step) != step or not 0 <= step <= expected_steps:
                    raise ValueError(f"Invalid checkpoint step: {path}")
                candidates.append((step, path.stat().st_mtime_ns, path))
    if candidates:
        return "resume", max(candidates, key=lambda item: item[:2])[2]
    if partial_checkpoint or config is not None:
        raise ValueError(f"Incomplete checkpoint artifacts with no valid recovery state: {output}")
    return "start", None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("sft", "rl"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--physical-gpu", choices=("0", "1"))
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    digest = hashlib.sha256(args.train_file.read_bytes()).hexdigest()
    mode, path = inspect_stage(args.output, stage=args.stage, model=args.model,
                               train_sha256=digest, max_steps=args.max_steps,
                               physical_gpu=args.physical_gpu)
    print(mode)
    print(str(path) if path is not None else "")


if __name__ == "__main__":
    main()
