"""Select a recoverable local Trainer checkpoint without loading tensors."""
import argparse
import json
import re
from pathlib import Path


def select_checkpoint(directory, model):
    choices = []
    if not directory.is_dir():
        return None
    for path in directory.iterdir():
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if not path.is_dir() or not match:
            continue
        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json")
        if not all((path / name).is_file() for name in required):
            continue
        if not any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
            continue
        config = json.loads((path / "adapter_config.json").read_text())
        if Path(config["base_model_name_or_path"]).resolve() != Path(model).resolve():
            raise ValueError(f"Checkpoint base model mismatch: {path}")
        state = json.loads((path / "trainer_state.json").read_text())
        step = int(match.group(1))
        if state["global_step"] != step:
            raise ValueError(f"Checkpoint step mismatch: {path}")
        choices.append((step, path))
    if not choices and any(path.is_dir() and path.name.startswith("checkpoint-") for path in directory.iterdir()):
        raise ValueError(f"No recoverable checkpoint; refusing a silent fresh restart: {directory}")
    return max(choices, key=lambda item:item[0])[1] if choices else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("model")
    args = parser.parse_args()
    result = select_checkpoint(args.directory, args.model)
    print(str(result) if result else "")
