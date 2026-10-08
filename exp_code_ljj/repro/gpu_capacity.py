"""Wait for enough BF16 GPU capacity without changing model precision."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence


def model_weight_bytes(model_path: str | Path) -> int:
    path = Path(model_path).expanduser().resolve()
    adapter_config = path / "adapter_config.json"
    if adapter_config.is_file():
        path = Path(json.loads(adapter_config.read_text(encoding="utf-8"))[
            "base_model_name_or_path"
        ]).expanduser().resolve()
    indexes = list(path.glob("*.safetensors.index.json"))
    if indexes:
        return int(json.loads(indexes[0].read_text(encoding="utf-8"))[
            "metadata"
        ]["total_size"])
    return sum(item.stat().st_size for item in path.glob("*.safetensors"))


def wait_for_bf16_capacity(
    model_paths: Sequence[str | Path],
    device_ids: Sequence[int],
    *,
    reserve_gib: float,
    description: str,
    poll_seconds: int = 30,
) -> None:
    """Wait until all requested full BF16 weights plus runtime headroom fit."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"{description} requires CUDA; BF16 precision will not be changed")
    devices = sorted(set(device_ids))
    if not devices:
        raise ValueError("At least one CUDA device is required")
    for device_id in devices:
        with torch.cuda.device(device_id):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(f"cuda:{device_id} does not support BF16")
            if "A100" not in torch.cuda.get_device_name(device_id):
                raise RuntimeError(
                    f"{description} is restricted to A100; "
                    f"cuda:{device_id} is {torch.cuda.get_device_name(device_id)}"
                )

    weight_bytes = sum(model_weight_bytes(path) for path in model_paths)
    if weight_bytes <= 0:
        raise FileNotFoundError(f"Could not find BF16 model weights for {description}")
    reserve_bytes = int(reserve_gib * 1024**3)
    required_bytes = weight_bytes + reserve_bytes
    total_bytes = sum(
        int(torch.cuda.get_device_properties(device_id).total_memory)
        for device_id in devices
    )
    if required_bytes > total_bytes:
        raise RuntimeError(
            f"{description} needs {required_bytes / 1024**3:.1f} GiB, "
            f"but selected A100 devices provide {total_bytes / 1024**3:.1f} GiB; "
            "the requested BF16 setup cannot fit"
        )

    while True:
        free_bytes = sum(torch.cuda.mem_get_info(device_id)[0] for device_id in devices)
        if free_bytes >= required_bytes:
            print(
                f"{description} BF16 capacity ready: "
                f"weights={weight_bytes / 1024**3:.1f} GiB, "
                f"free={free_bytes / 1024**3:.1f} GiB, "
                f"reserve={reserve_gib:.1f} GiB",
                flush=True,
            )
            return
        print(
            f"{description} waiting for A100 capacity: "
            f"weights={weight_bytes / 1024**3:.1f} GiB, "
            f"free={free_bytes / 1024**3:.1f} GiB, "
            f"required={required_bytes / 1024**3:.1f} GiB; "
            f"retrying in {poll_seconds}s",
            flush=True,
        )
        time.sleep(poll_seconds)
