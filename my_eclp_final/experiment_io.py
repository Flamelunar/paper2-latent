"""Small, standard-library-only helpers shared by the GPU4 pipeline."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _sanitize_surrogates(value: Any) -> Any:
    if isinstance(value, str):
        cleaned = []
        index = 0
        while index < len(value):
            codepoint = ord(value[index])
            if 0xD800 <= codepoint <= 0xDBFF and index + 1 < len(value):
                following = ord(value[index + 1])
                if 0xDC00 <= following <= 0xDFFF:
                    cleaned.append(chr(0x10000 + ((codepoint - 0xD800) << 10) + following - 0xDC00))
                    index += 2
                    continue
            cleaned.append("\uFFFD" if 0xD800 <= codepoint <= 0xDFFF else value[index])
            index += 1
        return "".join(cleaned)
    if isinstance(value, dict):
        return {_sanitize_surrogates(key): _sanitize_surrogates(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_surrogates(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_surrogates(item) for item in value)
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(_sanitize_surrogates(value), sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_sanitize_surrogates(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_sanitize_surrogates(value), ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{number}; retain the file for diagnosis") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at {path}:{number}")
            rows.append(item)
    return rows


@contextlib.contextmanager
def writer_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def immutable_json(path: Path, value: dict) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Run contract changed at {path}; do not mix incompatible results")
    else:
        atomic_json(path, value)


def require_indexes(rows: list[dict], count: int = 1000) -> dict[int, dict]:
    if len(rows) != count:
        raise ValueError(f"Expected exactly {count} rows, found {len(rows)}")
    if any(type(row.get("index")) is not int for row in rows):
        raise ValueError("Indexes must be integers")
    by_index = {row["index"]: row for row in rows}
    if len(by_index) != count or set(by_index) != set(range(count)):
        raise ValueError("Missing, duplicate, or out-of-range indexes")
    return by_index
