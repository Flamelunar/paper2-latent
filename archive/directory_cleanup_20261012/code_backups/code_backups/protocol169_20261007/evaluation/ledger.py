"""Per-request cost/error records, stable seeds and bounded structural retries."""
from __future__ import annotations

import hashlib
import inspect
import re
import time
from typing import Any, Callable


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()[:4], "big")


class StructuralFailure(RuntimeError):
    pass


class RequestLedger:
    def __init__(self, runners: dict[str, Any], episode_id: str, scenario_id: str, repeat_id: int,
                 retry_limit: int = 1, episode_resample_index: int = 0) -> None:
        if episode_resample_index < 0:
            raise ValueError("episode_resample_index must be nonnegative")
        self.runners, self.episode_id = runners, episode_id
        self.scenario_id, self.repeat_id, self.retry_limit = scenario_id, repeat_id, retry_limit
        self.episode_resample_index = episode_resample_index
        self.requests: list[dict[str, Any]] = []

    def call(self, role: str, prompt: str, request_id: str, *, max_tokens: int,
             temperature: float = 0.0, top_p: float = 1.0, seed_key: str | None = None,
             retry_count: int = 0) -> str:
        runner = self.runners[role]
        seed_parts = (self.scenario_id, self.repeat_id, seed_key or request_id, retry_count)
        if self.episode_resample_index:
            seed_parts += (f"episode-resample-{self.episode_resample_index}",)
        seed = stable_seed(*seed_parts)
        kwargs: dict[str, Any] = {"max_new_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "seed": seed}
        parameters = inspect.signature(runner.generate).parameters
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in parameters}
        started = time.perf_counter()
        entry: dict[str, Any] = {"episode_id": self.episode_id, "scenario_id": self.scenario_id,
            "request_id": request_id, "role": role, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt": prompt,
            "seed": seed, "episode_resample_index": self.episode_resample_index,
            "sampling_requested": {"temperature": temperature, "top_p": top_p},
            "retry_count": retry_count, "max_new_tokens": max_tokens, "status": "pending"}
        try:
            raw, output_tokens = runner.generate(prompt, **kwargs)
            usage = dict(getattr(runner, "last_usage", {}))
            entry.update(usage)
            entry.setdefault("input_tokens", None)
            entry.setdefault("output_tokens", output_tokens)
            entry.update(status="completed", raw=raw)
            return raw
        except Exception as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}", input_tokens=None, output_tokens=None)
            raise
        finally:
            entry["latency_ms"] = (time.perf_counter() - started) * 1000
            self.requests.append(entry)

    def structured(self, role: str, prompt: str, request_id: str, parser: Callable[[str], Any],
                   *, max_tokens: int, temperature: float = 0.0, top_p: float = 1.0,
                   seed_key: str | None = None) -> Any:
        for attempt in range(self.retry_limit + 1):
            suffix = "" if attempt == 0 else "\nPrevious response failed the required schema. Return the exact JSON schema, no extra text."
            raw = self.call(role, prompt + suffix, request_id + f"/attempt-{attempt}", max_tokens=max_tokens,
                            temperature=temperature, top_p=top_p, seed_key=seed_key, retry_count=attempt)
            try:
                value = parser(raw)
            except (ValueError, TypeError, KeyError) as exc:
                value = None
                self.requests[-1]["parse_error"] = f"{type(exc).__name__}: {exc}"
            self.requests[-1]["structural_valid"] = value is not None
            if value is None:
                self.requests[-1].setdefault("parse_error", "Required fields, types, evidence or JSON envelope invalid")
            elif isinstance(value, str):
                self.requests[-1]["accepted_text"] = value
                from evaluation.revision_schema import strict_json
                envelope = strict_json(raw)
                original = next(iter(envelope.values())) if envelope and len(envelope) == 1 else raw
                self.requests[-1]["turn_truncated"] = isinstance(original, str) and original.strip() != value
            if value is not None:
                return value
        raise StructuralFailure(f"Invalid {role} output after {self.retry_limit + 1} attempts: {request_id}")

    def costs(self) -> dict[str, Any]:
        def total(key: str) -> int:
            return sum(int(r[key]) for r in self.requests if r.get(key) is not None)
        by_role = {}
        for role in sorted({r["role"] for r in self.requests}):
            rows = [r for r in self.requests if r["role"] == role]
            by_role[role] = {"requests": len(rows),
                "input_tokens": sum(int(r["input_tokens"]) for r in rows if r.get("input_tokens") is not None),
                "output_tokens": sum(int(r["output_tokens"]) for r in rows if r.get("output_tokens") is not None),
                "latency_ms": sum(r["latency_ms"] for r in rows),
                "failed_requests": sum(r["status"] == "failed" for r in rows),
                "invalid_outputs": sum(r.get("structural_valid") is False for r in rows),
                "factual_rejections": sum(r.get("factual_valid") is False for r in rows)}
        return {"requests": len(self.requests), "input_tokens": total("input_tokens"), "output_tokens": total("output_tokens"),
                "usage_complete": all(r.get("input_tokens") is not None and r.get("output_tokens") is not None for r in self.requests),
                "request_latency_ms": sum(r["latency_ms"] for r in self.requests), "by_role": by_role,
                "retry_requests": sum(r["retry_count"] > 0 or bool(re.search(r"/factual-attempt-[1-9][0-9]*/", r["request_id"])) for r in self.requests),
                "structural_retry_requests": sum(r["retry_count"] > 0 for r in self.requests),
                "factual_retry_requests": sum(bool(re.search(r"/factual-attempt-[1-9][0-9]*/", r["request_id"])) for r in self.requests)}
