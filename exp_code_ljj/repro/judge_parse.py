"""Order-swapped pairwise Judge response parser copied from the original evaluator."""
from __future__ import annotations
import json
import re

def parse_winner(raw: str) -> str:
    """Parse strict JSON first, then tolerate local-model explanation wrappers."""
    text = raw.strip()
    candidates: list[str] = []
    for match in re.finditer(r"\{.*?\}", text, flags=re.S):
        candidates.append(match.group(0))
    candidates.append(text)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        winner = str(value.get("winner", "")).strip().upper()
        if winner in {"A", "B"}:
            return winner
    labelled = re.search(r"(?:winner|choice|answer)\s*[:=]\s*\**\s*([AB])\b", text, re.I)
    if labelled:
        return labelled.group(1).upper()
    # Last-resort parser for a local judge that ends with a one-letter decision.
    matches = re.findall(r"\b(?:Response\s*)?([AB])\b", text, flags=re.I)
    if matches:
        return matches[-1].upper()
    raise ValueError(f"Could not parse judge winner from: {raw[:500]}")
