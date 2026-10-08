#!/usr/bin/env python3
"""Test the zero-shot Llama persuader with the fixed Qwen persuadee."""

from __future__ import annotations

from repro.persuadee_simulation import main as run_fixed_persuadee_test


def main() -> None:
    run_fixed_persuadee_test(forced_method="01_zero_shot")


if __name__ == "__main__":
    main()
