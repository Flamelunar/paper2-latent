#!/usr/bin/env python3
"""Baseline 1: generate with an untrained Persuader on the shared test protocol.

There is no training stage. Extra arguments are forwarded to
run_baseline_generation.py, including --base-model and GPU settings.
"""

from __future__ import annotations

import sys

from run_baseline_generation import main


if __name__ == "__main__":
    main([*sys.argv[1:], "--method", "baseline1_zero_shot"])
