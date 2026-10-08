#!/usr/bin/env python3
"""Test all three persuaders through the shared fixed-Qwen multi-turn protocol."""

from __future__ import annotations

import sys
from pathlib import Path

# Running this file from tests/ must still resolve the project package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repro.profile import configure_profile_from_argv

configure_profile_from_argv(default_profile="small5000")

from repro.persuadee_simulation import main


if __name__ == "__main__":
    main()
