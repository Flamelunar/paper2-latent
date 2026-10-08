#!/usr/bin/env python3
"""Score fixed-Qwen simulation outputs with the paper's automatic metrics."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repro.profile import configure_profile_from_argv

configure_profile_from_argv(default_profile="small5000")

from repro.trajectory_evaluation import main


if __name__ == "__main__":
    main()
