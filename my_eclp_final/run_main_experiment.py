#!/usr/bin/env python3
"""Single entrypoint for the ECLP+ELPO main experiment.
All data, prompts, Persuadee calls and evaluation are delegated to exp_code_ljj.
Run from any directory, e.g. ``python my_eclp_final/run_main_experiment.py audit``.
"""
from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
# Keep the implementation in one place and bind it to exp_code_ljj via config.repo_root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eclp_elpo import main
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
