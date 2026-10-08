#!/usr/bin/env python3
"""Export the three LoRA adapters as standalone, directly loadable models."""

from __future__ import annotations

import argparse
from pathlib import Path

from repro.profile import add_profile_argument, configure_profile_from_argv

configure_profile_from_argv()
from repro.constants import OUTPUT_DIR
from repro.export import export_merged_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_profile_argument(parser)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any available adapter cannot be merged.",
    )
    args = parser.parse_args()
    statuses = export_merged_models(
        output_dir=args.output_dir,
        force=args.force,
        strict=args.strict,
    )
    for name, status in statuses.items():
        print(f"{name}: {status}")


if __name__ == "__main__":
    main()
