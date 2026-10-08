#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from repro.profile import add_profile_argument, configure_profile_from_argv

configure_profile_from_argv()
from repro.data import prepare_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Build isolated paper-protocol datasets.")
    add_profile_argument(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_all(seed=args.seed, force=args.force), indent=2))


if __name__ == "__main__":
    main()
