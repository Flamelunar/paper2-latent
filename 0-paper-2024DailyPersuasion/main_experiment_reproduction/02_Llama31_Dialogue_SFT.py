#!/usr/bin/env python3
"""Main route 1: local Llama 3.1 8B supervised fine-tuning on dialogue only."""

from __future__ import annotations

import argparse

from repro.profile import add_profile_argument, configure_profile_from_argv

ACTIVE_PROFILE = configure_profile_from_argv()
from repro.train_backend import add_train_arguments, add_sft_continuation_arguments, spec_from_args, train_sft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_profile_argument(parser)
    add_train_arguments(parser)
    add_sft_continuation_arguments(parser)
    parser.set_defaults(epochs=2.0 if ACTIVE_PROFILE == "small5000" else 1.0)
    args = parser.parse_args()
    output = train_sft(spec_from_args(args), with_isr=False, dry_run=args.dry_run)
    print(output)


if __name__ == "__main__":
    main()
