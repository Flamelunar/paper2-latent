"""Select a data-scale profile before importing profile-dependent modules."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any


PROFILE_CHOICES = ("paper", "paper10k", "small5000")


def configure_profile_from_argv(
    argv: list[str] | None = None, *, default_profile: str = "paper"
) -> str:
    """Read ``--profile`` early so constants resolve to the intended run root."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", choices=PROFILE_CHOICES)
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    profile = args.profile or os.getenv("DAILYPERSUASION_PROFILE", default_profile)
    os.environ["DAILYPERSUASION_PROFILE"] = profile
    return profile


def add_profile_argument(parser: Any) -> None:
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default=os.environ.get("DAILYPERSUASION_PROFILE", "paper"),
        help=(
            "Dataset/run profile. paper10k isolates the paper-scale 10,000/2,000/1,000 run; "
            "small5000 uses 5,000/1,000/500 scenarios."
        ),
    )
