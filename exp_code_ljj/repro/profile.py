"""Select a data-scale profile before importing profile-dependent modules."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any


PROFILE_CHOICES = ("paper10k",)


def configure_profile_from_argv(
    argv: list[str] | None = None, *, default_profile: str = "paper10k"
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
        default=os.environ.get("DAILYPERSUASION_PROFILE", "paper10k"),
        help=(
            "Fixed paper10k split shared by all baseline methods."
        ),
    )
