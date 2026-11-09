from __future__ import annotations

import argparse
from pathlib import Path

from src.analyses.simulation.generate_star_coord_settings import generate_settings


DEFAULT_OUTPUT_ROOT = Path("data/simulation/coord_diff")
DEFAULT_NUM_SETTINGS = 100


def generate_star_transition_settings(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    num_settings: int = DEFAULT_NUM_SETTINGS,
    overwrite_existing: bool = False,
) -> None:
    generate_settings(
        output_root / "star_3",
        num_settings=num_settings,
        seed=3,
        num_transition_times=3,
        resume_existing=not overwrite_existing,
        overwrite_existing=overwrite_existing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate star_3 simulator settings."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory for the star_3 settings.",
    )
    parser.add_argument(
        "--num-settings",
        type=int,
        default=DEFAULT_NUM_SETTINGS,
        help="Number of settings to generate in the star_3 directory.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Regenerate setting files using the current probability weights.",
    )
    args = parser.parse_args()
    generate_star_transition_settings(
        output_root=args.output_root,
        num_settings=args.num_settings,
        overwrite_existing=args.overwrite_existing,
    )


if __name__ == "__main__":
    main()
