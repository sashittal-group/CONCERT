from __future__ import annotations

import argparse
from pathlib import Path

from src.generate_star_coord_settings import generate_settings


DEFAULT_OUTPUT_ROOT = Path("data/simulation/coord_diff")
DEFAULT_NUM_SETTINGS = 100


def generate_star_transition_settings(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    num_settings: int = DEFAULT_NUM_SETTINGS,
    overwrite_existing: bool = False,
) -> None:
    for num_transition_times in (2, 3):
        generate_settings(
            output_root / f"star_{num_transition_times}",
            num_settings=num_settings,
            seed=num_transition_times,
            num_transition_times=num_transition_times,
            resume_existing=not overwrite_existing,
            overwrite_existing=overwrite_existing,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate separate star_2 and star_3 simulator settings."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory for the star_2 and star_3 settings.",
    )
    parser.add_argument(
        "--num-settings",
        type=int,
        default=DEFAULT_NUM_SETTINGS,
        help="Number of settings to generate in each directory.",
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
