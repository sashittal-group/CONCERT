from __future__ import annotations

import argparse
from pathlib import Path

from src.analyses.simulation.simulate_coordinate_tree import read_simulation_settings


def extract_transition_times(setting_file: Path, output_file: Path) -> list[float]:
    settings = read_simulation_settings(setting_file)
    simulation_times = settings["transition_times"]["A"]
    if not simulation_times:
        raise ValueError(f"No transition times for root state A in {setting_file}")

    transition_times = sorted(
        1.0 - time / settings["time_horizon"] for time in simulation_times
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "".join(f"{transition_time:.12g}\n" for transition_time in transition_times)
    )
    return transition_times


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract root-state transition times for star-tree inference."
    )
    parser.add_argument("setting_file", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()

    transition_times = extract_transition_times(args.setting_file, args.output_file)
    print(f"Wrote {len(transition_times)} transition times to {args.output_file}")


if __name__ == "__main__":
    main()
