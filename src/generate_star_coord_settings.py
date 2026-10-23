from __future__ import annotations

import argparse
import csv
from pathlib import Path


STATES = ("A", "B", "C", "D")
BIRTH_RATES = {"A": 6, "B": 8, "C": 10, "D": 12}
DEATH_RATES = {"A": 1, "B": 2, "C": 3, "D": 4}
SAMPLING_RATE = 0.5
TRANSITION_TIME = 0.5
TIME_HORIZON = 1.0
INITIAL_STATE = "A"
NUM_TREES = 100
TRANSITION_PROBABILITIES = {
    "A": {"A": 0.0, "B": 1 / 3, "C": 1 / 3, "D": 1 / 3},
    "B": {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0},
    "C": {"A": 0.0, "B": 0.0, "C": 1.0, "D": 0.0},
    "D": {"A": 0.0, "B": 0.0, "C": 0.0, "D": 1.0},
}


def render_setting() -> str:
    lines = [
        f"{len(STATES)} # num states",
        f"{' '.join(STATES)} # states",
        "",
        "# birth rates",
        *(f"{state} {BIRTH_RATES[state]}" for state in STATES),
        "",
        "# death rates",
        *(f"{state} {DEATH_RATES[state]}" for state in STATES),
        "",
        "# sampling rates",
        *(f"{state} {SAMPLING_RATE}" for state in STATES),
        "",
        "# transition times (use - if a state has none)",
        f"A {TRANSITION_TIME}",
        "B -",
        "C -",
        "D -",
        "",
        "# transition probabilities at each transition time",
        *(
            f"{source} -> {target} {TRANSITION_PROBABILITIES[source][target]}"
            for source in STATES
            for target in STATES
        ),
        "",
        "# simulation time horizon",
        str(TIME_HORIZON),
        "",
        "# initial state",
        INITIAL_STATE,
        "",
        "# num trees",
        str(NUM_TREES),
        "",
    ]
    return "\n".join(lines)


def generate_settings(output_dir: Path, start_id: int = 1) -> int:
    if start_id < 0:
        raise ValueError("start_id must be non-negative")

    setting_dir = output_dir / f"setting_{start_id}"
    manifest_path = output_dir / "generated_settings.csv"
    if setting_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing settings: {setting_dir}"
        )
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    setting_dir.mkdir()
    (setting_dir / "setting.txt").write_text(render_setting())

    fieldnames = [
        "setting",
        *(f"birth_{state}" for state in STATES),
        *(f"death_{state}" for state in STATES),
        *(f"p_A_to_{state}" for state in STATES),
        "transition_time",
        "sampling_rate",
        "num_trees",
    ]
    row = {
        "setting": f"setting_{start_id}",
        **{f"birth_{state}": BIRTH_RATES[state] for state in STATES},
        **{f"death_{state}": DEATH_RATES[state] for state in STATES},
        **{
            f"p_A_to_{state}": TRANSITION_PROBABILITIES["A"][state]
            for state in STATES
        },
        "transition_time": TRANSITION_TIME,
        "sampling_rate": SAMPLING_RATE,
        "num_trees": NUM_TREES,
    }
    with manifest_path.open("w", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a four-state star-system setting."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/simulation/coord_diff_star_4state"),
        help="Directory in which the setting_N folder will be created.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="Setting index to create; defaults to 1.",
    )
    args = parser.parse_args()

    count = generate_settings(args.output_dir, start_id=args.start_id)
    print(f"Generated {count} setting in {args.output_dir}")


if __name__ == "__main__":
    main()
