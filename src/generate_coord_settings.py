from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterator


STATES = ("A", "B")
BIRTH_RATES = (6, 8, 10, 12)
DEATH_RATES = (1, 2, 3, 4)
TRANSITION_PROBABILITIES = (0.25, 0.5, 0.75)
SAMPLING_RATE = 0.5
TRANSITION_TIME = 0.5
TIME_HORIZON = 1.0
INITIAL_STATE = "A"
NUM_TREES = 100


@dataclass(frozen=True)
class Setting:
    birth_a: int
    birth_b: int
    death_a: int
    death_b: int
    probability_a_to_b: float


def iter_settings() -> Iterator[Setting]:
    for birth_a, birth_b, death_a, death_b, probability_a_to_b in product(
        BIRTH_RATES,
        BIRTH_RATES,
        DEATH_RATES,
        DEATH_RATES,
        TRANSITION_PROBABILITIES,
    ):
        yield Setting(
            birth_a=birth_a,
            birth_b=birth_b,
            death_a=death_a,
            death_b=death_b,
            probability_a_to_b=probability_a_to_b,
        )


def render_setting(setting: Setting) -> str:
    probability_a_to_a = 1.0 - setting.probability_a_to_b
    return f"""\
2 # num states
A B # states

# birth rates
A {setting.birth_a}
B {setting.birth_b}

# death rates
A {setting.death_a}
B {setting.death_b}

# sampling rates
A {SAMPLING_RATE}
B {SAMPLING_RATE}

# transition times (use - if a state has none)
A {TRANSITION_TIME}
B -

# transition probabilities at each transition time
A -> A {probability_a_to_a:g}
A -> B {setting.probability_a_to_b:g}
B -> A 0.0
B -> B 1.0

# simulation time horizon
{TIME_HORIZON}

# initial state
{INITIAL_STATE}

# num trees
{NUM_TREES}
"""


def generate_settings(output_dir: Path, start_id: int = 1) -> int:
    if start_id < 0:
        raise ValueError("start_id must be non-negative")

    settings = list(iter_settings())
    setting_dirs = [
        output_dir / f"setting_{setting_id}"
        for setting_id in range(start_id, start_id + len(settings))
    ]
    manifest_path = output_dir / "generated_settings.csv"

    existing_dirs = [path for path in setting_dirs if path.exists()]
    if existing_dirs:
        raise FileExistsError(
            "Refusing to overwrite existing settings: "
            + ", ".join(str(path) for path in existing_dirs[:5])
        )
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for setting_id, (setting, setting_dir) in enumerate(
        zip(settings, setting_dirs),
        start=start_id,
    ):
        setting_dir.mkdir()
        (setting_dir / "setting.txt").write_text(render_setting(setting))
        manifest_rows.append(
            {
                "setting": f"setting_{setting_id}",
                "birth_A": setting.birth_a,
                "birth_B": setting.birth_b,
                "death_A": setting.death_a,
                "death_B": setting.death_b,
                "p_A_to_A": 1.0 - setting.probability_a_to_b,
                "p_A_to_B": setting.probability_a_to_b,
                "transition_time": TRANSITION_TIME,
            }
        )

    with manifest_path.open("w", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    return len(settings)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the two-state, single-transition setting sweep."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/simulation/coord_diff"),
        help="Directory in which setting_N folders will be created.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="First setting_N index; defaults to 1 to preserve setting_0.",
    )
    args = parser.parse_args()

    count = generate_settings(args.output_dir, start_id=args.start_id)
    print(f"Generated {count} settings in {args.output_dir}")


if __name__ == "__main__":
    main()
