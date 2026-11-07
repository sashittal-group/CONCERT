from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path


STATES = ("A", "B", "C", "D")
BIRTH_RATE_CHOICES = (6, 8, 10, 12)
DEATH_RATE_CHOICES = (1, 2, 3, 4)
TRANSITION_WEIGHTS = {
    "A": (1.0, 3.0),
    "B": (1.0, 2.0),
    "C": (1.0, 1.0),
}
SAMPLING_RATE = 0.5
TIME_HORIZON = 1.0
NUM_TREES = 100


@dataclass(frozen=True)
class Setting:
    birth_rates: dict[str, int]
    death_rates: dict[str, int]
    transition_times: dict[str, float | None]
    transition_probabilities: dict[str, dict[str, float]]


def _sample_weighted_pair(
    rng: random.Random,
    weights: tuple[float, float],
) -> tuple[float, float]:
    """Draw normalized stay/transition probabilities using weights as concentrations."""
    draws = [rng.gammavariate(weight, 1.0) for weight in weights]
    total = math.fsum(draws)
    return draws[0] / total, draws[1] / total


def sample_setting(rng: random.Random) -> Setting:
    ordered_times = sorted(time / 1000 for time in rng.sample(range(1, 1000), 3))
    transition_times = {
        "A": ordered_times[0],
        "B": ordered_times[1],
        "C": ordered_times[2],
        "D": None,
    }

    transition_probabilities = {
        source: {
            target: float(source == target)
            for target in STATES
        }
        for source in STATES
    }
    for source, target in zip(STATES[:-1], STATES[1:]):
        stay_probability, transition_probability = _sample_weighted_pair(
            rng,
            TRANSITION_WEIGHTS[source],
        )
        transition_probabilities[source][source] = stay_probability
        transition_probabilities[source][target] = transition_probability

    return Setting(
        birth_rates={
            state: rng.choice(BIRTH_RATE_CHOICES)
            for state in STATES
        },
        death_rates={
            state: rng.choice(DEATH_RATE_CHOICES)
            for state in STATES
        },
        transition_times=transition_times,
        transition_probabilities=transition_probabilities,
    )


def render_setting(setting: Setting) -> str:
    lines = [
        f"{len(STATES)} # num states",
        f"{' '.join(STATES)} # states",
        "",
        "# birth rates",
        *(f"{state} {setting.birth_rates[state]}" for state in STATES),
        "",
        "# death rates",
        *(f"{state} {setting.death_rates[state]}" for state in STATES),
        "",
        "# sampling rates",
        *(f"{state} {SAMPLING_RATE}" for state in STATES),
        "",
        "# transition times (use - if a state has none)",
        *(
            f"{state} {setting.transition_times[state]:.3f}"
            if setting.transition_times[state] is not None
            else f"{state} -"
            for state in STATES
        ),
        "",
        "# transition probabilities at each transition time",
        *(
            f"{source} -> {target} "
            f"{setting.transition_probabilities[source][target]:.17g}"
            for source in STATES
            for target in STATES
        ),
        "",
        "# simulation time horizon",
        str(TIME_HORIZON),
        "",
        "# initial state",
        "A",
        "",
        "# num trees",
        str(NUM_TREES),
        "",
    ]
    return "\n".join(lines)


def generate_settings(
    output_dir: Path,
    num_settings: int = 100,
    start_id: int = 1,
    seed: int = 3,
) -> int:
    if num_settings <= 0:
        raise ValueError("num_settings must be positive")
    if start_id < 0:
        raise ValueError("start_id must be non-negative")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for setting_id in range(start_id, start_id + num_settings):
        setting = sample_setting(random.Random(seed + setting_id))
        setting_dir = output_dir / f"setting_{setting_id}"
        setting_dir.mkdir(parents=True, exist_ok=True)
        (setting_dir / "setting.txt").write_text(render_setting(setting))
        manifest_rows.append(
            {
                "setting": f"setting_{setting_id}",
                **{
                    f"birth_{state}": setting.birth_rates[state]
                    for state in STATES
                },
                **{
                    f"death_{state}": setting.death_rates[state]
                    for state in STATES
                },
                "p_A_to_A": setting.transition_probabilities["A"]["A"],
                "p_A_to_B": setting.transition_probabilities["A"]["B"],
                "p_B_to_B": setting.transition_probabilities["B"]["B"],
                "p_B_to_C": setting.transition_probabilities["B"]["C"],
                "p_C_to_C": setting.transition_probabilities["C"]["C"],
                "p_C_to_D": setting.transition_probabilities["C"]["D"],
                **{
                    f"transition_time_{state}": setting.transition_times[state]
                    for state in ("A", "B", "C")
                },
                "sampling_rate": SAMPLING_RATE,
                "num_trees": NUM_TREES,
            }
        )

    manifest_path = output_dir / "generated_settings.csv"
    with manifest_path.open("w", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    return num_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate settings for the A -> B -> C -> D transition graph."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/simulation/coord_diff/linear_A_to_D"),
        help="Directory in which setting_N folders will be created.",
    )
    parser.add_argument(
        "--num-settings",
        type=int,
        default=100,
        help="Number of random settings to generate.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="First setting index.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="Seed for reproducible setting generation.",
    )
    args = parser.parse_args()

    count = generate_settings(
        args.output_dir,
        num_settings=args.num_settings,
        start_id=args.start_id,
        seed=args.seed,
    )
    print(f"Generated {count} settings in {args.output_dir}")


if __name__ == "__main__":
    main()
