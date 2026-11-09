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
SAMPLING_RATE = 0.5
TIME_HORIZON = 1.0
INITIAL_STATE = "A"
NUM_TREES = 100
MIN_A_TO_LEAF_PROBABILITY = 0.0700001
ROOT_PROBABILITY_WEIGHTS_BY_TRANSITION_COUNT = {
    2: (2.0, 1.0, 1.0, 1.0),
    3: (3.0, 1.0, 1.0, 1.0),
}


@dataclass(frozen=True)
class Setting:
    birth_rates: dict[str, int]
    death_rates: dict[str, int]
    transition_times: tuple[float, ...]
    transition_probabilities: dict[str, dict[str, float]]


def _sample_weighted_probabilities(
    rng: random.Random,
    weights: tuple[float, ...],
) -> tuple[float, ...]:
    raw_probabilities = [weight * rng.random() for weight in weights]
    total = math.fsum(raw_probabilities)
    while total == 0:
        raw_probabilities = [weight * rng.random() for weight in weights]
        total = math.fsum(raw_probabilities)

    probabilities = [value / total for value in raw_probabilities]
    probabilities[-1] = 1.0 - math.fsum(probabilities[:-1])
    return tuple(probabilities)


def sample_setting(rng: random.Random, num_transition_times: int = 5) -> Setting:
    birth_rates = {state: rng.choice(BIRTH_RATE_CHOICES) for state in STATES}
    death_rates = {state: rng.choice(DEATH_RATE_CHOICES) for state in STATES}
    transition_times = tuple(
        sorted(
            time / 1000
            for time in rng.sample(range(1, 1000), num_transition_times)
        )
    )

    if num_transition_times in ROOT_PROBABILITY_WEIGHTS_BY_TRANSITION_COUNT:
        root_probabilities = dict(
            zip(
                STATES,
                _sample_weighted_probabilities(
                    rng,
                    ROOT_PROBABILITY_WEIGHTS_BY_TRANSITION_COUNT[
                        num_transition_times
                    ],
                ),
            )
        )
    else:
        probability_a_to_a = rng.uniform(0.4, 0.6)
        split_points = sorted((rng.random(), rng.random()))
        leaf_probabilities = (
            split_points[0],
            split_points[1] - split_points[0],
            1.0 - split_points[1],
        )
        remaining_probability = (
            1.0
            - probability_a_to_a
            - len(STATES[1:]) * MIN_A_TO_LEAF_PROBABILITY
        )
        root_probabilities = {
            "A": probability_a_to_a,
            **{
                state: MIN_A_TO_LEAF_PROBABILITY
                + remaining_probability * probability
                for state, probability in zip(STATES[1:], leaf_probabilities)
            },
        }
    transition_probabilities = {
        "A": root_probabilities,
        "B": {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0},
        "C": {"A": 0.0, "B": 0.0, "C": 1.0, "D": 0.0},
        "D": {"A": 0.0, "B": 0.0, "C": 0.0, "D": 1.0},
    }
    return Setting(
        birth_rates=birth_rates,
        death_rates=death_rates,
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
        "A " + " ".join(f"{time:.3f}" for time in setting.transition_times),
        "B -",
        "C -",
        "D -",
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
        INITIAL_STATE,
        "",
        "# num trees",
        str(NUM_TREES),
        "",
    ]
    return "\n".join(lines)


def _read_setting(setting_path: Path) -> Setting:
    from src.analyses.simulation.simulate_coordinate_tree import read_simulation_settings

    parsed = read_simulation_settings(setting_path)
    states = parsed["states"]
    transition_probabilities = parsed["transition_probabilities"]
    return Setting(
        birth_rates={
            state: int(parsed["birth_rates"][state])
            for state in states
        },
        death_rates={
            state: int(parsed["death_rates"][state])
            for state in states
        },
        transition_times=tuple(parsed["transition_times"][INITIAL_STATE]),
        transition_probabilities={
            source: {
                target: float(transition_probabilities.loc[source, target])
                for target in states
            }
            for source in states
        },
    )


def generate_settings(
    output_dir: Path,
    num_settings: int = 100,
    start_id: int = 1,
    seed: int | None = None,
    num_transition_times: int = 5,
    resume_existing: bool = False,
    overwrite_existing: bool = False,
) -> int:
    if num_settings <= 0:
        raise ValueError("num_settings must be positive")
    if start_id < 0:
        raise ValueError("start_id must be non-negative")
    if not 1 <= num_transition_times <= 999:
        raise ValueError("num_transition_times must be between 1 and 999")
    if resume_existing and overwrite_existing:
        raise ValueError("resume_existing and overwrite_existing are mutually exclusive")

    setting_dirs = [
        output_dir / f"setting_{setting_id}"
        for setting_id in range(start_id, start_id + num_settings)
    ]
    manifest_path = output_dir / "generated_settings.csv"
    existing_dirs = [path for path in setting_dirs if path.exists()]
    if existing_dirs and not (resume_existing or overwrite_existing):
        raise FileExistsError(
            "Refusing to overwrite existing settings: "
            + ", ".join(str(path) for path in existing_dirs[:5])
        )
    if manifest_path.exists() and not (resume_existing or overwrite_existing):
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    if manifest_path.exists() and resume_existing and not existing_dirs:
        raise FileExistsError(
            f"Cannot resume {output_dir}: manifest exists but no setting directories do"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "setting",
        *(f"birth_{state}" for state in STATES),
        *(f"death_{state}" for state in STATES),
        *(f"p_A_to_{state}" for state in STATES),
        *(
            f"transition_time_{index}"
            for index in range(1, num_transition_times + 1)
        ),
        "sampling_rate",
        "num_trees",
    ]
    rng = random.Random(seed)
    manifest_rows = []
    for setting_id, setting_dir in enumerate(setting_dirs, start=start_id):
        setting = sample_setting(rng, num_transition_times)
        if setting_dir.exists():
            setting_path = setting_dir / "setting.txt"
            if overwrite_existing:
                if setting_path.exists() and not setting_path.is_file():
                    raise FileExistsError(
                        f"Cannot overwrite non-file setting path {setting_path}"
                    )
                if not setting_path.exists() and any(setting_dir.iterdir()):
                    raise FileExistsError(
                        f"Cannot overwrite incomplete setting directory {setting_dir}"
                    )
                setting_path.write_text(render_setting(setting))
            elif setting_path.is_file():
                setting = _read_setting(setting_path)
                if len(setting.transition_times) != num_transition_times:
                    raise ValueError(
                        f"Existing setting {setting_path} has "
                        f"{len(setting.transition_times)} root transition times; "
                        f"expected {num_transition_times}"
                    )
            elif any(setting_dir.iterdir()):
                raise FileExistsError(
                    f"Cannot resume incomplete setting directory {setting_dir}"
                )
            else:
                (setting_dir / "setting.txt").write_text(render_setting(setting))
        else:
            setting_dir.mkdir()
            (setting_dir / "setting.txt").write_text(render_setting(setting))
        manifest_rows.append({
            "setting": f"setting_{setting_id}",
            **{f"birth_{state}": setting.birth_rates[state] for state in STATES},
            **{f"death_{state}": setting.death_rates[state] for state in STATES},
            **{
                f"p_A_to_{state}": setting.transition_probabilities["A"][state]
                for state in STATES
            },
            **{
                f"transition_time_{index}": time
                for index, time in enumerate(setting.transition_times, start=1)
            },
            "sampling_rate": SAMPLING_RATE,
            "num_trees": NUM_TREES,
        })

    with manifest_path.open("w", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return num_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random settings for a four-state star system."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/simulation/coord_diff/star"),
        help="Directory in which setting_N folders will be created.",
    )
    parser.add_argument(
        "--num-settings",
        type=int,
        default=100,
        help="Number of random settings to generate.",
    )
    parser.add_argument(
        "--num-transition-times",
        type=int,
        default=5,
        help="Number of transition times to include in each setting.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse existing settings with the requested root transition count.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Regenerate setting files and manifest, replacing existing settings.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="First setting index; defaults to 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible settings.",
    )
    args = parser.parse_args()

    count = generate_settings(
        args.output_dir,
        num_settings=args.num_settings,
        start_id=args.start_id,
        seed=args.seed,
        num_transition_times=args.num_transition_times,
        resume_existing=args.resume_existing,
        overwrite_existing=args.overwrite_existing,
    )
    print(f"Generated {count} settings in {args.output_dir}")


if __name__ == "__main__":
    main()
