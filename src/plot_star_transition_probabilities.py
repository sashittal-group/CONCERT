from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.simulator_coord import read_simulation_settings


DEFAULT_SETTINGS_ROOT = Path("data/simulation/coord_diff")
DEFAULT_OUTPUT = DEFAULT_SETTINGS_ROOT / "star_transition_probability_boxplots.png"
TRANSITION_COUNTS = (2, 3)


def collect_root_transition_probabilities(
    settings_root: Path,
    num_transition_times: int,
) -> tuple[list[str], list[list[float]]]:
    root = settings_root / f"star_{num_transition_times}"
    setting_paths = sorted(
        root.glob("setting_*/setting.txt"),
        key=lambda path: int(path.parent.name.removeprefix("setting_")),
    )
    if not setting_paths:
        raise FileNotFoundError(f"No settings found under {root}")

    states: list[str] | None = None
    probabilities: list[list[float]] = []
    for setting_path in setting_paths:
        settings = read_simulation_settings(setting_path)
        actual_transition_count = len(
            settings["transition_times"][settings["root_state"]]
        )
        if actual_transition_count != num_transition_times:
            raise ValueError(
                f"{setting_path} has {actual_transition_count} root transition "
                f"times; expected {num_transition_times}"
            )
        setting_states = settings["states"]
        if states is None:
            states = setting_states
        elif setting_states != states:
            raise ValueError(f"State sets differ between settings under {root}")

        probabilities.append(
            [
                float(
                    settings["transition_probabilities"].loc[
                        settings["root_state"], state
                    ]
                )
                for state in states
            ]
        )

    if states is None:
        raise ValueError(f"No state data found under {root}")
    return states, probabilities


def plot_star_transition_probabilities(
    settings_root: Path = DEFAULT_SETTINGS_ROOT,
    output: Path = DEFAULT_OUTPUT,
) -> Path:
    fig, axes = plt.subplots(
        1,
        len(TRANSITION_COUNTS),
        figsize=(11, 5),
        sharey=True,
        constrained_layout=True,
    )
    colors = ("#4C72B0", "#DD8452")

    for ax, num_transition_times, color in zip(
        axes,
        TRANSITION_COUNTS,
        colors,
    ):
        states, probabilities = collect_root_transition_probabilities(
            settings_root,
            num_transition_times,
        )
        boxes = ax.boxplot(
            [
                [setting_probabilities[index] for setting_probabilities in probabilities]
                for index in range(len(states))
            ],
            patch_artist=True,
        )
        for box in boxes["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.7)
        ax.set_xticks(range(1, len(states) + 1))
        ax.set_xticklabels([f"A -> {state}" for state in states])
        ax.set_title(f"star_{num_transition_times}")
        ax.set_xlabel("Destination from root state A")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Transition probability")
    fig.suptitle("Root-state transition probabilities across settings")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot star_2 and star_3 transition-probability boxplots."
    )
    parser.add_argument(
        "--settings-root",
        type=Path,
        default=DEFAULT_SETTINGS_ROOT,
        help="Parent directory containing star_2 and star_3 settings.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the output PNG.",
    )
    args = parser.parse_args()
    output = plot_star_transition_probabilities(args.settings_root, args.output)
    print(f"Wrote transition-probability boxplots to {output}")


if __name__ == "__main__":
    main()
