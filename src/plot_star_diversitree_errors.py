from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.plot_star_inference_accuracy import (
    STATES,
    _add_identifiable_parameters,
    _read_ground_truth,
    load_completed_progressive_comparisons,
)


DEFAULT_DATA_ROOT = Path("data/simulation/coord_diff")
METHODS = ("Coordinated", "diversitree")
METHOD_COLORS = {"Coordinated": "#0072B2", "diversitree": "#E69F00"}
METRICS = (
    ("Net growth", "net_growth"),
    ("Scaled birth", "scaled_birth"),
)


@dataclass(frozen=True)
class AbsoluteError:
    setting: str
    batch: str
    state: str
    method: str
    metric: str
    value: float


def _read_diversitree_estimates(
    summary_path: Path,
) -> tuple[dict[str, float], tuple[float, ...]]:
    summary = {
        key: value
        for line in summary_path.read_text().splitlines()
        if "=" in line
        for key, value in [line.split("=", maxsplit=1)]
    }
    if summary.get("convergence_code") != "0":
        raise ValueError(f"Expected a converged diversitree fit in {summary_path}")

    try:
        sampling_fractions = tuple(
            float(value) for value in summary["sampling_fraction"].split(",")
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid sampling fractions in {summary_path}") from error
    if len(sampling_fractions) != len(STATES):
        raise ValueError(f"Expected four sampling fractions in {summary_path}")

    estimates_path = summary_path.with_name("parameter_estimates.csv")
    with estimates_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if any(not row.get("parameter") or not row.get("estimate") for row in rows):
        raise ValueError(f"Invalid parameter estimate row in {estimates_path}")
    estimates = {row["parameter"]: float(row["estimate"]) for row in rows}
    if len(estimates) != len(rows):
        raise ValueError(f"Duplicate parameter names in {estimates_path}")
    for parameter in (
        *(f"lambda{index}" for index in range(1, len(STATES) + 1)),
        *(f"mu{index}" for index in range(1, len(STATES) + 1)),
    ):
        if parameter not in estimates:
            raise ValueError(f"Missing {parameter} in {estimates_path}")
    return estimates, sampling_fractions


def _derived_rates(
    birth: tuple[float, ...],
    death: tuple[float, ...],
    sampling: tuple[float, ...],
) -> dict[str, tuple[float, ...]]:
    return {
        "net_growth": tuple(
            birth_rate - death_rate
            for birth_rate, death_rate in zip(birth, death)
        ),
        "scaled_birth": tuple(
            birth_rate * sampling_rate
            for birth_rate, sampling_rate in zip(birth, sampling)
        ),
    }


def load_matched_absolute_errors(
    data_dir: Path,
    num_transition_times: int = 2,
) -> list[AbsoluteError]:
    truth_by_setting = _read_ground_truth(
        data_dir / "generated_settings.csv",
        num_transition_times=num_transition_times,
    )
    truth_by_setting = {
        setting: _add_identifiable_parameters(values)
        for setting, values in truth_by_setting.items()
    }
    coordinated = {
        (comparison.setting, comparison.batch): comparison
        for comparison in load_completed_progressive_comparisons(
            data_dir,
            num_transition_times=num_transition_times,
        )
    }

    errors = []
    for summary_path in sorted(
        data_dir.glob("setting_*/batch_*/diversitree_musse/joint_musse_fit.txt")
    ):
        summary_values = {
            key: value
            for line in summary_path.read_text().splitlines()
            if "=" in line
            for key, value in [line.split("=", maxsplit=1)]
        }
        if summary_values.get("convergence_code") != "0":
            continue

        batch_dir = summary_path.parent.parent
        setting = batch_dir.parent.name
        batch = batch_dir.name
        key = (setting, batch)
        comparison = coordinated.get(key)
        truth = truth_by_setting.get(setting)
        if comparison is None or truth is None:
            continue

        estimates, sampling_fractions = _read_diversitree_estimates(summary_path)
        diversitree_rates = _derived_rates(
            tuple(estimates[f"lambda{index}"] for index in range(1, 5)),
            tuple(estimates[f"mu{index}"] for index in range(1, 5)),
            sampling_fractions,
        )
        coordinated_values = comparison.inferred
        coordinated_rates = _derived_rates(
            tuple(coordinated_values[f"birth_{state}"] for state in STATES),
            tuple(coordinated_values[f"death_{state}"] for state in STATES),
            tuple(coordinated_values[f"sampling_{state}"] for state in STATES),
        )
        truth_rates = {
            "net_growth": tuple(
                truth[f"net_growth_{state}"] for state in STATES
            ),
            "scaled_birth": tuple(
                truth[f"birth_{state}"] * truth[f"sampling_{state}"]
                for state in STATES
            ),
        }

        for method, rates in (
            ("Coordinated", coordinated_rates),
            ("diversitree", diversitree_rates),
        ):
            for label, metric in METRICS:
                for state_index, state in enumerate(STATES):
                    errors.append(
                        AbsoluteError(
                            setting=setting,
                            batch=batch,
                            state=state,
                            method=method,
                            metric=metric,
                            value=abs(
                                rates[metric][state_index]
                                - truth_rates[metric][state_index]
                            ),
                        )
                    )

    if not errors:
        raise ValueError(
            f"No matched, converged coordinated/diversitree fits found in {data_dir}"
        )
    return errors


def plot_error_boxplots(
    errors: list[AbsoluteError],
    output_path: Path,
    num_transition_times: int = 2,
) -> Path:
    matched_batches = {
        (error.setting, error.batch)
        for error in errors
        if error.method == "Coordinated"
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    offsets = {"Coordinated": -0.18, "diversitree": 0.18}

    for axis, (label, metric) in zip(axes, METRICS):
        for state_index, state in enumerate(STATES, start=1):
            for method in METHODS:
                values = [
                    error.value
                    for error in errors
                    if error.metric == metric
                    and error.state == state
                    and error.method == method
                ]
                if not values:
                    continue
                box = axis.boxplot(
                    [values],
                    positions=[state_index + offsets[method]],
                    widths=0.28,
                    patch_artist=True,
                    showfliers=True,
                    medianprops={"color": "#222222", "linewidth": 1.5},
                    whiskerprops={"color": METHOD_COLORS[method]},
                    capprops={"color": METHOD_COLORS[method]},
                    flierprops={
                        "marker": ".",
                        "markerfacecolor": METHOD_COLORS[method],
                        "markeredgecolor": METHOD_COLORS[method],
                        "alpha": 0.5,
                    },
                )
                box["boxes"][0].set_facecolor(METHOD_COLORS[method])
                box["boxes"][0].set_alpha(0.65)

        axis.set_xticks(range(1, len(STATES) + 1), STATES)
        axis.set_xlabel("State")
        axis.set_yscale("symlog", linthresh=0.1)
        axis.set_ylim(bottom=0)
        axis.set_ylabel(f"Absolute error in {label.lower()} (symlog scale)")
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)

    handles = [
        Patch(
            facecolor=METHOD_COLORS[method],
            edgecolor=METHOD_COLORS[method],
            alpha=0.65,
            label=method,
        )
        for method in METHODS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(METHODS),
    )
    figure.suptitle(
        f"Star_{num_transition_times} parameter errors by state "
        f"({len(matched_batches)} matched batches)\n"
        "Coordinated inference: first 8 trees; diversitree: all usable trees",
        y=1.04,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot state-wise absolute errors for net growth and scaled birth "
            "from matched star_2 coordinated and diversitree fits."
        )
    )
    parser.add_argument(
        "--num-transition-times",
        type=int,
        choices=(2, 3),
        default=2,
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir or (
        DEFAULT_DATA_ROOT / f"star_{args.num_transition_times}"
    )
    output_path = args.output or (
        data_dir / "figures" / "coordinated_vs_diversitree_error_boxplots.png"
    )
    errors = load_matched_absolute_errors(data_dir, args.num_transition_times)
    output_path = plot_error_boxplots(
        errors,
        output_path,
        args.num_transition_times,
    )
    matched_batches = {
        (error.setting, error.batch)
        for error in errors
        if error.method == "Coordinated"
    }
    print(f"Plotted {len(matched_batches)} matched batches to {output_path}")


if __name__ == "__main__":
    main()
