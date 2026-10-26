from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STATES = ("A", "B", "C", "D")
SAMPLE_SIZES = (4, 8)
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
TRANSITION_LINE = re.compile(
    rf"(\d+)->(\d+)\s+({NUMBER})"
)
TIME_LINE = re.compile(rf"(\d+)\s+({NUMBER})")
PARAMETERS = [
    *((f"Birth {state}", f"birth_{state}") for state in STATES),
    *((f"Death {state}", f"death_{state}") for state in STATES),
    *((f"Sampling {state}", f"sampling_{state}") for state in STATES),
    *((f"A → {state}", f"p_A_to_{state}") for state in STATES),
]
TRANSITION_TIME_PARAMETERS = [
    (f"Transition time {index}", f"transition_time_{index}")
    for index in range(1, 6)
]
IDENTIFIABLE_PARAMETERS = [
    *((f"Net growth {state}", f"net_growth_{state}") for state in STATES),
    *((f"Scaled sampling {state}", f"scaled_sampling_{state}") for state in STATES),
    *((f"Transition probability A→{state}", f"p_A_to_{state}") for state in STATES),
    *TRANSITION_TIME_PARAMETERS,
]


@dataclass(frozen=True)
class Comparison:
    sample_size: int
    setting: str
    batch: str
    truth: dict[str, float]
    inferred: dict[str, float]
    source: Path


def _parse_vector(lines: list[str], name: str, path: Path) -> tuple[float, ...]:
    matches = [line for line in lines if line.startswith(f"{name}:")]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} vector in {path}")
    try:
        values = tuple(
            float(value) for value in matches[0].split(":", maxsplit=1)[1].split()
        )
    except ValueError as error:
        raise ValueError(f"Invalid {name} vector in {path}") from error
    if len(values) != len(STATES) or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Expected four finite {name} values in {path}")
    return values


def _parse_output(path: Path) -> dict[str, float]:
    lines = path.read_text().splitlines()
    if not any(line.startswith("LL=") for line in lines):
        raise ValueError(f"Missing likelihood line in {path}")

    vectors = {
        name: _parse_vector(lines, name, path)
        for name in ("birth", "death", "sampling")
    }
    transitions: dict[tuple[int, int], float] = {}
    transition_times: dict[str, float] = {}
    in_transitions = False
    in_transition_times = False
    found_transitions = False
    for line in lines:
        stripped = line.strip()
        if stripped == "transitions:":
            if found_transitions:
                raise ValueError(f"Multiple transition sections in {path}")
            found_transitions = True
            in_transitions = True
            in_transition_times = False
            continue
        if stripped == "transition_times:":
            in_transitions = False
            in_transition_times = True
            continue
        if in_transition_times and stripped:
            match = TIME_LINE.fullmatch(stripped)
            if match is None:
                continue
            index, raw_value = match.groups()
            key = f"transition_time_{int(index) + 1}"
            value = float(raw_value)
            if key in transition_times or not math.isfinite(value):
                raise ValueError(f"Invalid transition time in {path}: {stripped}")
            transition_times[key] = value
            continue
        if not in_transitions or not stripped:
            continue
        match = TRANSITION_LINE.fullmatch(stripped)
        if match is None:
            continue
        source, target, raw_value = match.groups()
        key = (int(source), int(target))
        value = float(raw_value)
        if key in transitions or not math.isfinite(value):
            raise ValueError(f"Invalid transition probability in {path}: {stripped}")
        transitions[key] = value

    required_transitions = {(0, target) for target in range(len(STATES))}
    if not found_transitions or not required_transitions.issubset(transitions):
        raise ValueError(f"Missing root transition probabilities in {path}")

    estimates = {
        **{
            f"birth_{state}": vectors["birth"][index]
            for index, state in enumerate(STATES)
        },
        **{
            f"death_{state}": vectors["death"][index]
            for index, state in enumerate(STATES)
        },
        **{
            f"sampling_{state}": vectors["sampling"][index]
            for index, state in enumerate(STATES)
        },
        **{
            f"p_A_to_{state}": transitions[(0, index)]
            for index, state in enumerate(STATES)
        },
        **transition_times,
    }
    return estimates


def _read_ground_truth(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        required_columns = {
            "setting",
            *(f"birth_{state}" for state in STATES),
            *(f"death_{state}" for state in STATES),
            "sampling_rate",
            *(f"p_A_to_{state}" for state in STATES),
            *(f"transition_time_{index}" for index in range(1, 6)),
        }
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"Missing ground-truth columns in {path}")

        ground_truth = {}
        for row in reader:
            setting = row["setting"]
            if setting in ground_truth:
                raise ValueError(f"Duplicate ground-truth row for {setting}")
            values = {
                key: float(row[key])
                for key in required_columns
                if key != "setting" and key != "sampling_rate"
            }
            values.update(
                {f"sampling_{state}": float(row["sampling_rate"]) for state in STATES}
            )
            values.update(
                {
                    f"transition_time_{index}": time
                    for index, time in enumerate(
                        sorted(
                            1.0 - float(row[f"transition_time_{index}"])
                            for index in range(1, 6)
                        ),
                        start=1,
                    )
                }
            )
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite ground-truth value for {setting}")
            ground_truth[setting] = values
    return ground_truth


def _add_identifiable_parameters(values: dict[str, float]) -> dict[str, float]:
    derived = dict(values)
    for state in STATES:
        derived[f"net_growth_{state}"] = (
            values[f"birth_{state}"] - values[f"death_{state}"]
        )
        derived[f"scaled_sampling_{state}"] = (
            values[f"birth_{state}"] * values[f"sampling_{state}"]
        )
    return derived


def load_completed_comparisons(
    data_dir: Path,
    estimate_transition_times: bool = False,
) -> dict[int, list[Comparison]]:
    ground_truth = _read_ground_truth(data_dir / "generated_settings.csv")
    ground_truth = {
        setting: _add_identifiable_parameters(values)
        for setting, values in ground_truth.items()
    }
    comparisons = {sample_size: [] for sample_size in SAMPLE_SIZES}

    for sample_size in SAMPLE_SIZES:
        inference_name = (
            f"inference_joint_times_first_{sample_size}_trees.out"
            if estimate_transition_times
            else f"inference_first_{sample_size}_trees.out"
        )
        pattern = f"setting_*/batch_*/{inference_name}"
        for path in sorted(data_dir.glob(pattern)):
            setting = path.parent.parent.name
            batch = path.parent.name
            if setting not in ground_truth:
                warnings.warn(
                    f"Skipping {path}: no ground-truth row for {setting}",
                    stacklevel=2,
                )
                continue
            try:
                estimates = _parse_output(path)
                if estimate_transition_times and any(
                    f"transition_time_{index}" not in estimates
                    for index in range(1, 6)
                ):
                    raise ValueError(f"Expected five estimated transition times in {path}")
            except (OSError, ValueError) as error:
                warnings.warn(
                    f"Skipping incomplete inference output {path}: {error}",
                    stacklevel=2,
                )
                continue
            comparisons[sample_size].append(
                Comparison(
                    sample_size=sample_size,
                    setting=setting,
                    batch=batch,
                    truth=ground_truth[setting],
                    inferred=_add_identifiable_parameters(estimates),
                    source=path,
                )
            )

    return comparisons


def _r_squared(truth: list[float], inferred: list[float]) -> float | None:
    if len(truth) != len(inferred) or not truth:
        raise ValueError("R² requires equally sized, non-empty value lists")
    mean = sum(truth) / len(truth)
    total_sum_squares = sum((value - mean) ** 2 for value in truth)
    if math.isclose(total_sum_squares, 0.0, abs_tol=1e-15):
        return None
    residual_sum_squares = sum(
        (actual - estimate) ** 2 for actual, estimate in zip(truth, inferred)
    )
    return 1.0 - residual_sum_squares / total_sum_squares


def _plot_parameters(
    comparisons: list[Comparison],
    sample_size: int,
    output_path: Path,
    parameters: list[tuple[str, str]],
    title: str,
) -> None:
    color = "#0072B2" if sample_size == 4 else "#E69F00"
    row_count = math.ceil(len(parameters) / 4)
    figure, axes = plt.subplots(row_count, 4, figsize=(15, row_count * 3.25))
    for index, axis in enumerate(axes.flat):
        if index >= len(parameters):
            axis.set_visible(False)
            continue
        label, parameter = parameters[index]
        truth = [point.truth[parameter] for point in comparisons]
        inferred = [point.inferred[parameter] for point in comparisons]
        score = _r_squared(truth, inferred)
        lower = min(*truth, *inferred)
        upper = max(*truth, *inferred)
        padding = max((upper - lower) * 0.05, 0.05)
        lower -= padding
        upper += padding

        axis.scatter(truth, inferred, s=22, alpha=0.75, color=color, edgecolors="none")
        axis.plot(
            [lower, upper],
            [lower, upper],
            color="#444444",
            linestyle="--",
            linewidth=1,
        )
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(label)
        score_text = "undefined (constant truth)" if score is None else f"{score:.3f}"
        axis.text(
            0.04,
            0.96,
            f"$R^2$ = {score_text}\nn = {len(truth)}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        axis.set_xlabel("Ground truth")
        axis.set_ylabel("Inferred")
        axis.grid(True, alpha=0.2)

    figure.suptitle(
        f"{title} — first {sample_size} trees "
        f"({len(comparisons)} completed runs)",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_completed_inferences(data_dir: Path, output_dir: Path) -> list[Path]:
    outputs = []
    for estimate_times in (False, True):
        comparisons = load_completed_comparisons(
            data_dir,
            estimate_transition_times=estimate_times,
        )
        prefix = "joint_times_" if estimate_times else ""
        parameters = PARAMETERS + TRANSITION_TIME_PARAMETERS if estimate_times else PARAMETERS
        inference_title = (
            "Star-model joint inference vs. ground truth"
            if estimate_times
            else "Star-model inference vs. ground truth"
        )
        for sample_size in SAMPLE_SIZES:
            runs = comparisons[sample_size]
            if not runs:
                continue
            parameter_plot = output_dir / (
                f"{prefix}inference_vs_ground_truth_first_{sample_size}_trees.png"
            )
            _plot_parameters(
                runs,
                sample_size,
                parameter_plot,
                parameters,
                inference_title,
            )
            outputs.append(parameter_plot)
            print(f"Plotted {len(runs)} completed runs to {parameter_plot}")

            identifiable_plot = output_dir / (
                f"{prefix}identifiable_parameters_first_{sample_size}_trees.png"
            )
            _plot_parameters(
                runs,
                sample_size,
                identifiable_plot,
                IDENTIFIABLE_PARAMETERS,
                "Identifiable parameters vs. ground truth",
            )
            outputs.append(identifiable_plot)
            print(f"Plotted {len(runs)} completed runs to {identifiable_plot}")
    if not outputs:
        raise FileNotFoundError(f"No complete star inference outputs found in {data_dir}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot completed star-model inferences against ground truth."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/simulation/coord_diff/star"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <data-dir>/plots.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.data_dir / "plots"
    plot_completed_inferences(args.data_dir, output_dir)


if __name__ == "__main__":
    main()
