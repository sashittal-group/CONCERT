from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, replace
import math
from pathlib import Path
import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analyses.common.cell_state_colors import CELL_STATE_COLORS, CELL_STATE_ORDER


RUN_NAME = re.compile(r"inference_transition_times_(\d+)")
ITERATION_LINE = re.compile(r"ITER\s+(\d+)\s+LL=(\S+)")
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
TRANSITION = re.compile(r"(\d+)->(\d+)\s+(" + NUMBER.pattern + r")")
ITERATION_NUMBER = re.compile(r"^ITER\s+(\d+)\b")
PARAMETERS = ("birth", "death", "sampling")
DATASET_DATA_DIRS = {
    "TLS": Path("data/troupe/TLS/fixed_sampling_0.1"),
    "TLSC": Path("data/troupe/TLSC/fixed_sampling_0.1"),
}


class NoCompleteSnapshotError(ValueError):
    pass


@dataclass(frozen=True)
class RunParameters:
    count: int
    birth_rates: tuple[float, ...]
    death_rates: tuple[float, ...]
    sampling_rates: tuple[float, ...]
    transition_probabilities: dict[tuple[int, int], float]
    source: Path
    iterations_completed: int = 0
    completed: bool = False


def _parse_vector(lines: list[str], name: str, path: Path) -> tuple[float, ...]:
    matches = [line for line in lines if line.startswith(f"{name}:")]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {name!r} vector in {path}; found {len(matches)}"
        )
    try:
        values = tuple(float(value) for value in matches[0].split(":", 1)[1].split())
    except ValueError as error:
        raise ValueError(f"Invalid {name!r} vector in {path}") from error
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Invalid {name!r} vector in {path}")
    return values


def _parse_transition_times(lines: list[str]) -> tuple[float, ...]:
    times: list[float] = []
    in_transition_times = False
    for line in lines:
        stripped = line.strip()
        if stripped == "transition_times:":
            in_transition_times = True
            continue
        if not in_transition_times or not stripped:
            continue
        fields = stripped.split()
        try:
            if len(fields) == 2 and fields[0].isdigit():
                times.append(float(fields[1]))
            else:
                times.extend(float(value) for value in fields)
        except ValueError:
            break
    return tuple(times)


def _parse_transition_probabilities(
    lines: list[str],
    path: Path,
) -> dict[tuple[int, int], float]:
    transitions: dict[tuple[int, int], float] = {}
    in_transitions = False
    found_section = False
    for line in lines:
        stripped = line.strip()
        if stripped == "transitions:":
            if found_section:
                raise ValueError(f"Multiple transition sections in {path}")
            found_section = True
            in_transitions = True
            continue
        if stripped == "transition_times:":
            in_transitions = False
            continue
        if not in_transitions or not stripped:
            continue
        match = TRANSITION.fullmatch(stripped)
        if match is None:
            continue
        source, target, probability = match.groups()
        value = float(probability)
        if not math.isfinite(value):
            raise ValueError(f"Invalid transition probability in {path}: {stripped}")
        transitions[(int(source), int(target))] = value
    if not found_section:
        raise ValueError(f"No transition probabilities found in {path}")
    return transitions


def _read_parameters(path: Path, count: int) -> RunParameters:
    lines = path.read_text().splitlines()
    vectors = {name: _parse_vector(lines, name, path) for name in PARAMETERS}
    if len({len(values) for values in vectors.values()}) != 1:
        raise ValueError(f"Inconsistent parameter vector lengths in {path}")
    transition_probabilities = _parse_transition_probabilities(lines, path)
    if any(
        source >= len(vectors["birth"]) or target >= len(vectors["birth"])
        for source, target in transition_probabilities
    ):
        raise ValueError(f"Transition state index outside parameter vectors in {path}")

    return RunParameters(
        count=count,
        birth_rates=vectors["birth"],
        death_rates=vectors["death"],
        sampling_rates=vectors["sampling"],
        transition_probabilities=transition_probabilities,
        source=path,
    )


def _last_complete_log_block(path: Path) -> list[str]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in path.read_text().splitlines():
        if ITERATION_LINE.match(line):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)

    for block in reversed(blocks):
        try:
            for name in PARAMETERS:
                _parse_vector(block, name, path)
            _parse_transition_probabilities(block, path)
        except ValueError:
            continue
        return block
    raise NoCompleteSnapshotError(f"No complete parameter snapshot found in {path}")


def _latest_iteration(log_path: Path, stdout_path: Path) -> int:
    if log_path.is_file():
        try:
            block = _last_complete_log_block(log_path)
        except NoCompleteSnapshotError:
            pass
        else:
            match = ITERATION_NUMBER.match(block[0])
            if match is not None:
                return int(match.group(1))

    iterations = []
    for path in (log_path, stdout_path):
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            match = ITERATION_NUMBER.match(line)
            if match is not None:
                iterations.append(int(match.group(1)))
    return max(iterations, default=0)


def _write_temporary_output(log_path: Path, output_path: Path) -> None:
    block = _last_complete_log_block(log_path)
    header = ITERATION_LINE.match(block[0])
    assert header is not None
    iteration, likelihood = header.groups()
    vectors = {name: _parse_vector(block, name, log_path) for name in PARAMETERS}
    if len({len(values) for values in vectors.values()}) != 1:
        raise ValueError(f"Inconsistent parameter vector lengths in {log_path}")

    lines = [f"ITER {iteration} LL={likelihood}"]
    lines.extend(
        f"{name}: {' '.join(format(value, '.17g') for value in vectors[name])}"
        for name in PARAMETERS
    )
    lines.append("transitions:")
    transition_pattern = re.compile(r"\d+->\d+\s+\S+")
    lines.extend(
        line.strip()
        for line in block
        if transition_pattern.fullmatch(line.strip())
    )
    transition_times = _parse_transition_times(block)
    lines.append("transition_times:")
    lines.extend(
        f"{index} {format(value, '.17g')}"
        for index, value in enumerate(transition_times)
    )
    output_path.write_text("\n".join(lines) + "\n")


def load_runs(
    data_dir: Path,
    *,
    max_count: int | None = None,
) -> list[RunParameters]:
    if max_count is not None and max_count < 1:
        raise ValueError("max_count must be positive when provided")
    counts: set[int] = set()
    for pattern in (
        "inference_transition_times_*.out",
        "inference_transition_times_*.tmp.out",
        "inference_transition_times_*.log",
    ):
        for path in data_dir.glob(pattern):
            match = RUN_NAME.fullmatch(path.name.removesuffix(".tmp.out").removesuffix(".out").removesuffix(".log"))
            if match:
                count = int(match.group(1))
                if max_count is None or count <= max_count:
                    counts.add(count)

    runs: list[RunParameters] = []
    for count in sorted(counts):
        stem = f"inference_transition_times_{count}"
        output_path = data_dir / f"{stem}.out"
        temporary_path = data_dir / f"{stem}.tmp.out"
        log_path = data_dir / f"{stem}.log"
        stdout_path = data_dir / f"{stem}.stdout.log"
        if output_path.is_file():
            source = output_path
        elif log_path.is_file():
            try:
                _write_temporary_output(log_path, temporary_path)
            except NoCompleteSnapshotError as error:
                if temporary_path.is_file():
                    warnings.warn(
                        f"Using existing {temporary_path}: {error}",
                        stacklevel=2,
                    )
                    source = temporary_path
                else:
                    warnings.warn(f"Skipping run {count}: {error}", stacklevel=2)
                    continue
            else:
                source = temporary_path
        elif temporary_path.is_file():
            source = temporary_path
        else:
            continue
        run = _read_parameters(source, count)
        runs.append(
            replace(
                run,
            iterations_completed=_latest_iteration(log_path, stdout_path),
            completed=output_path.is_file(),
            )
        )

    if not runs:
        raise FileNotFoundError(f"No usable inference outputs found in {data_dir}")
    state_counts = {len(run.birth_rates) for run in runs}
    if len(state_counts) != 1 or next(iter(state_counts)) != len(CELL_STATE_ORDER):
        raise ValueError("Inference outputs have inconsistent numbers of states")
    return runs


def _plot_parameter(
    runs: list[RunParameters],
    figure_path: Path,
    values_for_run: Callable[[RunParameters, int], float],
    ylabel: str,
) -> None:
    number_of_states = len(runs[0].birth_rates)
    ordered_runs = sorted(runs, key=lambda run: run.count)
    figure, axis = plt.subplots(figsize=(8, 5))
    for state in range(number_of_states):
        axis.plot(
            [run.count for run in ordered_runs],
            [values_for_run(run, state) for run in ordered_runs],
            marker="o",
            color=CELL_STATE_COLORS[CELL_STATE_ORDER[state]],
            label=CELL_STATE_ORDER[state],
        )

    axis.set_xticks(
        [run.count for run in ordered_runs],
    )
    axis.set_xlabel("Number of transitions")
    axis.set_ylabel(ylabel)
    axis.set_title(f"{ylabel} as the time step decreases")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def _scaled_birth_rate(run: RunParameters, state: int) -> float:
    return run.birth_rates[state] * run.sampling_rates[state]


def _transition_rate(run: RunParameters, transition: tuple[int, int]) -> float:
    return run.transition_probabilities.get(transition, 0.0) * run.count


def _plot_transition_rates(runs: list[RunParameters], figure_path: Path) -> None:
    transitions = sorted(
        {
            transition
            for run in runs
            for transition in run.transition_probabilities
            if transition[0] != transition[1]
        }
    )
    if not transitions:
        raise ValueError("No state-changing transition probabilities to plot")

    ordered_runs = sorted(runs, key=lambda run: run.count)
    figure, axis = plt.subplots(figsize=(8, 5))
    for transition in transitions:
        source_name = CELL_STATE_ORDER[transition[0]]
        target_name = CELL_STATE_ORDER[transition[1]]
        axis.plot(
            [run.count for run in ordered_runs],
            [_transition_rate(run, transition) for run in ordered_runs],
            marker="o",
            color=CELL_STATE_COLORS[target_name],
            label=f"{source_name}->{target_name}",
        )

    axis.set_xticks([run.count for run in ordered_runs])
    axis.set_xlabel("Number of transitions")
    axis.set_ylabel("Transition probability / Δt")
    axis.set_title("State-changing transition probabilities scaled by 1/Δt")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def _plot_iteration_progress(runs: list[RunParameters], figure_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for completed, marker, label in (
        (False, ".", "Incomplete"),
        (True, "*", "Completed"),
    ):
        selected = [run for run in runs if run.completed is completed]
        if not selected:
            continue
        axis.scatter(
            [run.count for run in selected],
            [run.iterations_completed for run in selected],
            marker=marker,
            s=160 if completed else 100,
            label=label,
        )
        for run in selected:
            axis.annotate(
                f"{run.iterations_completed:,}",
                (run.count, run.iterations_completed),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

    axis.set_xticks(sorted(run.count for run in runs))
    axis.set_xlabel("Number of transitions")
    axis.set_ylabel("Iterations completed (last logged checkpoint)")
    axis.set_title("Optimization progress by transition count")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def create_summary_figures(
    data_dir: Path,
    figure_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    runs = load_runs(data_dir)
    net_growth_path = figure_dir / "summary_net_growth_rate.png"
    scaled_birth_path = figure_dir / "summary_scaled_birth_rate.png"
    transition_rate_path = figure_dir / "summary_transition_probabilities_over_dt.png"
    iteration_path = figure_dir / "summary_iterations_by_transition_count.png"
    _plot_parameter(
        runs,
        net_growth_path,
        lambda run, state: run.birth_rates[state] - run.death_rates[state],
        "Net growth rate (birth - death)",
    )
    _plot_parameter(
        runs,
        scaled_birth_path,
        _scaled_birth_rate,
        "Scaled birth rate (birth rate × sampling rate)",
    )
    _plot_transition_rates(runs, transition_rate_path)
    _plot_iteration_progress(runs, iteration_path)
    return net_growth_path, scaled_birth_path, transition_rate_path, iteration_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize fixed-sampling TLS and TLSC transition-time inference runs. "
            "Missing .out files are reconstructed from the latest complete "
            "iteration in the corresponding .log file."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("TLS", "TLSC", "both"),
        default="both",
        help="Dataset to plot; defaults to both TLS and TLSC.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Override the data directory and process only that directory.",
    )
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()
    if args.data_dir is not None:
        if args.dataset != "both":
            parser.error("--data-dir cannot be combined with --dataset")
        targets = [
            (args.data_dir.name, args.data_dir, args.figure_dir or args.data_dir)
        ]
    else:
        datasets = (
            tuple(DATASET_DATA_DIRS)
            if args.dataset == "both"
            else (args.dataset,)
        )
        targets = []
        for dataset in datasets:
            data_dir = DATASET_DATA_DIRS[dataset]
            if args.figure_dir is None:
                figure_dir = data_dir
            elif len(datasets) == 1:
                figure_dir = args.figure_dir
            else:
                figure_dir = args.figure_dir / dataset
            targets.append((dataset, data_dir, figure_dir))

    generated_figures = 0
    for dataset, data_dir, figure_dir in targets:
        try:
            figures = create_summary_figures(data_dir, figure_dir)
        except FileNotFoundError as error:
            warnings.warn(f"Skipping {dataset}: {error}", stacklevel=2)
            continue
        for figure in figures:
            print(f"Saved {figure}")
            generated_figures += 1
    if not generated_figures:
        raise FileNotFoundError("No summary figures could be generated")


if __name__ == "__main__":
    main()
