#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.results.kptracer.grouped_transition_schedules import (
    GroupedTransitionSchedule,
    load_grouped_transition_schedules,
    read_fit_log_likelihood,
)

SHARPTNI_RESULTS = (
    REPOSITORY_ROOT
    / "sharp-tni/results/kptracer_sharptni_20261101_1118"
)
SCHEDULE_TABLES = [
    SHARPTNI_RESULTS
    / "enumerated_solutions/gamma21_mixed_direction_transition_times.tsv",
    SHARPTNI_RESULTS
    / "enumerated_solutions/gamma22_mixed_direction_transition_times.tsv",
]
GROUPED_FIT_DIRECTORY = (
    SHARPTNI_RESULTS / "concert_initialized/grouped_schedule_fits"
)
GRID_FIT_DIRECTORY = (
    REPOSITORY_ROOT
    / "data/KPTracer-Data/processed_data/"
    "inference_progressive_fixed_sampling_0.05_5_20_regenerated_grid_100"
)
PLOT_PATH = (
    REPOSITORY_ROOT
    / "data/KPTracer-Data/processed_data/"
    "likelihood_vs_transition_count_grid_100_regenerated_precision_repaired.png"
)


def load_grid_likelihoods() -> dict[int, float]:
    likelihoods: dict[int, float] = {}
    for path in GRID_FIT_DIRECTORY.glob("transition_count_*.out"):
        match = re.fullmatch(r"transition_count_(\d+)", path.stem)
        if match is None:
            continue
        count = int(match.group(1))
        likelihoods[count] = read_fit_log_likelihood(path)

    expected_counts = set(range(5, 21))
    if set(likelihoods) != expected_counts:
        missing = sorted(expected_counts - set(likelihoods))
        unexpected = sorted(set(likelihoods) - expected_counts)
        raise ValueError(
            "Grid likelihood files must cover transition counts 5-20; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return dict(sorted(likelihoods.items()))


def load_completed_eleven_time_fits(
    schedules: dict[str, GroupedTransitionSchedule],
) -> list[tuple[str, float, GroupedTransitionSchedule]]:
    fits_directory = GROUPED_FIT_DIRECTORY / "fits"
    completed: list[tuple[str, float, GroupedTransitionSchedule]] = []
    for path in sorted(fits_directory.glob("n11-*.out")):
        schedule = schedules.get(path.stem)
        if schedule is None:
            raise ValueError(
                f"Completed fit has no matching SharpTNI schedule: {path}"
            )
        if len(schedule.transition_times) != 11:
            raise ValueError(
                f"Fit filename indicates 11 times but schedule has "
                f"{len(schedule.transition_times)}: {path}"
            )
        completed.append(
            (path.stem, read_fit_log_likelihood(path), schedule)
        )

    if not completed:
        raise FileNotFoundError(
            f"No completed 11-time fits found in {fits_directory}"
        )
    return sorted(completed, key=lambda fit: fit[1])


def write_completed_fits_tsv(
    fits: list[tuple[str, float, GroupedTransitionSchedule]],
) -> Path:
    output_path = GROUPED_FIT_DIRECTORY / "completed_11_time_schedule_likelihoods.tsv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "schedule_id",
                "transition_count",
                "log_likelihood",
                "gamma21_solution_indices",
                "gamma22_solution_indices",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for schedule_id, likelihood, schedule in fits:
            writer.writerow(
                {
                    "schedule_id": schedule_id,
                    "transition_count": len(schedule.transition_times),
                    "log_likelihood": f"{likelihood:.15g}",
                    "gamma21_solution_indices": ",".join(
                        str(solution_index)
                        for gamma, solution_index in sorted(
                            schedule.source_solutions
                        )
                        if gamma == 21
                    ),
                    "gamma22_solution_indices": ",".join(
                        str(solution_index)
                        for gamma, solution_index in sorted(
                            schedule.source_solutions
                        )
                        if gamma == 22
                    ),
                }
            )
    return output_path


def plot_likelihoods(
    grid_likelihoods: dict[int, float],
    grouped_fits: list[tuple[str, float, GroupedTransitionSchedule]],
) -> None:
    fig, axis = plt.subplots(figsize=(13.5, 7.8))
    counts = list(grid_likelihoods)
    axis.plot(
        counts,
        [grid_likelihoods[count] for count in counts],
        color="#2864ff",
        marker="o",
        linewidth=2.8,
        markersize=7,
    )

    axis.scatter(
        [11],
        [136.77],
        marker="*",
        s=330,
        color="#00a878",
        edgecolors="black",
        linewidths=1.2,
        label="MACH2 global min, CONCERT init (11; LL=136.77)",
    )
    axis.scatter(
        [21],
        [416.58],
        marker="*",
        s=330,
        color="#00a878",
        edgecolors="black",
        linewidths=1.2,
        label="MACH2 same-direction, CONCERT init (21; LL=416.58)",
    )
    axis.scatter(
        [23],
        [335.64],
        marker="*",
        s=330,
        color="#d980ad",
        edgecolors="black",
        linewidths=1.2,
        label="SharpTNI, CONCERT init (23; LL=335.64)",
    )

    axis.scatter(
        [11] * len(grouped_fits),
        [likelihood for _, likelihood, _ in grouped_fits],
        marker="o",
        s=75,
        color="#f28e2b",
        alpha=0.5,
        edgecolors="none",
        label=(
            "SharpTNI grouped schedules, 11 times "
            f"({len(grouped_fits)} completed)"
        ),
        zorder=4,
    )

    axis.set_title("KPTracer log likelihood vs. transition count", fontsize=22)
    axis.set_xlabel("Number of transition times", fontsize=18)
    axis.set_ylabel("Log likelihood", fontsize=18)
    axis.set_xlim(4.1, 23.9)
    axis.set_ylim(-300, 730)
    axis.set_xticks(range(5, 24))
    axis.set_yticks(range(-300, 701, 100))
    axis.tick_params(labelsize=14)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper left", fontsize=12)
    axis.text(
        0.99,
        0.07,
        "Sampling probability = 0.05; grid step = 0.01",
        transform=axis.transAxes,
        ha="right",
        color="dimgray",
        fontsize=14,
    )
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=120)
    plt.close(fig)


def main() -> None:
    schedules = load_grouped_transition_schedules(SCHEDULE_TABLES)
    grouped_fits = load_completed_eleven_time_fits(schedules)
    grid_likelihoods = load_grid_likelihoods()
    table_path = write_completed_fits_tsv(grouped_fits)
    plot_likelihoods(grid_likelihoods, grouped_fits)
    print(f"Plotted {len(grouped_fits)} completed 11-time fits: {PLOT_PATH}")
    print(f"Likelihood and solution mapping: {table_path}")


if __name__ == "__main__":
    main()
