from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.analyses.common.profile_config import load_config
from src.analyses.common.profile_data import load_inference_parameters
from src.analyses.common.state_coloring import (
    OUTPUT_FILENAME as SEGMENT_STATES_FILENAME,
    process_dataset,
)
from src.analyses.kptracer.grouped_transition_schedules import (
    load_grouped_transition_schedules,
    read_fit_log_likelihood,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DATA_DIRECTORY = (
    REPOSITORY_ROOT / "data/KPTracer-Data/processed_data"
)
PROGRESSIVE_FIT_DIRECTORY = (
    PROCESSED_DATA_DIRECTORY
    / "inference_progressive_fixed_sampling_0.05_5_20_regenerated_grid_100"
)
MACH2_RUN_DIRECTORY = (
    REPOSITORY_ROOT / "mach2/results/kptracer_mach2_20261101_1403"
)
SHARPTNI_RUN_DIRECTORY = (
    REPOSITORY_ROOT / "sharp-tni/results/kptracer_sharptni_20261101_1118"
)
SHARPTNI_SCHEDULE_TABLES = [
    SHARPTNI_RUN_DIRECTORY
    / "enumerated_solutions/gamma21_mixed_direction_transition_times.tsv",
    SHARPTNI_RUN_DIRECTORY
    / "enumerated_solutions/gamma22_mixed_direction_transition_times.tsv",
]
STATE_NAMES = ("T1", "L1", "L2", "L3", "S1")
SEGMENT_COLUMNS = (
    "segment_id",
    "tree",
    "subtree",
    "original_edge",
    "segment_start_time",
    "segment_end_time",
    "most_likely_state",
)
TIME_TOLERANCE = 1e-8
BAR_WIDTH = 0.004
MIN_BAR_SEPARATION = 0.006
OUTPUT_DIRECTORY = PROCESSED_DATA_DIRECTORY / "transition_type_counts"
PLOT_PATH = OUTPUT_DIRECTORY / "kptracer_transition_type_counts_by_time.png"
COUNTS_PATH = OUTPUT_DIRECTORY / "kptracer_transition_type_counts_by_time.csv"


@dataclass(frozen=True)
class FitSpec:
    label: str
    inference_path: Path
    coloring_directory: Path


def _best_completed_sharptni_11_fit() -> Path:
    fits_directory = (
        SHARPTNI_RUN_DIRECTORY
        / "concert_initialized/grouped_schedule_fits/fits"
    )
    completed_fits = [
        path
        for path in fits_directory.glob("n11-*.out")
        if path.is_file()
    ]
    if not completed_fits:
        raise FileNotFoundError(
            f"No completed 11-time SharpTNI grouped fits in {fits_directory}"
        )
    return max(completed_fits, key=read_fit_log_likelihood)


def _fit_specs() -> list[FitSpec]:
    grouped_fit = _best_completed_sharptni_11_fit()
    sharp_schedule_id = grouped_fit.stem
    mach2_concert_directory = MACH2_RUN_DIRECTORY / "concert_initialized"
    mach2_coloring_directory = mach2_concert_directory / "state_coloring"
    sharp_coloring_directory = (
        SHARPTNI_RUN_DIRECTORY
        / "concert_initialized/state_coloring"
        / f"grouped_best_11_{sharp_schedule_id}"
    )
    return [
        FitSpec(
            "CONCERT N=7",
            PROGRESSIVE_FIT_DIRECTORY / "transition_count_7.out",
            PROGRESSIVE_FIT_DIRECTORY / "state_coloring_n7",
        ),
        FitSpec(
            "CONCERT N=11",
            PROGRESSIVE_FIT_DIRECTORY / "transition_count_11.out",
            PROGRESSIVE_FIT_DIRECTORY / "state_coloring_n11",
        ),
        FitSpec(
            "MACH2-derived N=11, CONCERT initialized",
            mach2_concert_directory / "carta_concert_init_global_min_11.out",
            mach2_coloring_directory / "global_min_11",
        ),
        FitSpec(
            "MACH2-derived N=21, CONCERT initialized",
            mach2_concert_directory / "carta_concert_init_pairwise_21.out",
            mach2_coloring_directory / "same_direction_21",
        ),
        FitSpec(
            "SharpTNI grouped N=11, best completed LL",
            grouped_fit,
            sharp_coloring_directory,
        ),
    ]


def _transition_types() -> tuple[tuple[str, str], ...]:
    states = STATE_NAMES
    edge_path = REPOSITORY_ROOT / "src/analyses/kptracer/transition_edges.txt"
    transitions = []
    for line in edge_path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 2:
            raise ValueError(f"Invalid transition edge in {edge_path}: {line!r}")
        source_index, target_index = map(int, fields)
        try:
            transitions.append((states[source_index], states[target_index]))
        except IndexError as error:
            raise ValueError(
                f"Transition edge has an invalid state index: {line!r}"
            ) from error
    if not transitions or len(set(transitions)) != len(transitions):
        raise ValueError(f"Transition edges in {edge_path} must be unique.")
    return tuple(transitions)


def _analysis_config():
    config = load_config(REPOSITORY_ROOT / "src/analyses/tls/config.yaml")
    return replace(config, state_names=STATE_NAMES)


def _ensure_segment_states(
    fit: FitSpec,
    config,
) -> Path:
    if not fit.inference_path.is_file():
        raise FileNotFoundError(
            f"Missing inference output for {fit.label}: {fit.inference_path}"
        )
    state_data_path = fit.coloring_directory / SEGMENT_STATES_FILENAME
    metadata_path = fit.coloring_directory / "metadata.json"
    if state_data_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        stored_inference = Path(metadata.get("inference_file", ""))
        if not stored_inference.is_absolute():
            stored_inference = REPOSITORY_ROOT / stored_inference
        if stored_inference.resolve() == fit.inference_path.resolve():
            return state_data_path

    return process_dataset(
        "KPTracer",
        fit.inference_path,
        PROCESSED_DATA_DIRECTORY,
        fit.coloring_directory,
        config,
    )


def _unique_segments(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=list(SEGMENT_COLUMNS))
    consistency_columns = [
        column for column in SEGMENT_COLUMNS if column != "segment_id"
    ]
    grouped = frame.groupby("segment_id", sort=False, dropna=False)
    for column in consistency_columns:
        if grouped[column].nunique(dropna=False).gt(1).any():
            raise ValueError(
                f"Inconsistent {column} values for edge segments in {path}"
            )
    segments = frame.drop_duplicates("segment_id").copy()
    for column in ("segment_start_time", "segment_end_time"):
        segments[column] = pd.to_numeric(segments[column], errors="raise")
    times = segments[
        ["segment_start_time", "segment_end_time"]
    ].to_numpy(dtype=float)
    if not np.isfinite(times).all():
        raise ValueError(f"Non-finite segment times in {path}")
    times[np.abs(times) <= 1e-12] = 0.0
    segments.loc[
        :, ["segment_start_time", "segment_end_time"]
    ] = times
    start_times = segments["segment_start_time"].to_numpy(dtype=float)
    end_times = segments["segment_end_time"].to_numpy(dtype=float)
    if (start_times < end_times - TIME_TOLERANCE).any():
        raise ValueError(
            f"Expected tree edges to run from older to younger times in {path}"
        )
    segments = segments.loc[
        start_times - end_times > TIME_TOLERANCE
    ].copy()
    if segments.empty:
        raise ValueError(f"No positive-duration edge segments found in {path}")
    return segments


def _count_segment_state_changes(
    segments: pd.DataFrame,
    transition_times: tuple[float, ...],
    allowed_transitions: tuple[tuple[str, str], ...],
) -> dict[tuple[int, str, str], int]:
    counts: dict[tuple[int, str, str], int] = {}
    group_columns = ["tree", "subtree", "original_edge"]
    for _, edge_segments in segments.groupby(group_columns, sort=False):
        ordered = edge_segments.sort_values(
            "segment_start_time",
            ascending=False,
            kind="stable",
        ).to_dict("records")
        for older_segment, younger_segment in zip(ordered, ordered[1:]):
            older_end = float(older_segment["segment_end_time"])
            younger_start = float(younger_segment["segment_start_time"])
            if not np.isclose(
                older_end,
                younger_start,
                rtol=0.0,
                atol=TIME_TOLERANCE,
            ):
                raise ValueError(
                    "Adjacent colored segments do not meet on the same "
                    f"original edge: {older_segment['original_edge']}"
                )
            source = str(older_segment["most_likely_state"])
            target = str(younger_segment["most_likely_state"])
            if source == target:
                continue
            event_time = (older_end + younger_start) / 2.0
            time_index = min(
                range(len(transition_times)),
                key=lambda index: abs(transition_times[index] - event_time),
            )
            if (
                abs(transition_times[time_index] - event_time)
                > TIME_TOLERANCE
            ):
                raise ValueError(
                    f"Colored state change at {event_time:.15g} does not match "
                    "a fitted transition time."
                )
            transition = (source, target)
            if transition not in allowed_transitions:
                raise ValueError(
                    f"Colored state change {source}->{target} is not in the "
                    "configured transition graph."
                )
            key = (time_index, source, target)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _display_elapsed_times(
    transition_times: tuple[float, ...],
) -> tuple[float, ...]:
    exact_elapsed_times = np.asarray(
        [1.0 - value for value in transition_times],
        dtype=float,
    )
    order = np.argsort(exact_elapsed_times)
    positions = np.clip(
        exact_elapsed_times[order],
        BAR_WIDTH / 2.0,
        1.0 - BAR_WIDTH / 2.0,
    )
    for index in range(1, len(positions)):
        positions[index] = max(
            positions[index],
            positions[index - 1] + MIN_BAR_SEPARATION,
        )
    if positions[-1] > 1.0 - BAR_WIDTH / 2.0:
        positions[-1] = 1.0 - BAR_WIDTH / 2.0
        for index in range(len(positions) - 2, -1, -1):
            positions[index] = min(
                positions[index],
                positions[index + 1] - MIN_BAR_SEPARATION,
            )
    if positions[0] < BAR_WIDTH / 2.0:
        shift = BAR_WIDTH / 2.0 - positions[0]
        positions += shift
    if (
        positions[0] < BAR_WIDTH / 2.0 - TIME_TOLERANCE
        or positions[-1] > 1.0 - BAR_WIDTH / 2.0 + TIME_TOLERANCE
        or (
            len(positions) > 1
            and np.any(np.diff(positions) < MIN_BAR_SEPARATION - TIME_TOLERANCE)
        )
    ):
        raise ValueError("Could not separate closely spaced transition bars.")
    display_times = np.empty_like(positions)
    display_times[order] = positions
    return tuple(float(value) for value in display_times)


def collect_transition_counts(
    fits: list[FitSpec],
    allowed_transitions: tuple[tuple[str, str], ...],
    config,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    records: list[dict[str, object]] = []
    plot_data: dict[str, dict[str, object]] = {}
    for fit in fits:
        parameters = load_inference_parameters(fit.inference_path, STATE_NAMES)
        transition_times = tuple(
            float(value) for value in parameters.transition_times
        )
        if not transition_times or not np.all(np.diff(transition_times) > 0):
            raise ValueError(
                f"Transition times for {fit.label} must be strictly increasing."
            )
        display_times = _display_elapsed_times(transition_times)
        segment_path = _ensure_segment_states(fit, config)
        segments = _unique_segments(segment_path)
        counts = _count_segment_state_changes(
            segments,
            transition_times,
            allowed_transitions,
        )
        likelihood = read_fit_log_likelihood(fit.inference_path)
        plot_data[fit.label] = {
            "transition_times": transition_times,
            "display_elapsed_times": display_times,
            "counts": counts,
            "likelihood": likelihood,
        }
        for time_index, carta_time in enumerate(transition_times):
            for source, target in allowed_transitions:
                records.append(
                    {
                        "fit": fit.label,
                        "log_likelihood": likelihood,
                        "transition_count": len(transition_times),
                        "carta_time": carta_time,
                        "elapsed_time_root_to_tip": 1.0 - carta_time,
                        "display_elapsed_time_root_to_tip": display_times[
                            time_index
                        ],
                        "plot_time_shift": (
                            display_times[time_index] - (1.0 - carta_time)
                        ),
                        "transition_type": f"{source}->{target}",
                        "source_state": source,
                        "target_state": target,
                        "count": counts.get(
                            (time_index, source, target),
                            0,
                        ),
                    }
                )
    return records, plot_data


def _plot_counts(
    plot_data: dict[str, dict[str, object]],
    transition_types: tuple[tuple[str, str], ...],
) -> None:
    figure, axes = plt.subplots(
        len(plot_data),
        1,
        figsize=(14, 2.7 * len(plot_data)),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    colors = plt.get_cmap("tab10").colors
    transition_colors = {
        transition: colors[index]
        for index, transition in enumerate(transition_types)
    }

    for axis, (fit_label, data) in zip(axes, plot_data.items()):
        transition_times = data["transition_times"]
        display_elapsed_times = data["display_elapsed_times"]
        counts = data["counts"]
        time_order = sorted(
            range(len(transition_times)),
            key=lambda index: 1.0 - transition_times[index],
        )
        bar_positions = np.asarray(
            [display_elapsed_times[index] for index in time_order],
            dtype=float,
        )
        bottom = np.zeros(len(transition_times), dtype=float)
        for transition in transition_types:
            heights = np.asarray(
                [
                    counts.get((time_index, *transition), 0)
                    for time_index in time_order
                ],
                dtype=float,
            )
            axis.bar(
                bar_positions,
                heights,
                width=BAR_WIDTH,
                bottom=bottom,
                color=transition_colors[transition],
                linewidth=0,
                align="center",
            )
            bottom += heights

        zero_count_positions = bar_positions[bottom == 0]
        if len(zero_count_positions):
            axis.scatter(
                zero_count_positions,
                np.zeros(len(zero_count_positions)),
                marker="o",
                s=30,
                facecolors="white",
                edgecolors="black",
                linewidths=0.9,
                zorder=5,
                clip_on=False,
            )
            for position in zero_count_positions:
                axis.annotate(
                    "0",
                    (position, 0),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        axis.set_title(
            f"{fit_label} | N={len(transition_times)} | "
            f"LL={data['likelihood']:.3f}",
            loc="left",
            fontsize=11,
        )
        axis.set_ylabel("MAP changes")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[-1].set_xlim(0.0, 1.0)
    axes[-1].set_xticks(np.linspace(0.0, 1.0, 6))

    axes[-1].set_xlabel(
        "Evolutionary time from root to tips (1 - CARTA time)",
        labelpad=8,
    )
    figure.suptitle(
        "KPTracer transition types across time\n"
        "Posterior-MAP edge-segment changes; crowded bars slightly shifted for visibility",
        fontsize=15,
        y=0.995,
    )
    legend_handles = [
        Patch(
            facecolor=transition_colors[transition],
            label=f"{transition[0]}->{transition[1]}",
        )
        for transition in transition_types
    ]
    figure.legend(
        handles=legend_handles,
        title="Transition type",
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
    )
    figure.subplots_adjust(
        left=0.1,
        right=0.98,
        top=0.84,
        bottom=0.08,
        hspace=0.4,
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    fits = _fit_specs()
    transition_types = _transition_types()
    config = _analysis_config()
    records, plot_data = collect_transition_counts(
        fits,
        transition_types,
        config,
    )
    _plot_counts(plot_data, transition_types)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with COUNTS_PATH.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(records[0]),
        )
        writer.writeheader()
        writer.writerows(records)
    sharp_fit = fits[-1]
    print(
        f"Wrote {PLOT_PATH} using {len(fits)} fits; "
        f"best completed SharpTNI grouped fit: {sharp_fit.inference_path.name} "
        f"(LL={read_fit_log_likelihood(sharp_fit.inference_path):.6f})."
    )
    print(f"Wrote exact-time transition counts to {COUNTS_PATH}")


if __name__ == "__main__":
    main()
