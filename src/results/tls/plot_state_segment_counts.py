from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.results.cell_state_colors import CELL_STATE_COLORS, CELL_STATE_ORDER
from src.results.tls.data import load_inference_parameters
from src.results.tls.process_state_coloring import OUTPUT_FILENAME


TRANSITION_TIME_COUNT = 100
TIME_INTERVAL_COUNT = TRANSITION_TIME_COUNT + 1
PLOT_FILENAME = "state_segment_counts_by_time_100.png"
COUNTS_FILENAME = "state_segment_counts_by_time_100.csv"
PROPORTIONS_PLOT_FILENAME = "state_segment_proportions_by_time_100.png"
PROPORTIONS_FILENAME = "state_segment_proportions_by_time_100.csv"
SEGMENT_COLUMNS = (
    "segment_id",
    "segment_start_time",
    "segment_end_time",
    "most_likely_state",
)


def _unique_segments(segments: pd.DataFrame) -> pd.DataFrame:
    if segments.empty:
        raise ValueError("No edge-segment state data to count.")
    missing_columns = set(SEGMENT_COLUMNS) - set(segments.columns)
    if missing_columns:
        raise ValueError(f"Missing segment columns: {sorted(missing_columns)}")

    grouped = segments.groupby("segment_id", sort=False, dropna=False)
    for column in SEGMENT_COLUMNS[1:]:
        if grouped[column].nunique(dropna=False).gt(1).any():
            raise ValueError(
                f"Inconsistent {column} values for an edge segment."
            )

    unique = segments.drop_duplicates("segment_id").loc[
        :, list(SEGMENT_COLUMNS)
    ].copy()
    unique["segment_start_time"] = pd.to_numeric(
        unique["segment_start_time"], errors="raise"
    )
    unique["segment_end_time"] = pd.to_numeric(
        unique["segment_end_time"], errors="raise"
    )
    for column in ("segment_start_time", "segment_end_time"):
        times = unique[column].to_numpy(dtype=float)
        times[np.abs(times) <= 1e-12] = 0.0
        unique[column] = times
    times = unique[["segment_start_time", "segment_end_time"]].to_numpy()
    if not np.isfinite(times).all():
        raise ValueError("Segment times must be finite.")
    unique["time_lower"] = np.minimum(
        unique["segment_start_time"], unique["segment_end_time"]
    )
    unique["time_upper"] = np.maximum(
        unique["segment_start_time"], unique["segment_end_time"]
    )
    if (unique["time_lower"] == unique["time_upper"]).any():
        raise ValueError("Edge segments must have positive duration.")
    return unique


def count_segments_by_time(
    segments: pd.DataFrame,
    bin_edges: np.ndarray,
    state_names: tuple[str, ...] = CELL_STATE_ORDER,
) -> pd.DataFrame:
    unique = _unique_segments(segments)
    edges = np.asarray(bin_edges, dtype=float)
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.isfinite(edges).all()
        or not np.all(np.diff(edges) > 0)
    ):
        raise ValueError("Time-bin edges must be a finite, increasing sequence.")

    state_index = pd.Index(state_names).get_indexer(unique["most_likely_state"])
    if (state_index < 0).any():
        unknown_states = sorted(
            set(unique.loc[state_index < 0, "most_likely_state"].astype(str))
        )
        raise ValueError(f"Unknown most-likely states: {unknown_states}")

    lower = unique["time_lower"].to_numpy(dtype=float)
    upper = unique["time_upper"].to_numpy(dtype=float)
    counts = np.zeros((len(state_names), len(edges) - 1), dtype=np.int64)
    overlap_tolerance = 1e-12 * max(1.0, abs(edges[0]), abs(edges[-1]))
    for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        overlaps = (lower < right - overlap_tolerance) & (
            upper > left + overlap_tolerance
        )
        counts[:, bin_index] = np.bincount(
            state_index[overlaps],
            minlength=len(state_names),
        )

    result = pd.DataFrame(
        {
            "bin_index": np.arange(len(edges) - 1),
            "time_start": edges[:-1],
            "time_end": edges[1:],
        }
    )
    for state_index, state in enumerate(state_names):
        result[state] = counts[state_index]
    result["total_segments"] = counts.sum(axis=0)
    return result


def normalize_state_counts(
    counts: pd.DataFrame,
    state_names: tuple[str, ...] = CELL_STATE_ORDER,
) -> pd.DataFrame:
    required_columns = {
        *state_names,
        "bin_index",
        "time_start",
        "time_end",
        "total_segments",
    }
    missing_columns = required_columns - set(counts.columns)
    if missing_columns:
        raise ValueError(f"Missing count columns: {sorted(missing_columns)}")

    totals = pd.to_numeric(counts["total_segments"], errors="raise").to_numpy(
        dtype=float
    )
    state_counts = counts.loc[:, list(state_names)].apply(
        pd.to_numeric,
        errors="raise",
    )
    if (
        not np.isfinite(totals).all()
        or (totals <= 0).any()
        or (state_counts.to_numpy(dtype=float) < 0).any()
        or not np.allclose(
            state_counts.sum(axis=1).to_numpy(dtype=float),
            totals,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError("Each time interval must have valid, positive state counts.")

    proportions = counts.loc[:, ["bin_index", "time_start", "time_end"]].copy()
    for state in state_names:
        proportions[state] = state_counts[state].to_numpy(dtype=float) / totals
    proportions["total_segments"] = totals.astype(np.int64)
    if not np.allclose(
        proportions.loc[:, list(state_names)].sum(axis=1).to_numpy(),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Normalized state proportions do not sum to one.")
    return proportions


def make_transition_bin_edges(
    transition_times: list[float],
    time_min: float,
    time_max: float,
) -> np.ndarray:
    for value_name, value in (("time_min", time_min), ("time_max", time_max)):
        nearest_integer = round(value)
        if np.isclose(value, nearest_integer, rtol=0.0, atol=1e-12):
            if value_name == "time_min":
                time_min = float(nearest_integer)
            else:
                time_max = float(nearest_integer)
    transition_array = np.asarray(transition_times, dtype=float)
    if (
        transition_array.ndim != 1
        or len(transition_array) != TRANSITION_TIME_COUNT
        or not np.isfinite(transition_array).all()
        or not np.all(np.diff(transition_array) > 0)
        or transition_array[0] <= time_min
        or transition_array[-1] >= time_max
    ):
        raise ValueError(
            f"Expected {TRANSITION_TIME_COUNT} ordered transition times strictly "
            "inside the time range."
        )
    edges = np.concatenate(([time_min], transition_array, [time_max]))
    if len(edges) != TIME_INTERVAL_COUNT + 1:
        raise ValueError("Transition times did not define the expected intervals.")
    return edges


def _load_dataset_segments(
    dataset: str,
    data_root: Path,
    output_root: Path,
    state_names: tuple[str, ...],
) -> tuple[pd.DataFrame, list[float]]:
    coloring_dir = output_root / dataset
    metadata_path = coloring_dir / "metadata.json"
    state_data_path = coloring_dir / OUTPUT_FILENAME
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("dataset") != dataset:
        raise ValueError(f"Dataset mismatch in {metadata_path}")
    if metadata.get("transition_time_count") != TRANSITION_TIME_COUNT:
        raise ValueError(
            f"{metadata_path} must describe {TRANSITION_TIME_COUNT} transition times."
        )
    if tuple(metadata.get("state_names", ())) != state_names:
        raise ValueError(f"State names in {metadata_path} do not match {state_names}.")
    inference_path = Path(metadata["inference_file"])
    if not inference_path.is_file():
        raise FileNotFoundError(f"Missing fitted inference output: {inference_path}")
    transition_times = load_inference_parameters(
        inference_path,
        state_names,
    ).transition_times
    if len(transition_times) != TRANSITION_TIME_COUNT:
        raise ValueError(
            f"Expected {TRANSITION_TIME_COUNT} fitted transition times for {dataset}; "
            f"found {len(transition_times)}."
        )

    segments = pd.read_csv(state_data_path, usecols=list(SEGMENT_COLUMNS))
    unique = _unique_segments(segments)
    expected_count = int(metadata["edge_segment_count"])
    if len(unique) != expected_count:
        raise ValueError(
            f"Expected {expected_count} unique segments for {dataset}; "
            f"found {len(unique)}."
        )
    return unique, transition_times


def _plot_stacked_values(
    dataset: str,
    values: pd.DataFrame,
    state_names: tuple[str, ...],
    output_path: Path,
    dpi: int,
    ylabel: str,
    title_measure: str,
    y_limits: tuple[float, float] | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 6))
    bin_widths = values["time_end"].to_numpy() - values["time_start"].to_numpy()
    bin_centers = (
        values["time_start"].to_numpy() + values["time_end"].to_numpy()
    ) / 2
    bottom = np.zeros(len(values), dtype=float)
    for state in state_names:
        heights = values[state].to_numpy(dtype=float)
        axis.bar(
            bin_centers,
            heights,
            width=bin_widths * 0.94,
            bottom=bottom,
            color=CELL_STATE_COLORS[state],
            label=state,
            linewidth=0,
        )
        bottom += heights

    time_min = float(values["time_start"].iloc[0])
    time_max = float(values["time_end"].iloc[-1])
    axis.set_xlim(time_min, time_max)
    axis.set_xticks(np.linspace(time_min, time_max, 11))
    axis.set_xlabel("Time")
    axis.set_ylabel(ylabel)
    axis.set_title(
        f"{dataset}: {title_measure} across all subtrees "
        f"({TIME_INTERVAL_COUNT} transition-time intervals)"
    )
    if y_limits is not None:
        axis.set_ylim(*y_limits)
        axis.set_yticks(np.linspace(y_limits[0], y_limits[1], 6))
    axis.legend(
        title="Most likely state",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.subplots_adjust(left=0.1, right=0.8, bottom=0.12, top=0.88)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_counts(
    dataset: str,
    counts: pd.DataFrame,
    state_names: tuple[str, ...],
    output_path: Path,
    dpi: int,
) -> None:
    _plot_stacked_values(
        dataset,
        counts,
        state_names,
        output_path,
        dpi,
        ylabel="Number of edge segments",
        title_measure="edge-segment MAP-state counts",
    )


def _plot_proportions(
    dataset: str,
    proportions: pd.DataFrame,
    state_names: tuple[str, ...],
    output_path: Path,
    dpi: int,
) -> None:
    _plot_stacked_values(
        dataset,
        proportions,
        state_names,
        output_path,
        dpi,
        ylabel="Proportion of edge segments",
        title_measure="normalized edge-segment MAP-state proportions",
        y_limits=(0.0, 1.0),
    )


def generate_plots(
    data_root: Path = Path("data/troupe"),
    dpi: int = 180,
    output_root: Path = Path("outputs/tls/state_coloring"),
) -> dict[str, Path]:
    state_names = CELL_STATE_ORDER
    datasets = ("TLS", "TLSC")
    loaded_data = {
        dataset: _load_dataset_segments(
            dataset,
            data_root,
            output_root,
            state_names,
        )
        for dataset in datasets
    }
    time_min = min(
        float(segments["time_lower"].min())
        for segments, _ in loaded_data.values()
    )
    time_max = max(
        float(segments["time_upper"].max())
        for segments, _ in loaded_data.values()
    )
    if not time_max > time_min:
        raise ValueError("The combined TLS/TLSC time range must have positive width.")

    outputs = {}
    for dataset, (segments, transition_times) in loaded_data.items():
        bin_edges = make_transition_bin_edges(
            transition_times,
            time_min,
            time_max,
        )
        counts = count_segments_by_time(segments, bin_edges, state_names)
        proportions = normalize_state_counts(counts, state_names)
        output_dir = output_root / dataset / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        counts.to_csv(output_dir / COUNTS_FILENAME, index=False)
        proportions.to_csv(output_dir / PROPORTIONS_FILENAME, index=False)
        output_path = output_dir / PLOT_FILENAME
        _plot_counts(dataset, counts, state_names, output_path, dpi)
        proportions_path = output_dir / PROPORTIONS_PLOT_FILENAME
        _plot_proportions(dataset, proportions, state_names, proportions_path, dpi)
        outputs[dataset] = output_path
        print(
            f"{dataset}: wrote count and proportion plots to {output_dir} "
            f"using {len(counts)} time intervals."
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot most-likely edge-segment state counts across TLS and TLSC "
            "in 101 fitted transition-time intervals."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/troupe"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/tls/state_coloring"),
    )
    parser.add_argument("--dpi", type=int, default=180)
    arguments = parser.parse_args()
    generate_plots(
        arguments.data_root,
        dpi=arguments.dpi,
        output_root=arguments.output_root,
    )


if __name__ == "__main__":
    main()
