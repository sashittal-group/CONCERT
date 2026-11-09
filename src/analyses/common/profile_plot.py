from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import StrMethodFormatter

from src.analyses.common.profile_config import PipelineConfig, load_config


PANELS = [
    ("Net growth rate", "growth", r"$\beta_s - \delta_s$"),
    ("Transition ratio", "transition_ratio", r"$p_{0,S}/p_{0,NT}$"),
    ("Scaled birth rates", "scaled_sampling", r"$\beta_s \, \rho_s$"),
]
PANEL_DETAILS = {
    parameter_type: (title, y_label)
    for title, parameter_type, y_label in PANELS
}
PANEL_DETAILS["transition"] = ("Transition probabilities", r"$p_{s,d}$")
PANEL_DETAILS["sampling_ratio"] = (
    "Relative sampling probabilities",
    r"$\rho_s/\rho_{NC}$",
)
PANEL_DETAILS["birth_ratio"] = ("Relative birth rates", r"$\beta_s/\beta_{NC}$")
PARAMETER_PLOTS = [
    ("birth", "Birth rates", r"$\beta_s$"),
    ("death", "Death rates", r"$\delta_s$"),
    ("transition", "Transition probabilities", r"$p_{s,d}$"),
    ("sampling", "Sampling rates", r"$\rho_s$"),
]
REQUIRED_RANGE_COLUMNS = {
    "parameter",
    "likelihood_fraction",
    "lower_parameter_value",
    "higher_parameter_value",
}
REQUIRED_METADATA_COLUMNS = {
    "parameter",
    "parameter_type",
    "parameter_index",
    "original_value",
}
LEAF_COUNT_REQUIRED_COLUMNS = {
    "dataset",
    "tree_name",
    "unlabeled_leaf_count",
    "total_labeled_leaves",
}


def _load_plot_data(
    config: PipelineConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    ranges_by_dataset = {}
    metadata_by_dataset = {}
    for dataset in config.datasets:
        output_dir = config.dataset_output_dir(dataset)
        ranges_path = output_dir / config.ranges_filename
        metadata_path = output_dir / config.metadata_filename
        ranges = pd.read_csv(ranges_path)
        metadata = pd.read_csv(metadata_path)
        _validate_columns(ranges, REQUIRED_RANGE_COLUMNS, ranges_path)
        _validate_columns(metadata, REQUIRED_METADATA_COLUMNS, metadata_path)
        ranges_by_dataset[dataset] = ranges
        metadata_by_dataset[dataset] = metadata

    expected = set(metadata_by_dataset[config.datasets[0]]["parameter"])
    for dataset in config.datasets[1:]:
        if set(metadata_by_dataset[dataset]["parameter"]) != expected:
            raise ValueError(f"Parameter sets differ for dataset {dataset}")
    return ranges_by_dataset, metadata_by_dataset


def _validate_columns(
    frame: pd.DataFrame,
    required: set[str],
    path: Path,
) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def _parameter_bounds(
    ranges: pd.DataFrame,
    parameter: str,
    fraction: float,
    dataset: str,
) -> tuple[float, float]:
    parameter_ranges = ranges[ranges["parameter"] == parameter]
    fractions = parameter_ranges["likelihood_fraction"].to_numpy(dtype=float)
    matches = np.flatnonzero(np.isclose(fractions, fraction, rtol=1e-12, atol=0.0))
    if matches.size != 1:
        raise ValueError(
            f"Expected one range for {parameter}, fraction {fraction}, {dataset}; "
            f"found {matches.size}"
        )
    row = parameter_ranges.iloc[matches[0]]
    lower = float(row["lower_parameter_value"])
    higher = float(row["higher_parameter_value"])
    if not np.isfinite(lower) or not np.isfinite(higher) or lower > higher:
        raise ValueError(
            f"Invalid parameter range for {parameter}, fraction {fraction}, {dataset}"
        )
    return lower, higher


def _draw_profile_box(
    axis: plt.Axes,
    x_position: float,
    original_value: float,
    box_bounds: tuple[float, float],
    whisker_bounds: tuple[float, float],
    color: str,
    width: float,
) -> None:
    box_lower, box_higher = box_bounds
    whisker_lower, whisker_higher = whisker_bounds
    if not (
        whisker_lower <= box_lower <= original_value <= box_higher <= whisker_higher
    ):
        raise ValueError(
            "Expected the original value and box interval to lie within the "
            "whisker interval"
        )

    cap_width = width * 0.5
    left = x_position - width / 2
    axis.add_patch(
        Rectangle(
            (left, box_lower),
            width,
            box_higher - box_lower,
            facecolor=color,
            edgecolor="#202124",
            alpha=0.58,
            linewidth=1.5,
            zorder=2,
        )
    )
    axis.plot(
        [left, left + width],
        [original_value, original_value],
        color="#171717",
        linewidth=2.0,
        zorder=4,
    )
    axis.plot(
        [x_position, x_position],
        [whisker_lower, box_lower],
        color="#303030",
        linewidth=1.5,
        zorder=1,
    )
    axis.plot(
        [x_position, x_position],
        [box_higher, whisker_higher],
        color="#303030",
        linewidth=1.5,
        zorder=1,
    )
    for end_value in (whisker_lower, whisker_higher):
        axis.plot(
            [x_position - cap_width / 2, x_position + cap_width / 2],
            [end_value, end_value],
            color="#303030",
            linewidth=1.5,
            zorder=1,
        )


def _dataset_offsets(
    datasets: tuple[str, ...],
    maximum_offset: float,
) -> dict[str, float]:
    if not datasets:
        return {}
    if len(datasets) == 1:
        return {datasets[0]: 0.0}
    spacing = 2.0 * maximum_offset / (len(datasets) - 1)
    center = (len(datasets) - 1) / 2.0
    return {
        dataset: (index - center) * spacing
        for index, dataset in enumerate(datasets)
    }


def _profile_box_color(
    config: PipelineConfig,
    parameter_type: str,
    parameter_index: int,
    dataset: str,
    parameter_name: str | None = None,
) -> str:
    if parameter_type == "sampling" and parameter_name is not None:
        if parameter_name.startswith("sampling[tree_"):
            _, _, state_label = (
                parameter_name.removeprefix("sampling[")
                .removesuffix("]")
                .partition(",")
            )
            if state_label:
                state_color = config.plotting.get("state_colors", {}).get(state_label)
                if state_color is not None:
                    return state_color
            return config.plotting["dataset_colors"][dataset]
    if (
        parameter_type == "transition"
        and parameter_name is not None
        and parameter_name.startswith("transition[tree_")
    ):
        edge = parameter_name.removeprefix("transition[").removesuffix("]")
        _, _, transition = edge.partition(",")
        source = transition.partition("->")[0]
        state_color = config.plotting.get("state_colors", {}).get(source)
        if state_color is not None:
            return state_color

    state_index: int | None = None
    if parameter_type in {
        "birth",
        "death",
        "growth",
        "sampling",
        "scaled_sampling",
        "sampling_ratio",
        "birth_ratio",
    }:
        if parameter_type == "sampling":
            state_index = parameter_index % len(config.state_names)
        elif parameter_type == "sampling_ratio":
            tree_ratio = re.fullmatch(
                r"sampling_ratio\[tree_\d+,([^,\]]+)/NC\]",
                parameter_name or "",
            )
            state_index = (
                config.state_names.index(tree_ratio.group(1))
                if tree_ratio is not None
                else parameter_index
            )
        else:
            state_index = parameter_index
    elif parameter_type == "transition":
        state_index = parameter_index // len(config.state_names)

    state_colors = config.plotting.get("state_colors", {})
    if state_index is not None:
        if not 0 <= state_index < len(config.state_names):
            raise ValueError(
                f"Unexpected state index {state_index} for {parameter_type}"
            )
        state_color = state_colors.get(config.state_names[state_index])
        if state_color is not None:
            return state_color
    return config.plotting["dataset_colors"][dataset]


def _plot_panel(
    axis: plt.Axes,
    title: str,
    parameter_type: str,
    y_label: str,
    ranges_by_dataset: dict[str, pd.DataFrame],
    metadata_by_dataset: dict[str, pd.DataFrame],
    config: PipelineConfig,
) -> None:
    first_dataset = config.datasets[0]
    parameters = (
        metadata_by_dataset[first_dataset]
        .query("parameter_type == @parameter_type")
        .sort_values("parameter_index")
        .reset_index(drop=True)
    )
    if parameters.empty:
        raise ValueError(f"No metadata found for parameter type {parameter_type}")

    plot = config.plotting
    box_fraction = float(plot["boxplot_likelihood_fraction"])
    whisker_fraction = float(plot["whisker_likelihood_fraction"])
    font_size = int(plot["font_size"])
    group_labels = []
    x_ticks = []
    excluded_parameters = set(
        plot.get("composite_excluded_parameters", {}).get(parameter_type, [])
    )
    parameters = parameters[~parameters["parameter"].isin(excluded_parameters)]
    if parameters.empty:
        raise ValueError(f"No metadata found for parameter type {parameter_type}")
    if parameter_type == "transition_ratio":
        limits = plot["single_group_x_limits"]
    elif parameter_type in {"sampling_ratio", "birth_ratio"}:
        limits = [
            -0.5,
            (len(parameters) - 1) * float(plot["group_spacing"]) + 0.5,
        ]
    elif parameter_type == "transition":
        left_margin = abs(float(plot["x_limits"][0]))
        right_margin = float(plot["x_limits"][1]) - (
            (len(config.state_names) - 1) * float(plot["group_spacing"])
        )
        limits = [
            -left_margin,
            (len(parameters) - 1) * float(plot["group_spacing"]) + right_margin,
        ]
    elif parameter_type == "sampling" and parameters["parameter"].str.startswith(
        "sampling[tree_"
    ).all():
        limits = [
            -0.5,
            (len(parameters) - 1) * float(plot["group_spacing"]) + 0.5,
        ]
    else:
        limits = plot["x_limits"]
    axis.set_xlim(*limits)
    horizontal_scale = (
        (limits[1] - limits[0]) / (plot["x_limits"][1] - plot["x_limits"][0])
    )
    box_width = float(plot["box_width"]) * horizontal_scale
    dataset_offset = float(plot["dataset_offset"]) * horizontal_scale
    dataset_offsets = _dataset_offsets(config.datasets, dataset_offset)

    for position, row in enumerate(parameters.itertuples(index=False)):
        if parameter_type == "transition_ratio":
            group_center = 0.0
            group_label = plot.get(
                "transition_ratio_group_label",
                f"{config.state_names[1]} / {config.state_names[2]}",
            )
        elif parameter_type in {"sampling_ratio", "birth_ratio"}:
            tree_ratio = (
                re.fullmatch(
                    r"sampling_ratio\[tree_(\d+),([^,\]]+)/NC\]",
                    row.parameter,
                )
                if parameter_type == "sampling_ratio"
                else None
            )
            state_index = (
                config.state_names.index(tree_ratio.group(2))
                if tree_ratio is not None
                else int(row.parameter_index)
            )
            if not 0 <= state_index < len(config.state_names):
                raise ValueError(
                    f"Unexpected state index {state_index} for {row.parameter}"
                )
            group_center = position * float(plot["group_spacing"])
            group_label = f"{config.state_names[state_index]} / NC"
            if tree_ratio is not None:
                tree_index = int(tree_ratio.group(1))
                tree_names = plot.get("tree_names", [])
                tree_name = (
                    tree_names[tree_index]
                    if tree_index < len(tree_names)
                    else f"Tree {tree_index}"
                )
                group_label = f"{tree_name}\n{group_label}"
        elif parameter_type == "transition":
            source, target = divmod(
                int(row.parameter_index),
                len(config.state_names),
            )
            if (
                source == target
                or not 0 <= source < len(config.state_names)
                or not 0 <= target < len(config.state_names)
            ):
                raise ValueError(f"Unexpected transition index {row.parameter_index}")
            group_center = position * float(plot["group_spacing"])
            group_label = (
                f"{config.state_names[source]}->{config.state_names[target]}"
            )
        elif parameter_type == "sampling" and row.parameter.startswith(
            "sampling[tree_"
        ):
            group_center = position * float(plot["group_spacing"])
            tree_state = row.parameter.removeprefix("sampling[").removesuffix("]")
            tree_label, separator, state_label = tree_state.partition(",")
            group_label = f"T{tree_label.removeprefix('tree_')}"
            if separator:
                group_label += f"-{state_label}"
        else:
            state_index = int(row.parameter_index)
            if not 0 <= state_index < len(config.state_names):
                raise ValueError(f"Unexpected state index {state_index} for {row.parameter}")
            group_center = position * float(plot["group_spacing"])
            group_label = config.state_names[state_index]
        group_labels.append((group_center, group_label))

        for dataset in config.datasets:
            x_position = group_center + dataset_offsets[dataset]
            metadata = metadata_by_dataset[dataset]
            matching_metadata = metadata[metadata["parameter"] == row.parameter]
            if len(matching_metadata) != 1:
                raise ValueError(
                    f"Expected one metadata row for {row.parameter} in {dataset}; "
                    f"found {len(matching_metadata)}"
                )
            original_value = float(matching_metadata["original_value"].iloc[0])
            box_bounds = _parameter_bounds(
                ranges_by_dataset[dataset],
                row.parameter,
                box_fraction,
                dataset,
            )
            whisker_bounds = _parameter_bounds(
                ranges_by_dataset[dataset],
                row.parameter,
                whisker_fraction,
                dataset,
            )
            _draw_profile_box(
                axis,
                x_position,
                original_value,
                box_bounds,
                whisker_bounds,
                _profile_box_color(
                    config,
                    parameter_type,
                    int(row.parameter_index),
                    dataset,
                    row.parameter,
                ),
                box_width,
            )
            dataset_label = plot.get("dataset_labels", {}).get(dataset, dataset)
            x_ticks.append((x_position, dataset_label))

    axis.set_title(
        title,
        fontsize=font_size,
        fontweight="normal",
        pad=5,
    )
    if len(config.datasets) == 1:
        axis.set_xticks(
            [x for x, _ in group_labels],
            labels=[label for _, label in group_labels],
        )
    else:
        axis.set_xticks([x for x, _ in x_ticks], labels=[label for _, label in x_ticks])
        for group_center, label in group_labels:
            axis.text(
                group_center,
                -0.10,
                label,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=font_size,
                clip_on=False,
            )
    axis.tick_params(
        axis="x",
        labelsize=int(plot.get("dataset_label_font_size", font_size)),
        length=3,
        pad=1,
    )
    dataset_label_rotation = float(plot.get("dataset_label_rotation", 0))
    if dataset_label_rotation:
        plt.setp(
            axis.get_xticklabels(),
            rotation=dataset_label_rotation,
            ha="right",
            rotation_mode="anchor",
        )
    panel_y_label = plot.get("panel_y_labels", {}).get(parameter_type, y_label)
    axis.set_ylabel(panel_y_label, fontsize=font_size, labelpad=2)
    axis.tick_params(axis="y", labelsize=font_size, length=3, pad=1)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="0.87", linewidth=0.7, linestyle=":")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.1)
    axis.spines["bottom"].set_linewidth(1.1)
    if parameter_type in {"sampling_ratio", "birth_ratio"}:
        axis.axhline(
            1.0,
            color="0.4",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )
    elif parameter_type == "transition_ratio":
        axis.axhline(
            1.0,
            color="0.4",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )


def _load_sampled_leaf_counts(
    config: PipelineConfig,
) -> dict[str, dict[str, int]]:
    count_path = (
        config.preprocessing_root
        / config.plotting["leaf_state_counts_filename"]
    )
    counts = pd.read_csv(count_path)
    _validate_columns(
        counts,
        LEAF_COUNT_REQUIRED_COLUMNS | set(config.state_names),
        count_path,
    )
    count_columns = [
        column
        for column in counts.columns
        if column
        not in {
            "dataset",
            "tree_name",
            "unlabeled_leaf_count",
            "total_labeled_leaves",
        }
    ]
    totals = counts[counts["tree_name"] == "TOTAL"]
    sampled_counts = {}
    for dataset in config.datasets:
        dataset_totals = totals[totals["dataset"] == dataset]
        if len(dataset_totals) != 1:
            raise ValueError(
                f"Expected one total count row for {dataset} in {count_path}; "
                f"found {len(dataset_totals)}"
            )
        row = dataset_totals.iloc[0]
        state_counts = {}
        for column in count_columns:
            value = float(row[column])
            if not np.isfinite(value) or value < 0 or not value.is_integer():
                raise ValueError(
                    f"Invalid leaf count for {dataset}/{column} in {count_path}"
                )
            state_counts[column] = int(value)
        total_labeled = float(row["total_labeled_leaves"])
        if (
            not np.isfinite(total_labeled)
            or total_labeled < 0
            or not total_labeled.is_integer()
            or sum(state_counts.values()) != int(total_labeled)
        ):
            raise ValueError(
                f"State counts do not match total labeled leaves for {dataset} "
                f"in {count_path}"
            )
        sampled_counts[dataset] = state_counts
    return sampled_counts


def _render_population_size_profile(
    config: PipelineConfig,
    ranges_by_dataset: dict[str, pd.DataFrame],
    metadata_by_dataset: dict[str, pd.DataFrame],
    sampled_counts: dict[str, dict[str, int]],
) -> list[Path]:
    plot = config.plotting
    font_size = int(plot["font_size"])
    group_spacing = float(plot["group_spacing"])
    box_width = float(plot["box_width"])
    dataset_offsets = _dataset_offsets(
        config.datasets,
        float(plot["dataset_offset"]),
    )
    box_fraction = float(plot["population_size_boxplot_likelihood_fraction"])
    whisker_fraction = float(plot["whisker_likelihood_fraction"])
    group_centers = [
        index * group_spacing for index, _ in enumerate(config.state_names)
    ]

    figure, axis = plt.subplots(figsize=tuple(plot["parameter_figsize"]))
    x_ticks = []
    for state_index, state in enumerate(config.state_names):
        parameter = f"sampling[{state}]"
        group_center = group_centers[state_index]
        for dataset in config.datasets:
            metadata = metadata_by_dataset[dataset]
            matching_metadata = metadata[metadata["parameter"] == parameter]
            if len(matching_metadata) != 1:
                raise ValueError(
                    f"Expected one {parameter} metadata row for {dataset}; "
                    f"found {len(matching_metadata)}"
                )
            sampling_rate = float(matching_metadata["original_value"].iloc[0])
            count = sampled_counts[dataset][state]
            if not np.isfinite(sampling_rate) or sampling_rate <= 0:
                raise ValueError(
                    f"Invalid sampling rate for {dataset}/{state}: {sampling_rate}"
                )

            rate_box = _parameter_bounds(
                ranges_by_dataset[dataset],
                parameter,
                box_fraction,
                dataset,
            )
            rate_whiskers = _parameter_bounds(
                ranges_by_dataset[dataset],
                parameter,
                whisker_fraction,
                dataset,
            )
            if min(*rate_box, *rate_whiskers) <= 0:
                raise ValueError(
                    f"Sampling-rate profile for {dataset}/{state} is not positive"
                )

            _draw_profile_box(
                axis,
                group_center + dataset_offsets[dataset],
                count / sampling_rate,
                (count / rate_box[1], count / rate_box[0]),
                (count / rate_whiskers[1], count / rate_whiskers[0]),
                plot["dataset_colors"][dataset],
                box_width,
            )
            dataset_label = plot.get("dataset_labels", {}).get(dataset, dataset)
            x_ticks.append((group_center + dataset_offsets[dataset], dataset_label))

    axis.set_title(
        "Estimated population size",
        fontsize=font_size,
        fontweight="normal",
        pad=5,
    )
    axis.set_ylabel("Population size", fontsize=font_size, labelpad=2)
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.set_xlim(
        -abs(float(plot["x_limits"][0])),
        group_centers[-1] + abs(float(plot["x_limits"][0])),
    )
    axis.set_xticks([x for x, _ in x_ticks], labels=[label for _, label in x_ticks])
    for center, state in zip(group_centers, config.state_names):
        axis.text(
            center,
            -0.10,
            state,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=int(plot["font_size"]),
            clip_on=False,
        )
    axis.tick_params(
        axis="x",
        labelsize=int(plot.get("dataset_label_font_size", font_size)),
        length=3,
        pad=1,
    )
    axis.tick_params(axis="y", labelsize=font_size, length=3, pad=1)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="0.87", linewidth=0.7, linestyle=":")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.1)
    axis.spines["bottom"].set_linewidth(1.1)
    figure.subplots_adjust(left=0.14, right=0.985, top=0.94, bottom=0.22)

    output_base = (
        config.figure_root / plot["population_size_output_basename"]
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    formats = plot["population_size_formats"]
    if not isinstance(formats, list) or not formats:
        raise ValueError("population_size_formats must be a non-empty list")
    outputs = []
    for extension in formats:
        output = output_base.with_suffix(f".{extension}")
        figure.savefig(
            output,
            bbox_inches="tight",
            pad_inches=0.01,
            dpi=200,
        )
        outputs.append(output)
    plt.close(figure)
    return outputs


def _load_population_sampling_profiles(
    config: PipelineConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    ranges_by_dataset = {}
    metadata_by_dataset = {}
    for dataset in config.datasets:
        profile_dir = config.data_root / "figures" / dataset
        ranges_path = profile_dir / "parameter_value_ranges.csv"
        metadata_path = profile_dir / "parameter_metadata.csv"
        ranges = pd.read_csv(ranges_path)
        metadata = pd.read_csv(metadata_path)
        _validate_columns(ranges, REQUIRED_RANGE_COLUMNS, ranges_path)
        _validate_columns(metadata, REQUIRED_METADATA_COLUMNS, metadata_path)
        ranges_by_dataset[dataset] = ranges
        metadata_by_dataset[dataset] = metadata
    return ranges_by_dataset, metadata_by_dataset


def _render_sampled_population_size_plot(
    config: PipelineConfig,
    sampled_counts: dict[str, dict[str, int]],
) -> list[Path]:
    plot = config.plotting
    font_size = int(plot["font_size"])
    x_positions = np.arange(len(config.state_names), dtype=float)
    dataset_offsets = _dataset_offsets(config.datasets, 0.12)
    figure, axis = plt.subplots(figsize=tuple(plot["parameter_figsize"]))

    for dataset in config.datasets:
        values = [sampled_counts[dataset][state] for state in config.state_names]
        positions = x_positions + dataset_offsets[dataset]
        axis.scatter(
            positions,
            values,
            s=70,
            color=plot["dataset_colors"][dataset],
            edgecolor="#303030",
            linewidth=0.6,
            label=dataset,
            zorder=3,
        )
        for position, value in zip(positions, values):
            axis.annotate(
                f"{value:,}",
                (position, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    axis.set_title(
        "Sampled population size",
        fontsize=font_size,
        fontweight="normal",
        pad=5,
    )
    axis.set_xlabel("Cell state", fontsize=font_size, labelpad=3)
    axis.set_ylabel("Sampled leaf count", fontsize=font_size, labelpad=2)
    axis.set_xticks(x_positions, labels=config.state_names)
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.set_ylim(bottom=0)
    axis.tick_params(axis="both", labelsize=font_size, length=3, pad=2)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="0.87", linewidth=0.7, linestyle=":")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.1)
    axis.spines["bottom"].set_linewidth(1.1)
    axis.legend(title="Dataset", frameon=False)
    figure.tight_layout()

    output = config.figure_root / (
        f"{plot['sampled_population_size_output_basename']}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    formats = plot["population_size_formats"]
    if not isinstance(formats, list) or not formats:
        raise ValueError("population_size_formats must be a non-empty list")
    outputs = []
    for extension in formats:
        figure_path = output.with_suffix(f".{extension}")
        figure.savefig(
            figure_path,
            bbox_inches="tight",
            pad_inches=0.01,
            dpi=200,
        )
        outputs.append(figure_path)
    plt.close(figure)
    return outputs


def plot_population_size_profiles(config: PipelineConfig) -> list[Path]:
    ranges, metadata = _load_population_sampling_profiles(config)
    sampled_counts = _load_sampled_leaf_counts(config)
    outputs = _render_population_size_profile(
        config,
        ranges,
        metadata,
        sampled_counts,
    )
    outputs.extend(_render_sampled_population_size_plot(config, sampled_counts))
    return outputs


def _render_sensitivity(
    config: PipelineConfig,
    ranges: dict[str, pd.DataFrame],
    metadata: dict[str, pd.DataFrame],
) -> list[Path]:
    if not config.datasets:
        raise ValueError("The sensitivity plot requires at least one dataset")
    panel_types = config.plotting.get(
        "composite_panel_types",
        [parameter_type for _, parameter_type, _ in PANELS],
    )
    if not isinstance(panel_types, list) or not panel_types:
        raise ValueError("composite_panel_types must define at least one panel")
    panel_rows = config.plotting.get("composite_panel_rows")
    if panel_rows is not None and (
        not isinstance(panel_rows, list)
        or not panel_rows
        or any(not isinstance(row, list) or not row for row in panel_rows)
        or [parameter_type for row in panel_rows for parameter_type in row]
        != panel_types
    ):
        raise ValueError(
            "composite_panel_rows must be non-empty rows matching "
            "composite_panel_types"
        )
    panel_width_ratios = config.plotting.get("composite_panel_width_ratios")
    if panel_width_ratios is not None and (
        not isinstance(panel_width_ratios, list)
        or len(panel_width_ratios) != len(panel_types)
        or any(
            not np.isfinite(float(ratio)) or float(ratio) <= 0
            for ratio in panel_width_ratios
        )
    ):
        raise ValueError("composite_panel_width_ratios must match panel count")
    if panel_width_ratios is None:
        panel_width_ratios = [1.0] * len(panel_types)

    panel_gaps = config.plotting.get("composite_panel_gaps")
    if panel_gaps is not None:
        expected_gap_count = (
            sum(len(row) - 1 for row in panel_rows)
            if panel_rows is not None
            else len(panel_types) - 1
        )
        if (
            not isinstance(panel_gaps, list)
            or len(panel_gaps) != expected_gap_count
            or any(
                not np.isfinite(float(gap)) or float(gap) <= 0
                for gap in panel_gaps
            )
        ):
            raise ValueError("composite_panel_gaps must match the number of panel gaps")
    if panel_rows is not None:
        if panel_gaps is None:
            panel_gaps = [0.2] * sum(len(row) - 1 for row in panel_rows)
        figure = plt.figure(figsize=tuple(config.plotting["figsize"]))
        outer_grid = figure.add_gridspec(
            len(panel_rows),
            1,
            hspace=float(config.plotting.get("composite_row_hspace", 0.5)),
        )
        axes = []
        panel_index = 0
        gap_index = 0
        for row_index, row in enumerate(panel_rows):
            row_widths = panel_width_ratios[panel_index:panel_index + len(row)]
            row_gaps = panel_gaps[gap_index:gap_index + len(row) - 1]
            width_ratios = []
            for index, panel_width in enumerate(row_widths):
                width_ratios.append(panel_width)
                if index < len(row_gaps):
                    width_ratios.append(row_gaps[index])
            row_grid = outer_grid[row_index].subgridspec(
                1,
                len(width_ratios),
                width_ratios=width_ratios,
                wspace=0.0,
            )
            axes.extend(
                figure.add_subplot(row_grid[0, index * 2])
                for index in range(len(row))
            )
            panel_index += len(row)
            gap_index += len(row_gaps)
    elif panel_gaps is not None:
        figure = plt.figure(figsize=tuple(config.plotting["figsize"]))
        width_ratios = []
        for index, panel_width in enumerate(panel_width_ratios):
            width_ratios.append(panel_width)
            if index < len(panel_gaps):
                width_ratios.append(panel_gaps[index])
        grid = figure.add_gridspec(
            1,
            len(width_ratios),
            width_ratios=width_ratios,
            wspace=0.0,
        )
        axes = [
            figure.add_subplot(grid[0, index * 2])
            for index in range(len(panel_types))
        ]
    else:
        figure, axes = plt.subplots(
            1,
            len(panel_types),
            figsize=tuple(config.plotting["figsize"]),
            gridspec_kw={"width_ratios": panel_width_ratios},
        )
    for axis, parameter_type in zip(axes, panel_types):
        if not isinstance(parameter_type, str) or parameter_type not in PANEL_DETAILS:
            raise ValueError(f"Unsupported composite panel type: {parameter_type}")
        title, y_label = PANEL_DETAILS[parameter_type]
        if (
            parameter_type == "transition"
            and config.plotting.get("transition_plot_style") == "heatmap"
        ):
            _draw_transition_heatmap(
                axis,
                metadata[config.datasets[0]],
                config,
                title,
                add_colorbar=True,
            )
        else:
            _plot_panel(
                axis,
                title,
                parameter_type,
                y_label,
                ranges,
                metadata,
                config,
            )
    figure.subplots_adjust(
        left=0.06,
        right=0.995,
        top=0.95,
        bottom=float(config.plotting.get("composite_bottom", 0.16)),
        wspace=(
            0.0
            if panel_gaps is not None
            else float(config.plotting.get("composite_wspace", 0.22))
        ),
        hspace=(
            float(config.plotting.get("composite_row_hspace", 0.5))
            if panel_rows is not None
            else 0.2
        ),
    )

    config.figure_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in config.plotting["formats"]:
        output = config.figure_root / (
            f"{config.plotting['output_basename']}.{extension}"
        )
        figure.savefig(output, bbox_inches="tight", pad_inches=0.01)
        outputs.append(output)
    plt.close(figure)
    return outputs


def _parameter_group_label(
    parameter_name: str,
    parameter_index: int,
    parameter_type: str,
    config: PipelineConfig,
) -> str:
    if parameter_type in {"birth", "death"}:
        return config.state_names[parameter_index]
    if parameter_type == "transition":
        if parameter_name.startswith("transition[tree_"):
            transition = parameter_name.removeprefix("transition[").removesuffix("]")
            tree_label, _, edge = transition.partition(",")
            tree_index = int(tree_label.removeprefix("tree_"))
            tree_names = config.plotting.get("tree_names", [])
            tree_name = (
                tree_names[tree_index]
                if tree_index < len(tree_names)
                else f"T{tree_index}"
            )
            return f"{tree_name} {edge.replace('->', ' to ')}"
        edge_label = parameter_name.removeprefix("transition[").removesuffix("]")
        if "->" in edge_label:
            return edge_label.replace("->", " to ")
        if not 0 < parameter_index < len(config.state_names):
            raise ValueError(f"Unexpected transition target index {parameter_index}")
        return f"{config.state_names[0]} to {config.state_names[parameter_index]}"
    if parameter_name == "sampling[shared]":
        return "Shared"
    if parameter_type == "sampling" and parameter_name.startswith("sampling[tree_"):
        sampling_label = parameter_name.removeprefix("sampling[").removesuffix("]")
        tree_label, _, state_label = sampling_label.partition(",")
        tree_index = tree_label.removeprefix("tree_")
        tree_names = config.plotting.get("tree_names", [])
        tree_name = (
            tree_names[int(tree_index)]
            if int(tree_index) < len(tree_names)
            else f"T{tree_index}"
        )
        return f"{tree_name}-{state_label}" if state_label else tree_name
    if parameter_type == "sampling_ratio":
        tree_match = re.fullmatch(
            r"sampling_ratio\[tree_(\d+),([^,\]]+)/NC\]",
            parameter_name,
        )
        if tree_match is not None:
            tree_index = int(tree_match.group(1))
            tree_names = config.plotting.get("tree_names", [])
            tree_name = (
                tree_names[tree_index]
                if tree_index < len(tree_names)
                else f"T{tree_index}"
            )
            return f"{tree_name} {tree_match.group(2)} / NC"
        return f"{config.state_names[parameter_index]} / NC"
    if parameter_type == "birth_ratio":
        return f"{config.state_names[parameter_index]} / NC"
    return config.state_names[parameter_index]


def _transition_probability_matrix(
    metadata: pd.DataFrame,
    config: PipelineConfig,
    tree_index: int | None = None,
) -> np.ndarray:
    num_states = len(config.state_names)
    transition_names = metadata.loc[
        metadata["parameter_type"] == "transition",
        "parameter",
    ]
    tree_indices = sorted(
        {
            int(match.group(1))
            for parameter in transition_names
            if (match := re.fullmatch(r"transition\[tree_(\d+),.+\]", parameter))
        }
    )
    if tree_indices and tree_index is None:
        return np.mean(
            [
                _transition_probability_matrix(metadata, config, index)
                for index in tree_indices
            ],
            axis=0,
        )
    matrix = np.zeros((num_states, num_states), dtype=float)
    transitions = metadata[metadata["parameter_type"] == "transition"]
    for row in transitions.itertuples(index=False):
        tree_match = re.fullmatch(
            r"transition\[tree_(\d+),([^,\]]+)->([^,\]]+)\]",
            row.parameter,
        )
        if tree_match is not None:
            row_tree_index = int(tree_match.group(1))
            if tree_index is not None and row_tree_index != tree_index:
                continue
            source = config.state_names.index(tree_match.group(2))
            target = config.state_names.index(tree_match.group(3))
        else:
            source, target = divmod(int(row.parameter_index), num_states)
        probability = float(row.original_value)
        if (
            source == target
            or not 0 <= source < num_states
            or not 0 <= target < num_states
            or not np.isfinite(probability)
            or not 0.0 <= probability <= 1.0
        ):
            raise ValueError(f"Invalid transition metadata for {row.parameter}")
        matrix[source, target] = probability

    for source in range(num_states):
        outgoing_probability = float(matrix[source].sum())
        if outgoing_probability > 1.0 + 1e-8:
            raise ValueError(
                f"Transition probabilities from {config.state_names[source]} "
                "sum to more than one"
            )
        matrix[source, source] = max(0.0, 1.0 - outgoing_probability)
    return matrix


def _draw_transition_heatmap(
    axis: plt.Axes,
    metadata: pd.DataFrame,
    config: PipelineConfig,
    title: str,
    *,
    add_colorbar: bool,
    tree_index: int | None = None,
    norm=None,
):
    matrix = _transition_probability_matrix(metadata, config, tree_index)
    positive = matrix[matrix > 0.0]
    if positive.size == 0:
        raise ValueError("Transition matrix has no positive probabilities")
    lower = float(positive.min())
    upper = float(positive.max())
    if lower == upper:
        lower = upper / 10.0

    masked = np.ma.masked_where(matrix <= 0.0, matrix)
    image = axis.imshow(
        masked,
        cmap=config.plotting.get("transition_heatmap_colormap", "Blues"),
        norm=norm if norm is not None else LogNorm(vmin=lower, vmax=upper),
        interpolation="nearest",
    )
    font_size = int(config.plotting["font_size"])
    for source in range(len(config.state_names)):
        for target in range(len(config.state_names)):
            probability = matrix[source, target]
            label = f"{probability:.2g}" if probability > 0.0 else "0"
            normalized = image.norm(probability) if probability > 0.0 else 0.0
            color = "white" if normalized > 0.62 else "#202124"
            axis.text(
                target,
                source,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=max(font_size - 4, 8),
            )

    states = config.state_names
    axis.set_xticks(range(len(states)), labels=states)
    axis.set_yticks(range(len(states)), labels=states)
    axis.set_xlabel(
        config.plotting.get("transition_heatmap_x_label", "Destination"),
        fontsize=font_size,
        labelpad=3,
    )
    axis.set_ylabel(
        config.plotting.get("transition_heatmap_y_label", "Source"),
        fontsize=font_size,
        labelpad=3,
    )
    tree_names = config.plotting.get("tree_names", [])
    if tree_index is not None and tree_index < len(tree_names):
        title = f"{tree_names[tree_index]}: {title}"
    axis.set_title(title, fontsize=font_size, fontweight="normal", pad=5)
    axis.set_aspect("equal")
    axis.set_xticks(np.arange(-0.5, len(states), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(states), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.tick_params(axis="both", labelsize=max(font_size - 2, 8), length=0, pad=2)
    axis.spines[:].set_visible(False)
    if add_colorbar:
        colorbar = axis.figure.colorbar(
            image,
            ax=axis,
            fraction=0.046,
            pad=0.04,
        )
        colorbar.set_label(
            "Transition probability (log scale)",
            fontsize=max(font_size - 3, 8),
        )
        colorbar.ax.tick_params(labelsize=max(font_size - 4, 8), length=2)
    return image


def _render_transition_heatmap_plot(
    config: PipelineConfig,
    metadata_by_dataset: dict[str, pd.DataFrame],
) -> list[Path]:
    dataset = config.datasets[0]
    plot = config.plotting
    metadata = metadata_by_dataset[dataset]
    tree_specific = metadata["parameter"].str.match(
        r"transition\[tree_\d+,"
    ).any()
    if tree_specific:
        tree_count = len(config.plotting.get("tree_names", []))
        if not tree_count:
            tree_count = int(
                metadata.loc[
                    metadata["parameter_type"] == "transition",
                    "parameter",
                ]
                .str.extract(r"tree_(\d+)", expand=False)
                .astype(int)
                .max()
            ) + 1
        rows = int(math.ceil(tree_count / 3))
        figure, axes = plt.subplots(
            rows,
            3,
            figsize=(9.0, 3.1 * rows),
            squeeze=False,
        )
        matrices = [
            _transition_probability_matrix(metadata, config, index)
            for index in range(tree_count)
        ]
        positive = np.concatenate([matrix[matrix > 0.0] for matrix in matrices])
        lower, upper = float(positive.min()), float(positive.max())
        if lower == upper:
            lower = upper / 10.0
        norm = LogNorm(vmin=lower, vmax=upper)
        images = []
        for tree_index, axis in enumerate(axes.flat):
            if tree_index >= tree_count:
                axis.set_visible(False)
                continue
            images.append(
                _draw_transition_heatmap(
                    axis,
                    metadata,
                    config,
                    "Transition probabilities",
                    add_colorbar=False,
                    tree_index=tree_index,
                    norm=norm,
                )
            )
        figure.colorbar(
            images[0],
            ax=[axis for axis in axes.flat if axis.get_visible()],
            fraction=0.025,
            pad=0.02,
            label="Transition probability (log scale)",
        )
        figure.subplots_adjust(left=0.06, right=0.92, top=0.94, bottom=0.08)
    else:
        figure_size = tuple(
            plot.get("transition_heatmap_figsize", plot["parameter_figsize"])
        )
        figure, axis = plt.subplots(figsize=figure_size)
        _draw_transition_heatmap(
            axis,
            metadata,
            config,
            "Transition probabilities",
            add_colorbar=True,
        )
        figure.subplots_adjust(left=0.18, right=0.86, top=0.92, bottom=0.12)

    outputs = []
    output_basename = plot["parameter_output_basenames"]["transition"]
    for extension in plot["formats"]:
        output = config.figure_root / f"{output_basename}.{extension}"
        figure.savefig(output, bbox_inches="tight", pad_inches=0.01)
        outputs.append(output)
    plt.close(figure)
    return outputs


def _render_parameter_plot(
    config: PipelineConfig,
    ranges_by_dataset: dict[str, pd.DataFrame],
    metadata_by_dataset: dict[str, pd.DataFrame],
    parameter_type: str,
    title: str,
    y_label: str,
) -> list[Path]:
    first_dataset = config.datasets[0]
    parameters = (
        metadata_by_dataset[first_dataset]
        .query("parameter_type == @parameter_type")
        .sort_values("parameter_index")
        .reset_index(drop=True)
    )
    if parameters.empty:
        raise ValueError(f"No metadata found for parameter type {parameter_type}")

    plot = config.plotting
    group_spacing = float(plot["group_spacing"])
    group_centers = [index * group_spacing for index in range(len(parameters))]
    dataset_label_font_size = int(
        plot.get("dataset_label_font_size", plot["font_size"])
    )
    if parameter_type == "transition" and plot.get("transition_plot_style") == "heatmap":
        raise ValueError("Transition heatmaps are rendered by the matrix plotter")
    if len(group_centers) == 1:
        limits = plot["single_group_x_limits"]
    else:
        left_margin = abs(float(plot["x_limits"][0]))
        limits = [
            -left_margin,
            group_centers[-1] + left_margin,
        ]
    box_width = float(plot["box_width"])
    dataset_offset = float(plot["dataset_offset"])
    dataset_offsets = _dataset_offsets(config.datasets, dataset_offset)

    figure_size = (
        plot.get("transition_parameter_figsize", plot["parameter_figsize"])
        if parameter_type == "transition"
        else plot["parameter_figsize"]
    )
    figure, axis = plt.subplots(figsize=tuple(figure_size))
    axis.set_xlim(*limits)
    axis.set_title(title, fontsize=int(plot["font_size"]), fontweight="normal", pad=5)
    axis.set_ylabel(y_label, fontsize=int(plot["font_size"]), labelpad=2)
    group_labels = []
    x_ticks = []

    for position, row in enumerate(parameters.itertuples(index=False)):
        group_center = group_centers[position]
        group_labels.append(
            (
                group_center,
                _parameter_group_label(
                    row.parameter,
                    int(row.parameter_index),
                    parameter_type,
                    config,
                ),
            )
        )
        for dataset in config.datasets:
            x_position = group_center + dataset_offsets[dataset]
            matching_metadata = metadata_by_dataset[dataset][
                metadata_by_dataset[dataset]["parameter"] == row.parameter
            ]
            if len(matching_metadata) != 1:
                raise ValueError(
                    f"Expected one metadata row for {row.parameter} in {dataset}; "
                    f"found {len(matching_metadata)}"
                )
            original_value = float(matching_metadata["original_value"].iloc[0])
            box_bounds = _parameter_bounds(
                ranges_by_dataset[dataset],
                row.parameter,
                float(plot["boxplot_likelihood_fraction"]),
                dataset,
            )
            whisker_bounds = _parameter_bounds(
                ranges_by_dataset[dataset],
                row.parameter,
                float(plot["whisker_likelihood_fraction"]),
                dataset,
            )
            _draw_profile_box(
                axis,
                x_position,
                original_value,
                box_bounds,
                whisker_bounds,
                _profile_box_color(
                    config,
                    parameter_type,
                    int(row.parameter_index),
                    dataset,
                    row.parameter,
                ),
                box_width,
            )
            dataset_label = plot.get("dataset_labels", {}).get(dataset, dataset)
            x_ticks.append((x_position, dataset_label))

    if parameter_type in plot.get("log_scale_parameter_types", []):
        axis.set_yscale("log")
    if len(config.datasets) == 1:
        axis.set_xticks(
            [x for x, _ in group_labels],
            labels=[label for _, label in group_labels],
        )
    else:
        axis.set_xticks([x for x, _ in x_ticks], labels=[label for _, label in x_ticks])
        for group_center, label in group_labels:
            axis.text(
                group_center,
                -0.10,
                label,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=int(plot["font_size"]),
                clip_on=False,
            )
    axis.tick_params(
        axis="x",
        labelsize=(
            int(plot["font_size"])
            if len(config.datasets) == 1
            else dataset_label_font_size
        ),
        length=3,
        pad=1,
    )
    tree_transition_profiles = (
        parameter_type == "transition"
        and parameters["parameter"].str.match(r"transition\[tree_\d+,").any()
    )
    if tree_transition_profiles:
        plt.setp(
            axis.get_xticklabels(),
            rotation=55,
            ha="right",
            rotation_mode="anchor",
        )
    if parameter_type == "sampling" and any(
        parameter.startswith("sampling[tree_")
        for parameter in parameters["parameter"]
    ):
        plt.setp(
            axis.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
    axis.tick_params(axis="y", labelsize=int(plot["font_size"]), length=3, pad=1)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="0.87", linewidth=0.7, linestyle=":")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.1)
    axis.spines["bottom"].set_linewidth(1.1)
    figure.subplots_adjust(
        left=0.12,
        right=0.985,
        top=0.94,
        bottom=(
            0.38
            if tree_transition_profiles
            else (
                0.32
                if parameter_type == "sampling"
                and any(
                    parameter.startswith("sampling[tree_")
                    for parameter in parameters["parameter"]
                )
                else 0.22
            )
        ),
    )

    output_basename = plot["parameter_output_basenames"][parameter_type]
    outputs = []
    for extension in plot["formats"]:
        output = config.figure_root / f"{output_basename}.{extension}"
        figure.savefig(output, bbox_inches="tight", pad_inches=0.01)
        outputs.append(output)
    plt.close(figure)
    return outputs


def _render_grouped_sampling_profiles(
    config: PipelineConfig,
    ranges_by_dataset: dict[str, pd.DataFrame],
    metadata_by_dataset: dict[str, pd.DataFrame],
) -> list[Path]:
    dataset = config.datasets[0]
    metadata = metadata_by_dataset[dataset]
    parameters = metadata[
        metadata["parameter_type"] == "sampling"
    ].sort_values("parameter_index")
    parsed_parameters = []
    for row in parameters.itertuples(index=False):
        match = re.fullmatch(r"sampling\[tree_(\d+),([^,\]]+)\]", row.parameter)
        if match is None:
            raise ValueError(
                f"Invalid tree/state sampling parameter name: {row.parameter}"
            )
        parsed_parameters.append((row, int(match.group(1)), match.group(2)))
    tree_indices = sorted({tree_index for _, tree_index, _ in parsed_parameters})
    if tree_indices != list(range(len(tree_indices))):
        raise ValueError("Tree/state sampling profiles must use contiguous tree indices")
    state_names = list(config.state_names)
    tree_names = config.plotting.get("tree_names", [])
    if tree_names and len(tree_names) != len(tree_indices):
        raise ValueError(
            "Configured tree names must match the inferred sampling tree count"
        )
    if len(parsed_parameters) != len(tree_indices) * len(state_names):
        raise ValueError(
            "Expected one tree/state sampling profile for each combination"
        )
    parameter_by_tree_state = {
        (tree_index, state_name): row
        for row, tree_index, state_name in parsed_parameters
    }
    if set(parameter_by_tree_state) != {
        (tree_index, state_name)
        for tree_index in tree_indices
        for state_name in state_names
    }:
        raise ValueError("Tree/state sampling profiles are incomplete or duplicated")

    plot = config.plotting
    font_size = int(plot["font_size"])
    box_fraction = float(plot["boxplot_likelihood_fraction"])
    whisker_fraction = float(plot["whisker_likelihood_fraction"])
    figure_root = config.figure_root
    figure_root.mkdir(parents=True, exist_ok=True)
    output_base = plot["parameter_output_basenames"]["sampling"]
    outputs = []

    for group_by in ("tree", "state"):
        if group_by == "tree":
            group_values = tree_indices
            subgroup_values = state_names
            group_labels = [
                tree_names[index] if tree_names else f"T{index}"
                for index in tree_indices
            ]
            subgroup_labels = state_names
            group_spacing = 1.6
            subgroup_spacing = 0.42
            box_width = 0.25
        else:
            group_values = state_names
            subgroup_values = tree_indices
            group_labels = state_names
            subgroup_labels = [
                tree_names[index] if tree_names else f"T{index}"
                for index in tree_indices
            ]
            group_spacing = 3.6
            subgroup_spacing = 0.5
            box_width = 0.22

        figure_width = max(
            10.0,
            group_spacing * (len(group_values) - 1) + 5.0,
        )
        figure, axis = plt.subplots(figsize=(figure_width, 4.5))
        group_centers = [
            group_index * group_spacing
            for group_index in range(len(group_values))
        ]

        for group_index, group_value in enumerate(group_values):
            center = group_centers[group_index]
            offsets = [
                (subgroup_index - (len(subgroup_values) - 1) / 2.0)
                * subgroup_spacing
                for subgroup_index in range(len(subgroup_values))
            ]
            for subgroup_index, subgroup_value in enumerate(subgroup_values):
                if group_by == "tree":
                    tree_index = group_value
                    state_name = subgroup_value
                else:
                    tree_index = subgroup_value
                    state_name = group_value
                parameter = parameter_by_tree_state[(tree_index, state_name)]
                box_bounds = _parameter_bounds(
                    ranges_by_dataset[dataset],
                    parameter.parameter,
                    box_fraction,
                    dataset,
                )
                whisker_bounds = _parameter_bounds(
                    ranges_by_dataset[dataset],
                    parameter.parameter,
                    whisker_fraction,
                    dataset,
                )
                _draw_profile_box(
                    axis,
                    center + offsets[subgroup_index],
                    float(parameter.original_value),
                    box_bounds,
                    whisker_bounds,
                    _profile_box_color(
                        config,
                        "sampling",
                        int(parameter.parameter_index),
                        dataset,
                        parameter.parameter,
                    ),
                    box_width,
                )

            axis.text(
                center,
                -0.18 if group_by == "tree" else -0.40,
                group_labels[group_index],
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=font_size,
                clip_on=False,
            )

        tick_positions = []
        tick_labels = []
        for group_index in range(len(group_values)):
            center = group_centers[group_index]
            for subgroup_index, subgroup_label in enumerate(subgroup_labels):
                tick_positions.append(
                    center
                    + (subgroup_index - (len(subgroup_values) - 1) / 2.0)
                    * subgroup_spacing
                )
                tick_labels.append(subgroup_label)
        axis.set_xticks(tick_positions, labels=tick_labels)
        if group_by == "state":
            plt.setp(
                axis.get_xticklabels(),
                rotation=45,
                ha="right",
                rotation_mode="anchor",
            )
        axis.set_xlim(
            group_centers[0]
            - subgroup_spacing * (len(subgroup_values) - 1) / 2.0
            - box_width / 2.0
            - 0.3,
            group_centers[-1]
            + subgroup_spacing * (len(subgroup_values) - 1) / 2.0
            + box_width / 2.0
            + 0.3,
        )
        axis.set_ylabel(r"$\rho_s$", fontsize=font_size, labelpad=2)
        axis.set_title(
            "Sampling rates grouped by tree"
            if group_by == "tree"
            else "Sampling rates grouped by cell state",
            fontsize=font_size,
            fontweight="normal",
            pad=5,
        )
        axis.tick_params(axis="x", labelsize=font_size - 4, length=3, pad=1)
        axis.tick_params(axis="y", labelsize=font_size, length=3, pad=1)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="0.87", linewidth=0.7, linestyle=":")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.1)
        axis.spines["bottom"].set_linewidth(1.1)
        figure.subplots_adjust(
            left=0.10,
            right=0.985,
            top=0.90,
            bottom=0.25 if group_by == "tree" else 0.40,
        )

        for extension in plot["formats"]:
            output = figure_root / f"{output_base}_by_{group_by}.{extension}"
            figure.savefig(output, bbox_inches="tight", pad_inches=0.01)
            outputs.append(output)
        plt.close(figure)
    return outputs


def plot_profile_data(
    config: PipelineConfig,
    ranges: dict[str, pd.DataFrame],
    metadata: dict[str, pd.DataFrame],
) -> list[Path]:
    with matplotlib.rc_context(
        {
            "font.family": config.plotting["font_family"],
            "mathtext.fontset": config.plotting["math_fontset"],
        }
    ):
        outputs = _render_sensitivity(config, ranges, metadata)
        for parameter_type, title, y_label in PARAMETER_PLOTS:
            if (
                parameter_type == "transition"
                and config.plotting.get("transition_plot_style") == "heatmap"
            ):
                outputs.extend(
                    _render_transition_heatmap_plot(config, metadata)
                )
                tree_transition_profiles = any(
                    frame["parameter"].str.match(r"transition\[tree_\d+,").any()
                    for frame in metadata.values()
                )
                if tree_transition_profiles:
                    plotting = dict(config.plotting)
                    output_names = dict(plotting["parameter_output_basenames"])
                    output_names["transition"] = (
                        f"{output_names['transition']}_profile_intervals"
                    )
                    plotting["parameter_output_basenames"] = output_names
                    plotting["transition_plot_style"] = "profile_intervals"
                    transition_profile_config = replace(
                        config,
                        plotting=plotting,
                    )
                    outputs.extend(
                        _render_parameter_plot(
                            transition_profile_config,
                            ranges,
                            metadata,
                            parameter_type,
                            title,
                            y_label,
                        )
                    )
                continue
            outputs.extend(
                _render_parameter_plot(
                    config,
                    ranges,
                    metadata,
                    parameter_type,
                    title,
                    y_label,
                )
            )
        if any(
            re.fullmatch(r"sampling\[tree_\d+,[^,\]]+\]", str(parameter))
            for frame in metadata.values()
            for parameter in frame["parameter"]
        ):
            outputs.extend(
                _render_grouped_sampling_profiles(config, ranges, metadata)
            )
        return outputs


def plot_sensitivity(config: PipelineConfig) -> list[Path]:
    ranges, metadata = _load_plot_data(config)
    return plot_profile_data(config, ranges, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot TLS/TLSC profile-likelihood interval boxplots."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config.yaml"),
    )
    parser.add_argument(
        "--population-size-only",
        action="store_true",
        help="Plot population-size profiles and sampled leaf-count dot plots.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.population_size_only:
        outputs = plot_population_size_profiles(config)
    else:
        outputs = plot_sensitivity(config)
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
