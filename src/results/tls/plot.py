from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from src.results.tls.config import PipelineConfig, load_config


PANELS = [
    ("Net growth rate", "growth", r"$\beta_s - \delta_s$"),
    ("Transition ratio", "transition_ratio", r"$p_{0,S}/p_{0,NT}$"),
    ("Scaled sampling", "scaled_sampling", r"$\beta_s \, \rho_s$"),
]
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
    limits = (
        plot["single_group_x_limits"]
        if parameter_type == "transition_ratio"
        else plot["x_limits"]
    )
    axis.set_xlim(*limits)
    horizontal_scale = (
        (limits[1] - limits[0]) / (plot["x_limits"][1] - plot["x_limits"][0])
    )
    box_width = float(plot["box_width"]) * horizontal_scale
    dataset_offset = float(plot["dataset_offset"]) * horizontal_scale

    for position, row in enumerate(parameters.itertuples(index=False)):
        if parameter_type == "transition_ratio":
            group_center = 0.0
            group_label = f"{config.state_names[1]} / {config.state_names[2]}"
        else:
            state_index = int(row.parameter_index)
            if not 0 <= state_index < len(config.state_names):
                raise ValueError(f"Unexpected state index {state_index} for {row.parameter}")
            group_center = position * float(plot["group_spacing"])
            group_label = config.state_names[state_index]
        group_labels.append((group_center, group_label))

        for dataset_index, dataset in enumerate(config.datasets):
            direction = -1.0 if dataset_index == 0 else 1.0
            x_position = group_center + direction * dataset_offset
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
                plot["dataset_colors"][dataset],
                box_width,
            )
            x_ticks.append((x_position, dataset))

    axis.set_title(
        title,
        fontsize=font_size,
        fontweight="normal",
        pad=5,
    )
    axis.set_xticks([x for x, _ in x_ticks], labels=[label for _, label in x_ticks])
    axis.tick_params(axis="x", labelsize=font_size, length=3, pad=1)
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
    axis.set_ylabel(y_label, fontsize=font_size, labelpad=2)
    axis.tick_params(axis="y", labelsize=font_size, length=3, pad=1)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="0.87", linewidth=0.7, linestyle=":")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.1)
    axis.spines["bottom"].set_linewidth(1.1)
    if parameter_type == "transition_ratio":
        axis.axhline(
            1.0,
            color="0.4",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )


def _render_sensitivity(
    config: PipelineConfig,
    ranges: dict[str, pd.DataFrame],
    metadata: dict[str, pd.DataFrame],
) -> list[Path]:
    if len(config.datasets) != 2:
        raise ValueError("The comparison plot requires exactly two configured datasets")
    figure, axes = plt.subplots(1, 3, figsize=tuple(config.plotting["figsize"]))
    for axis, (title, parameter_type, y_label) in zip(axes, PANELS):
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
        bottom=0.16,
        wspace=0.22,
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
        if not 0 < parameter_index < len(config.state_names):
            raise ValueError(f"Unexpected transition target index {parameter_index}")
        return f"{config.state_names[0]} to {config.state_names[parameter_index]}"
    if parameter_name == "sampling[shared]":
        return "Shared"
    return config.state_names[parameter_index]


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
    if len(group_centers) == 1:
        limits = plot["single_group_x_limits"]
    else:
        left_margin = abs(float(plot["x_limits"][0]))
        right_margin = float(plot["x_limits"][1]) - (
            (len(config.state_names) - 1) * group_spacing
        )
        limits = [
            -left_margin,
            group_centers[-1] + right_margin,
        ]
    box_width = float(plot["box_width"])
    dataset_offset = float(plot["dataset_offset"])

    figure, axis = plt.subplots(figsize=tuple(plot["parameter_figsize"]))
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
        for dataset_index, dataset in enumerate(config.datasets):
            direction = -1.0 if dataset_index == 0 else 1.0
            x_position = group_center + direction * dataset_offset
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
                plot["dataset_colors"][dataset],
                box_width,
            )
            x_ticks.append((x_position, dataset))

    axis.set_xticks([x for x, _ in x_ticks], labels=[label for _, label in x_ticks])
    axis.tick_params(axis="x", labelsize=int(plot["font_size"]), length=3, pad=1)
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
    axis.tick_params(axis="y", labelsize=int(plot["font_size"]), length=3, pad=1)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="0.87", linewidth=0.7, linestyle=":")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.1)
    axis.spines["bottom"].set_linewidth(1.1)
    figure.subplots_adjust(left=0.12, right=0.985, top=0.94, bottom=0.22)

    output_basename = plot["parameter_output_basenames"][parameter_type]
    outputs = []
    for extension in plot["formats"]:
        output = config.figure_root / f"{output_basename}.{extension}"
        figure.savefig(output, bbox_inches="tight", pad_inches=0.01)
        outputs.append(output)
    plt.close(figure)
    return outputs


def plot_sensitivity(config: PipelineConfig) -> list[Path]:
    with matplotlib.rc_context(
        {
            "font.family": config.plotting["font_family"],
            "mathtext.fontset": config.plotting["math_fontset"],
        }
    ):
        ranges, metadata = _load_plot_data(config)
        outputs = _render_sensitivity(config, ranges, metadata)
        for parameter_type, title, y_label in PARAMETER_PLOTS:
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
        return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot TLS/TLSC profile-likelihood interval boxplots."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config.yaml"),
    )
    args = parser.parse_args()
    for output in plot_sensitivity(load_config(args.config)):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
