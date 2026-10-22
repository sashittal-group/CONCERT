from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from src.results.tls.config import PipelineConfig, load_config


PANELS = [
    ("Net growth rate", "growth", r"$\beta_s - \delta_s$"),
    ("Transition ratio", "transition_ratio", r"$p_{0,S}/p_{0,NT}$"),
    ("Scaled sampling", "scaled_sampling", r"$\beta_s \, \rho_s$"),
]
PANEL_DETAILS = {
    parameter_type: (title, y_label)
    for title, parameter_type, y_label in PANELS
}
PANEL_DETAILS["transition"] = ("Transition probabilities", r"$p_{s,d}$")
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
    elif parameter_type == "transition":
        left_margin = abs(float(plot["x_limits"][0]))
        right_margin = float(plot["x_limits"][1]) - (
            (len(config.state_names) - 1) * float(plot["group_spacing"])
        )
        limits = [
            -left_margin,
            (len(parameters) - 1) * float(plot["group_spacing"]) + right_margin,
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
                plot["dataset_colors"][dataset],
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
    panel_y_label = plot.get("panel_y_labels", {}).get(parameter_type, y_label)
    axis.set_ylabel(panel_y_label, fontsize=font_size, labelpad=2)
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
    if not config.datasets:
        raise ValueError("The sensitivity plot requires at least one dataset")
    figure, axes = plt.subplots(1, 3, figsize=tuple(config.plotting["figsize"]))
    panel_types = config.plotting.get(
        "composite_panel_types",
        [parameter_type for _, parameter_type, _ in PANELS],
    )
    if not isinstance(panel_types, list) or len(panel_types) != len(axes):
        raise ValueError("composite_panel_types must define one type per panel")
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
        edge_label = parameter_name.removeprefix("transition[").removesuffix("]")
        if "->" in edge_label:
            return edge_label.replace("->", " to ")
        if not 0 < parameter_index < len(config.state_names):
            raise ValueError(f"Unexpected transition target index {parameter_index}")
        return f"{config.state_names[0]} to {config.state_names[parameter_index]}"
    if parameter_name == "sampling[shared]":
        return "Shared"
    return config.state_names[parameter_index]


def _transition_probability_matrix(
    metadata: pd.DataFrame,
    config: PipelineConfig,
) -> np.ndarray:
    num_states = len(config.state_names)
    matrix = np.zeros((num_states, num_states), dtype=float)
    transitions = metadata[metadata["parameter_type"] == "transition"]
    for row in transitions.itertuples(index=False):
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
) -> None:
    matrix = _transition_probability_matrix(metadata, config)
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
        norm=LogNorm(vmin=lower, vmax=upper),
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


def _render_transition_heatmap_plot(
    config: PipelineConfig,
    metadata_by_dataset: dict[str, pd.DataFrame],
) -> list[Path]:
    dataset = config.datasets[0]
    plot = config.plotting
    figure_size = tuple(
        plot.get("transition_heatmap_figsize", plot["parameter_figsize"])
    )
    figure, axis = plt.subplots(figsize=figure_size)
    _draw_transition_heatmap(
        axis,
        metadata_by_dataset[dataset],
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
                plot["dataset_colors"][dataset],
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
    args = parser.parse_args()
    for output in plot_sensitivity(load_config(args.config)):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
