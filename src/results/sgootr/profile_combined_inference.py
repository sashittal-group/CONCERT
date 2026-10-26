from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import math
from pathlib import Path
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.dataprocessor import load_edges, load_nodes
from src.main2_given_transitions import (
    find_tree_file_pairs,
    load_per_edge_transition_data,
)
from src.results.profile_tools import ParameterProfile, profile_rows
from src.results.sgootr.preprocess import (
    _birth_ratio_profiles,
    _profile_likelihoods,
    _sampling_ratio_profiles,
    build_parameter_profiles,
)
from src.results.tls.config import PipelineConfig, load_config
from src.results.tls.data import (
    DatasetContext,
    SubtreeLikelihoodInput,
    load_inference_parameters,
)
from src.results.tls.likelihood import evaluate_total_log_likelihood
from src.results.tls.plot import (
    _draw_profile_box,
    _parameter_bounds,
    _render_sensitivity,
    plot_profile_data,
)
from src.tree_utils import (
    cut_tree,
    get_datastructure_for_likelihood_calculation,
    prepare_nx_tree,
)


def _read_log_likelihood(path: Path) -> float:
    matches = [
        line.partition("=")[2]
        for line in path.read_text().splitlines()
        if line.startswith("LL=")
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one log likelihood in {path}; found {len(matches)}")
    try:
        value = float(matches[0])
    except ValueError as error:
        raise ValueError(f"Invalid log likelihood in {path}") from error
    if not math.isfinite(value):
        raise ValueError(f"Log likelihood in {path} must be finite")
    return value


def _cohort_name(inference_path: Path) -> str:
    for directory in inference_path.parents:
        match = re.match(r"^(S\d+)(?:_|$)", directory.name)
        if match is not None:
            return match.group(1)
    raise ValueError(
        "Expected an inference parent directory to begin with a cohort "
        f"label such as S10 or S15: {inference_path.parent}"
    )


def _config_with_likelihood_drops(
    config: PipelineConfig,
    likelihood_drops: tuple[float, ...],
) -> PipelineConfig:
    drops = tuple(sorted(float(drop) for drop in likelihood_drops))
    if (
        len(drops) < 2
        or len(set(drops)) != len(drops)
        or any(not math.isfinite(drop) or drop <= 0.0 for drop in drops)
    ):
        raise ValueError(
            "At least two unique, finite, positive likelihood drops are required."
        )
    fractions = tuple(math.exp(-drop) for drop in drops)
    if any(not 0.0 < fraction < 1.0 for fraction in fractions):
        raise ValueError("Likelihood drops must convert to fractions in (0, 1).")

    plotting = dict(config.plotting)
    plotting["boxplot_likelihood_fraction"] = fractions[-2]
    plotting["whisker_likelihood_fraction"] = fractions[-1]
    return replace(
        config,
        likelihood_fractions=fractions,
        plotting=plotting,
    )


def _read_tree_names(input_dir: Path, expected_count: int) -> tuple[str, ...]:
    path = input_dir / "trees.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Combined tree-name file does not exist: {path}")
    tree_names = tuple(
        line.strip() for line in path.read_text().splitlines() if line.strip()
    )
    if len(tree_names) != expected_count:
        raise ValueError(
            f"Expected {expected_count} tree names in {path}; found {len(tree_names)}."
        )
    return tree_names


def load_combined_dataset_context(
    input_dir: Path,
    inference_path: Path,
    state_names: tuple[str, ...],
    *,
    use_per_edge_transition_data: bool = True,
) -> DatasetContext:
    parameters = load_inference_parameters(inference_path, state_names)
    tree_file_pairs = find_tree_file_pairs(input_dir)
    if not tree_file_pairs:
        raise ValueError(
            f"Expected at least one tree pair in {input_dir}; found none."
        )
    if (
        parameters.tree_transition_probabilities is not None
        and parameters.tree_transition_probabilities.shape[0] != len(tree_file_pairs)
    ):
        raise ValueError(
            "Per-tree transition matrix count does not match the combined tree count"
        )
    if (
        parameters.tree_transition_times is not None
        and len(parameters.tree_transition_times) != len(tree_file_pairs)
    ):
        raise ValueError(
            "Per-tree transition-time count does not match the combined tree count"
        )

    sensitivity_inputs = []
    for tree_position, (node_path, edge_path) in enumerate(tree_file_pairs):
        match = re.search(r"nodes_(\d+)", node_path.stem, flags=re.IGNORECASE)
        tree_index = int(match.group(1)) if match else tree_position
        transition_path = input_dir / f"transition_times_{tree_index}.csv"
        with redirect_stdout(StringIO()):
            transition_data = (
                load_per_edge_transition_data(transition_path)
                if use_per_edge_transition_data
                else None
            )
            nodes = load_nodes(str(node_path))
            edges = load_edges(str(edge_path))
            tree = prepare_nx_tree(nodes, edges)

        root = tree.graph["root"]
        if tree.out_degree(root) != 2:
            raise ValueError(
                f"{node_path} must have a stem-free binary crown root."
            )
        invalid_leaf_states = [
            node.state
            for node in nodes
            if node.is_leaf and not 0 <= node.state < len(state_names)
        ]
        if invalid_leaf_states:
            raise ValueError(
                f"Invalid leaf states in {node_path}: "
                f"{sorted(set(invalid_leaf_states))}"
            )

        root_time = float(tree.nodes[root]["T"])
        candidate_times = (
            parameters.tree_transition_times[tree_position]
            if parameters.tree_transition_times is not None
            else parameters.transition_times
        )
        local_transition_times = [
            time for time in candidate_times if time < root_time
        ]
        levels, hierarchy = cut_tree(tree, local_transition_times)
        with redirect_stdout(StringIO()):
            datastructure = get_datastructure_for_likelihood_calculation(
                levels,
                hierarchy,
                local_transition_times,
                len(state_names),
                per_edge_transition_data=transition_data,
            )
        sensitivity_inputs.append(
            SubtreeLikelihoodInput(
                root_time,
                local_transition_times,
                datastructure,
            )
        )

    return DatasetContext(
        birth_rates=parameters.birth_rates,
        death_rates=parameters.death_rates,
        sampling_probabilities=(
            (
                parameters.tree_sampling_probabilities[:, None]
                .expand(-1, len(state_names))
                .clone()
                if parameters.tree_sampling_probabilities.ndim == 1
                else parameters.tree_sampling_probabilities
            )
            if parameters.tree_sampling_probabilities is not None
            else parameters.sampling_probabilities
        ),
        transition_probabilities=(
            parameters.tree_transition_probabilities
            if parameters.tree_transition_probabilities is not None
            else parameters.transition_probabilities
        ),
        transition_times=parameters.transition_times,
        sensitivity_inputs=sensitivity_inputs,
        tree_sampling_scope=parameters.tree_sampling_scope,
        tree_transition_times=parameters.tree_transition_times,
    )


def _inference_sampling_profiles(
    context: DatasetContext,
    config: PipelineConfig,
    *,
    fixed_sampling: bool,
) -> list[ParameterProfile]:
    profiles = build_parameter_profiles(context, config)
    if not fixed_sampling:
        return profiles

    sampling = context.sampling_probabilities.detach().cpu().numpy()
    if not np.allclose(sampling, 0.1, rtol=0.0, atol=1e-12):
        raise ValueError(
            "Fixed-sampling profiles require the inference sampling probability "
            "to be 0.1."
        )
    sampling_profiles = [
        profile for profile in profiles if profile.parameter_type == "sampling"
    ]
    if len(sampling_profiles) != 1 or sampling_profiles[0].name != "sampling[shared]":
        raise ValueError("Expected one shared fixed-sampling parameter.")
    fixed_profile = sampling_profiles[0]
    return [
        ParameterProfile(
            fixed_profile.name,
            fixed_profile.parameter_type,
            fixed_profile.index,
            fixed_profile.original_value,
            np.asarray([fixed_profile.original_value], dtype=float),
        )
        if profile.parameter_type == "sampling"
        else profile
        for profile in profiles
    ]


def _plot_config(
    config: PipelineConfig,
    output_dir: Path,
    dataset_name: str,
    cohort_name: str,
    sampling_mode: str,
    tree_names: tuple[str, ...] | None = None,
    include_rate_ratio_profiles: bool = True,
    transition_times_only: bool = False,
) -> PipelineConfig:
    plotting = dict(config.plotting)
    figure_identity = cohort_name
    dataset_label = f"Combined {cohort_name}"
    if transition_times_only and tree_names is not None and len(tree_names) == 1:
        figure_identity = f"{cohort_name}_{tree_names[0]}"
        dataset_label = f"{tree_names[0]} ({cohort_name})"
    suffix = {
        "fixed": "fixed_sampling_0.1",
        "inferred": "inferred_sampling",
        "per_tree": "per_tree_sampling",
        "per_tree_state": "per_tree_state_sampling",
        "per_tree_transition": "per_tree_transition_rates",
    }[sampling_mode]
    plotting.update(
        {
            "output_basename": (
                f"SGOOTR_combined_{figure_identity}_{suffix}_composite_sensitivity"
            ),
            "parameter_output_basenames": {
                "birth": (
                    f"SGOOTR_combined_{figure_identity}_{suffix}_birth_rates"
                ),
                "death": (
                    f"SGOOTR_combined_{figure_identity}_{suffix}_death_rates"
                ),
                "transition": (
                    f"plot_2_SGOOTR_combined_{figure_identity}_{suffix}_"
                    "transition_probabilities"
                ),
                "sampling": (
                    f"SGOOTR_combined_{figure_identity}_{suffix}_sampling_rates"
                ),
            },
            "sampling_ratio_output_basename": (
                f"plot_3_SGOOTR_combined_{figure_identity}_{suffix}_"
                "sampling_rate_ratio_profiles"
            ),
            "birth_ratio_output_basename": (
                f"SGOOTR_combined_{figure_identity}_{suffix}_birth_rate_ratio_profiles"
            ),
            "ratio_interval_output_basename": (
                f"SGOOTR_combined_{figure_identity}_{suffix}_rate_ratio_profile_intervals"
            ),
            "dataset_colors": {dataset_name: "#6f91bd"},
            "dataset_labels": {dataset_name: dataset_label},
            "composite_panel_types": [
                "growth",
                "transition",
                "scaled_sampling",
                "sampling_ratio",
                "birth_ratio",
            ],
            "composite_panel_rows": [
                ["growth", "transition", "scaled_sampling"],
                ["sampling_ratio", "birth_ratio"],
            ],
            "composite_panel_width_ratios": [1.0, 0.55, 1.0, 0.85, 0.85],
            "composite_panel_gaps": [0.18, 0.4, 0.16],
            "composite_row_hspace": 0.35,
            "composite_bottom": 0.13,
            "figsize": [18.5, 8.4],
            "parameter_figsize": [5.8, 4.2],
            "box_width": 0.3,
            "transition_plot_style": "heatmap",
        }
    )
    if sampling_mode in {"per_tree_state", "per_tree_transition"}:
        plotting["parameter_figsize"] = [9.9, 4.2]
        plotting["formats"] = ["pdf", "svg", "png"]
    if transition_times_only:
        plotting["formats"] = ["pdf", "svg", "png"]
    if sampling_mode == "per_tree_transition":
        plotting["transition_parameter_figsize"] = [15.5, 4.8]
    if tree_names is not None:
        plotting["tree_names"] = list(tree_names)
    if not include_rate_ratio_profiles:
        plotting.update(
            {
                "composite_panel_types": [
                    "growth",
                    "transition",
                    "scaled_sampling",
                ],
                "composite_panel_rows": [
                    ["growth", "transition", "scaled_sampling"],
                ],
                "composite_panel_width_ratios": [1.0, 0.55, 1.0],
                "composite_panel_gaps": [0.18, 0.4],
                "composite_bottom": 0.18,
                "figsize": [11.5, 4.2],
            }
        )
    return replace(
        config,
        data_root=output_dir / "data",
        preprocessing_root=output_dir / "data",
        figure_root=output_dir / "figures",
        datasets=(dataset_name,),
        plotting=plotting,
    )


def _plot_ratio_profiles(
    config: PipelineConfig,
    profile_rows: list[dict[str, object]],
    parameter_type: str,
    parameter_symbol: str,
) -> list[Path]:
    ratio_rows = [
        row for row in profile_rows if row["parameter_type"] == parameter_type
    ]
    if not ratio_rows:
        return []

    tree_indices = sorted(
        {
            int(match.group(1))
            for row in ratio_rows
            if (
                match := re.fullmatch(
                    r"sampling_ratio\[tree_(\d+),[^,\]]+/NC\]",
                    str(row["parameter"]),
                )
            )
        }
    )
    tree_names = config.plotting.get("tree_names", [])
    if parameter_type == "sampling_ratio" and tree_indices:
        columns = 3
        rows = int(math.ceil(len(tree_indices) / columns))
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(15.0, 4.2 * rows),
            sharey=True,
            squeeze=False,
        )
        for tree_index, axis in enumerate(axes.flat):
            if tree_index >= len(tree_indices):
                axis.set_visible(False)
                continue
            x_limits = []
            likelihood_drops = []
            for state_name in ("PT", "LN"):
                parameter_name = (
                    f"sampling_ratio[tree_{tree_index},{state_name}/NC]"
                )
                rows_for_parameter = sorted(
                    (
                        row
                        for row in ratio_rows
                        if row["parameter"] == parameter_name
                    ),
                    key=lambda row: float(row["parameter_value"]),
                )
                if not rows_for_parameter:
                    raise ValueError(
                        f"Missing profile values for {parameter_name}"
                    )
                values = np.asarray(
                    [float(row["parameter_value"]) for row in rows_for_parameter]
                )
                likelihoods = np.asarray(
                    [
                        float(row["summed_log_likelihood"])
                        for row in rows_for_parameter
                    ]
                )
                delta_likelihood = likelihoods - likelihoods.max()
                original_value = float(rows_for_parameter[0]["original_value"])
                color = config.plotting["state_colors"][state_name]
                axis.plot(
                    values,
                    delta_likelihood,
                    color=color,
                    linewidth=2.0,
                    label=f"{state_name}/NC (fit {original_value:.3g})",
                )
                axis.axvline(
                    original_value,
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                )
                axis.axvline(
                    1.0,
                    color="0.4",
                    linestyle="--",
                    linewidth=1.0,
                )
                x_limits.append(
                    (
                        max(float(values[0]), original_value / 50.0),
                        min(float(values[-1]), original_value * 50.0),
                    )
                )
                likelihood_drops.append(delta_likelihood)
            tree_name = (
                tree_names[tree_index]
                if tree_index < len(tree_names)
                else f"Tree {tree_index}"
            )
            axis.axhline(0.0, color="#555555", linewidth=0.8)
            axis.set_xscale("log")
            axis.set_xlim(
                min(bounds[0] for bounds in x_limits),
                max(bounds[1] for bounds in x_limits),
            )
            axis.set_ylim(
                bottom=max(
                    min(float(drops.min()) for drops in likelihood_drops),
                    -12.0,
                ),
                top=0.5,
            )
            axis.set_xlabel(r"$\rho_s/\rho_{NC}$")
            axis.set_title(tree_name, fontsize=13, fontweight="normal")
            axis.grid(axis="y", color="0.87", linewidth=0.7, linestyle=":")
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.legend(frameon=False, fontsize=9, loc="best")
            axis.tick_params(axis="both", labelsize=10)
        for axis in axes[:, 0]:
            axis.set_ylabel("Log likelihood relative to profile maximum")
    else:
        tree_indices = [None]
        figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
        axes = np.asarray(axes).reshape(1, 2)

    if not (
        parameter_type == "sampling_ratio"
        and tree_indices
        and tree_indices[0] is not None
    ):
        for row_index, tree_index in enumerate(tree_indices):
            for column_index, state_name in enumerate(("PT", "LN")):
                axis = axes[row_index, column_index]
                parameter_name = (
                    f"{parameter_type}[{state_name}/NC]"
                    if tree_index is None
                    else f"sampling_ratio[tree_{tree_index},{state_name}/NC]"
                )
                rows = sorted(
                    (
                        row
                        for row in ratio_rows
                        if row["parameter"] == parameter_name
                    ),
                    key=lambda row: float(row["parameter_value"]),
                )
                if not rows:
                    raise ValueError(f"Missing profile values for {parameter_name}")
                values = np.asarray([float(row["parameter_value"]) for row in rows])
                likelihoods = np.asarray(
                    [float(row["summed_log_likelihood"]) for row in rows]
                )
                maximum_index = int(np.argmax(likelihoods))
                delta_likelihood = likelihoods - likelihoods[maximum_index]
                original_value = float(rows[0]["original_value"])
                axis.plot(
                    values,
                    delta_likelihood,
                    color=config.plotting["state_colors"][state_name],
                    linewidth=2.0,
                )
                axis.axvline(
                    original_value,
                    color="#333333",
                    linestyle="--",
                    linewidth=1.0,
                    label=f"fitted ratio = {original_value:.3g}",
                )
                axis.axvline(
                    1.0,
                    color="0.4",
                    linestyle="--",
                    linewidth=1.0,
                )
                axis.axhline(0.0, color="#555555", linewidth=0.8)
                axis.set_xscale("log")
                axis.set_xlim(
                    max(float(values[0]), original_value / 50.0),
                    min(float(values[-1]), original_value * 50.0),
                )
                axis.set_ylim(
                    bottom=max(float(delta_likelihood.min()), -12.0),
                    top=0.5,
                )
                axis.set_xlabel(
                    rf"${parameter_symbol}_{{{state_name}}}/{parameter_symbol}_{{NC}}$"
                )
                panel_title = f"{state_name} / NC"
                if tree_index is not None:
                    tree_name = (
                        tree_names[tree_index]
                        if tree_index < len(tree_names)
                        else f"Tree {tree_index}"
                    )
                    panel_title = f"{tree_name}: {panel_title}"
                axis.set_title(panel_title, fontsize=13, fontweight="normal")
                axis.grid(axis="y", color="0.87", linewidth=0.7, linestyle=":")
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.legend(frameon=False, fontsize=9, loc="best")
                axis.tick_params(axis="both", labelsize=10)
            axes[row_index, 0].set_ylabel(
                "Log likelihood relative to profile maximum"
            )
    title = {
        "sampling_ratio": "Relative sampling-probability profiles",
        "birth_ratio": "Relative birth-rate profiles",
    }[parameter_type]
    figure.suptitle(title, fontsize=14, y=0.99)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    outputs = []
    output_basename = config.plotting[f"{parameter_type}_output_basename"]
    for extension in config.plotting["formats"]:
        output = config.figure_root / f"{output_basename}.{extension}"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight", pad_inches=0.01, dpi=200)
        outputs.append(output)
    plt.close(figure)
    return outputs


def _plot_ratio_profile_intervals(
    config: PipelineConfig,
    ranges_by_dataset: dict[str, pd.DataFrame],
    metadata_by_dataset: dict[str, pd.DataFrame],
) -> list[Path]:
    ratio_types = ("sampling_ratio", "birth_ratio")
    metadata = metadata_by_dataset[config.datasets[0]]
    if not metadata["parameter_type"].isin(ratio_types).any():
        return []

    titles_and_labels = {
        "sampling_ratio": (
            "Relative sampling probabilities",
            r"$\rho_s/\rho_{NC}$",
        ),
        "birth_ratio": ("Relative birth rates", r"$\beta_s/\beta_{NC}$"),
    }
    state_names = ("PT", "LN")
    figure, axes = plt.subplots(1, 2, figsize=(15.0, 4.2))
    box_fraction = float(config.plotting["boxplot_likelihood_fraction"])
    whisker_fraction = float(config.plotting["whisker_likelihood_fraction"])
    font_size = int(config.plotting["font_size"])

    for axis, parameter_type in zip(axes, ratio_types):
        title, y_label = titles_and_labels[parameter_type]
        parameters = metadata[metadata["parameter_type"] == parameter_type]
        tree_matches = {
            row.parameter: re.fullmatch(
                r"sampling_ratio\[tree_(\d+),([^,\]]+)/NC\]",
                row.parameter,
            )
            for row in parameters.itertuples(index=False)
        }
        tree_specific_sampling = (
            parameter_type == "sampling_ratio"
            and any(match is not None for match in tree_matches.values())
        )
        if len(parameters) != len(state_names) and not tree_specific_sampling:
            raise ValueError(
                f"Expected {len(state_names)} {parameter_type} profiles; "
                f"found {len(parameters)}."
            )
        ordered_parameters = (
            parameters.sort_values("parameter_index")
            if tree_specific_sampling
            else parameters.set_index("parameter").reindex(
                [f"{parameter_type}[{state_name}/NC]" for state_name in state_names]
            ).reset_index()
        )
        tick_labels = []
        for position, row in enumerate(ordered_parameters.itertuples(index=False)):
            parameter_name = row.parameter
            matching = parameters[parameters["parameter"] == parameter_name]
            if len(matching) != 1:
                raise ValueError(
                    f"Expected one profile for {parameter_name}; found "
                    f"{len(matching)}."
                )
            original_value = float(matching["original_value"].iloc[0])
            tree_match = tree_matches.get(parameter_name)
            if tree_match is None:
                state_name = parameter_name.removeprefix(
                    f"{parameter_type}["
                ).removesuffix("/NC]")
                tick_labels.append(f"{state_name} / NC")
            else:
                tree_index = int(tree_match.group(1))
                state_name = tree_match.group(2)
                tree_names = config.plotting.get("tree_names", [])
                tree_name = (
                    tree_names[tree_index]
                    if tree_index < len(tree_names)
                    else f"Tree {tree_index}"
                )
                tick_labels.append(f"{tree_name}\n{state_name}/NC")
            dataset = config.datasets[0]
            box_bounds = _parameter_bounds(
                ranges_by_dataset[dataset],
                parameter_name,
                box_fraction,
                dataset,
            )
            whisker_bounds = _parameter_bounds(
                ranges_by_dataset[dataset],
                parameter_name,
                whisker_fraction,
                dataset,
            )
            _draw_profile_box(
                axis,
                position,
                original_value,
                box_bounds,
                whisker_bounds,
                config.plotting["state_colors"][state_name],
                float(config.plotting["box_width"]),
            )

        axis.set_xlim(-0.5, len(ordered_parameters) - 0.5)
        axis.set_xticks(
            range(len(ordered_parameters)),
            labels=tick_labels,
        )
        axis.set_title(title, fontsize=font_size, fontweight="normal", pad=5)
        axis.set_ylabel(y_label, fontsize=font_size, labelpad=2)
        axis.axhline(
            1.0,
            color="0.4",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )
        axis.tick_params(axis="both", labelsize=font_size, length=3, pad=2)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="0.87", linewidth=0.7, linestyle=":")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.1)
        axis.spines["bottom"].set_linewidth(1.1)

    figure.subplots_adjust(left=0.09, right=0.99, top=0.94, bottom=0.16, wspace=0.3)
    config.figure_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    basename = config.plotting["ratio_interval_output_basename"]
    for extension in config.plotting["formats"]:
        output = config.figure_root / f"{basename}.{extension}"
        figure.savefig(output, bbox_inches="tight", pad_inches=0.01, dpi=200)
        outputs.append(output)
    plt.close(figure)
    return outputs


def _plot_sampling_ratio_profile_intervals_by_tree(
    config: PipelineConfig,
    ranges_by_dataset: dict[str, pd.DataFrame],
    metadata_by_dataset: dict[str, pd.DataFrame],
) -> list[Path]:
    dataset = config.datasets[0]
    metadata = metadata_by_dataset[dataset]
    parameters = metadata[metadata["parameter_type"] == "sampling_ratio"]
    tree_matches = {
        row.parameter: re.fullmatch(
            r"sampling_ratio\[tree_(\d+),([^,\]]+)/NC\]",
            row.parameter,
        )
        for row in parameters.itertuples(index=False)
    }
    tree_indices = sorted(
        {
            int(match.group(1))
            for match in tree_matches.values()
            if match is not None
        }
    )
    tree_specific = bool(tree_indices)
    if not tree_indices:
        if (
            len(config.plotting.get("tree_names", [])) != 1
            or not all(
                f"sampling_ratio[{state_name}/NC]" in set(parameters["parameter"])
                for state_name in ("PT", "LN")
            )
        ):
            return []
        tree_indices = [0]

    tree_names = config.plotting.get("tree_names", [])
    columns = 3
    rows = int(math.ceil(len(tree_indices) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(13.5, 4.6 * rows),
        sharey=True,
        squeeze=False,
    )
    box_fraction = float(config.plotting["boxplot_likelihood_fraction"])
    whisker_fraction = float(config.plotting["whisker_likelihood_fraction"])
    box_width = float(config.plotting["box_width"])
    for tree_index, axis in enumerate(axes.flat):
        if tree_index >= len(tree_indices):
            axis.set_visible(False)
            continue
        for position, state_name in enumerate(("PT", "LN")):
            parameter_name = (
                f"sampling_ratio[tree_{tree_index},{state_name}/NC]"
                if tree_specific
                else f"sampling_ratio[{state_name}/NC]"
            )
            matching = parameters[parameters["parameter"] == parameter_name]
            if len(matching) != 1:
                raise ValueError(
                    f"Expected one profile for {parameter_name}; found "
                    f"{len(matching)}."
                )
            original_value = float(matching["original_value"].iloc[0])
            box_bounds = _parameter_bounds(
                ranges_by_dataset[dataset],
                parameter_name,
                box_fraction,
                dataset,
            )
            whisker_bounds = _parameter_bounds(
                ranges_by_dataset[dataset],
                parameter_name,
                whisker_fraction,
                dataset,
            )
            _draw_profile_box(
                axis,
                position,
                original_value,
                box_bounds,
                whisker_bounds,
                config.plotting["state_colors"][state_name],
                box_width,
            )
        tree_name = (
            tree_names[tree_index]
            if tree_index < len(tree_names)
            else f"Tree {tree_index}"
        )
        axis.axhline(
            1.0,
            color="0.4",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )
        axis.set_title(tree_name, fontsize=12, fontweight="normal")
        axis.set_xticks((0, 1), labels=("PT/NC", "LN/NC"))
        axis.set_xlim(-0.5, 1.5)
        axis.grid(axis="y", color="0.87", linewidth=0.7, linestyle=":")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="both", labelsize=10)
    for axis in axes[:, 0]:
        axis.set_ylabel(r"Relative sampling probability $\rho_s/\rho_{NC}$")
    figure.suptitle("Relative sampling probabilities by tree", fontsize=14, y=0.99)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    output_stem = (
        config.figure_root.parent.parent
        / "figures"
        / "plot_3_relative_sampling_probability_profile_intervals_by_tree"
    )
    outputs = []
    for extension in config.plotting["formats"]:
        output = output_stem.with_suffix(f".{extension}")
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight", pad_inches=0.01, dpi=200)
        outputs.append(output)
    plt.close(figure)
    return outputs


def profile_combined_inference(
    inference_path: Path,
    input_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    *,
    transition_times_only: bool = False,
) -> list[Path]:
    cohort_name = _cohort_name(inference_path)
    parameters = load_inference_parameters(inference_path, config.state_names)
    if parameters.tree_transition_probabilities is not None:
        sampling_mode = "per_tree_transition"
        fixed_sampling = False
    elif parameters.tree_sampling_scope == "per_tree_by_state":
        sampling_mode = "per_tree_state"
        fixed_sampling = False
    elif parameters.tree_sampling_probabilities is not None:
        sampling_mode = "per_tree"
        fixed_sampling = False
    elif "fixed_sampling_0.1" in inference_path.stem:
        sampling_mode = "fixed"
        fixed_sampling = True
    else:
        sampling_mode = "inferred"
        fixed_sampling = False
    dataset_name = {
        "fixed": (
            f"combined_{cohort_name}_upgma_min_subtracted_fixed_sampling_0.1"
        ),
        "inferred": (
            f"combined_{cohort_name}_upgma_min_subtracted_inferred_sampling"
        ),
        "per_tree": (
            f"combined_{cohort_name}_upgma_min_subtracted_per_tree_sampling"
        ),
        "per_tree_state": (
            f"combined_{cohort_name}_upgma_min_subtracted_per_tree_state_sampling"
        ),
        "per_tree_transition": (
            f"combined_{cohort_name}_upgma_min_subtracted_per_tree_transition_rates"
        ),
    }[sampling_mode]
    context = load_combined_dataset_context(
        input_dir,
        inference_path,
        config.state_names,
        use_per_edge_transition_data=not transition_times_only,
    )
    tree_names = (
        _read_tree_names(input_dir, len(context.sensitivity_inputs))
        if transition_times_only
        else None
    )
    if tree_names is not None and len(tree_names) == 1:
        dataset_name = (
            f"combined_{cohort_name}_{tree_names[0]}_"
            f"{sampling_mode}_transition_times_only"
        )
    baseline = evaluate_total_log_likelihood(
        context,
        context.birth_rates,
        context.death_rates,
        context.sampling_probabilities,
        context.transition_probabilities,
        no_stem=not transition_times_only,
    )
    fitted_log_likelihood = _read_log_likelihood(inference_path)
    if not np.isclose(baseline, fitted_log_likelihood, rtol=1e-8, atol=1e-8):
        raise ValueError(
            "Profile baseline does not match the combined inference result: "
            f"{baseline:.15g} != {fitted_log_likelihood:.15g}."
        )

    profiles = _inference_sampling_profiles(
        context,
        config,
        fixed_sampling=fixed_sampling,
    )
    profiles.extend(_sampling_ratio_profiles(context, config))
    profiles.extend(_birth_ratio_profiles(context, config))

    profile_rows_by_parameter = []
    range_rows = []
    metadata_rows = []
    for profile in profiles:
        likelihoods = _profile_likelihoods(
            context,
            profile,
            config,
            no_stem=not transition_times_only,
        )
        profile_data, ranges = profile_rows(
            dataset_name,
            profile,
            likelihoods,
            baseline,
            config.likelihood_fractions,
        )
        profile_rows_by_parameter.extend(profile_data)
        range_rows.extend(ranges)
        metadata_rows.append(
            {
                "dataset": dataset_name,
                "parameter": profile.name,
                "parameter_type": profile.parameter_type,
                "parameter_index": profile.index,
                "original_value": profile.original_value,
            }
        )

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ranges_by_dataset = {dataset_name: pd.DataFrame(range_rows)}
    metadata_by_dataset = {dataset_name: pd.DataFrame(metadata_rows)}
    pd.DataFrame(profile_rows_by_parameter).to_csv(
        data_dir / config.profile_filename,
        index=False,
    )
    ranges_by_dataset[dataset_name].to_csv(
        data_dir / config.ranges_filename,
        index=False,
    )
    metadata_by_dataset[dataset_name].to_csv(
        data_dir / config.metadata_filename,
        index=False,
    )
    plot_config = _plot_config(
        config,
        output_dir,
        dataset_name,
        cohort_name,
        sampling_mode,
        tree_names
        or (
            _read_tree_names(input_dir, len(context.sensitivity_inputs))
            if sampling_mode
            in {"per_tree", "per_tree_state", "per_tree_transition"}
            else None
        ),
        include_rate_ratio_profiles=True,
        transition_times_only=transition_times_only,
    )
    outputs = plot_profile_data(
        plot_config,
        ranges_by_dataset,
        metadata_by_dataset,
    )
    rate_profile_plotting = dict(plot_config.plotting)
    rate_profile_plotting.update(
        {
            "output_basename": (
                "plot_1_net_growth_scaled_birth_relative_birth_profiles"
            ),
            "composite_panel_types": [
                "growth",
                "scaled_sampling",
                "birth_ratio",
            ],
            "composite_panel_rows": [
                ["growth", "scaled_sampling", "birth_ratio"],
            ],
            "composite_panel_width_ratios": [1.0, 1.0, 1.0],
            "composite_panel_gaps": [0.22, 0.22],
            "composite_bottom": 0.18,
            "figsize": [16.0, 4.8],
        }
    )
    rate_profile_config = replace(
        plot_config,
        figure_root=output_dir.parent / "figures",
        plotting=rate_profile_plotting,
    )
    outputs.extend(
        _render_sensitivity(
            rate_profile_config,
            ranges_by_dataset,
            metadata_by_dataset,
        )
    )
    outputs.extend(
        _plot_ratio_profiles(
            plot_config,
            profile_rows_by_parameter,
            "sampling_ratio",
            r"\rho",
        )
    )
    outputs.extend(
        _plot_ratio_profiles(
            plot_config,
            profile_rows_by_parameter,
            "birth_ratio",
            r"\beta",
        )
    )
    outputs.extend(
        _plot_ratio_profile_intervals(
            plot_config,
            ranges_by_dataset,
            metadata_by_dataset,
        )
    )
    outputs.extend(
        _plot_sampling_ratio_profile_intervals_by_tree(
            plot_config,
            ranges_by_dataset,
            metadata_by_dataset,
        )
    )
    print(
        f"Combined {sampling_mode}-sampling profile baseline: {baseline:.15g}."
    )
    print(f"Saved profile tables under {data_dir}.")
    for output in outputs:
        print(f"Wrote {output}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile parameters for a combined SGOOTR inference."
    )
    parser.add_argument(
        "inference",
        type=Path,
        help="Combined inference result file.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing nodes_#, edges_#, and transition_times_# files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Profile output directory; defaults to profiles/ beside the inference.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config.yaml"),
    )
    parser.add_argument(
        "--transition-times-only",
        action="store_true",
        help=(
            "Profile the main2.py likelihood using common transition epochs "
            "without per-edge transition assignments or crown conditioning."
        ),
    )
    parser.add_argument(
        "--likelihood-drops",
        type=float,
        nargs="+",
        help=(
            "Override profile interval thresholds with positive "
            "delta-log-likelihood drops. This affects only the current run."
        ),
    )
    arguments = parser.parse_args()
    input_dir = arguments.input_dir or arguments.inference.parent
    output_dir = arguments.output_dir or arguments.inference.parent / "profiles"
    config = load_config(arguments.config, section="sgootr_profiles")
    if arguments.likelihood_drops is not None:
        config = _config_with_likelihood_drops(
            config,
            tuple(arguments.likelihood_drops),
        )
        print(
            "Using profile likelihood drops Δlog L="
            f"{tuple(sorted(arguments.likelihood_drops))}."
        )
    profile_combined_inference(
        arguments.inference,
        input_dir,
        output_dir,
        config,
        transition_times_only=arguments.transition_times_only,
    )


if __name__ == "__main__":
    main()
