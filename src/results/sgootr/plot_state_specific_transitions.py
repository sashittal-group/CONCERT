from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import math
from pathlib import Path
import re
import warnings
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.closed_form_state_specific_transitions import (
    _safe_logsumexp,
    calculate_Us,
    calculate_log_likelihood,
)
from src.dataprocessor import load_edges, load_nodes
from src.main2 import find_tree_file_pairs
from src.mle_state_specific_transitions import EPOCH_EDGES, STATE_NAMES
from src.plotters import draw_ultrametric_tree
from src.results.profile_tools import profile_rows
from src.results.sgootr.preprocess import (
    _birth_ratio_profiles,
    _growth_profiles,
    _sampling_ratio_profiles,
    _scaled_sampling_profiles,
)
from src.results.sgootr.profile_combined_inference import (
    _config_with_likelihood_drops,
    _plot_config,
    _plot_sampling_ratio_profile_intervals_by_tree,
    _read_log_likelihood,
)
from src.results.tls.config import load_config
from src.results.tls.plot import _render_sensitivity
from src.results.tls.process_state_coloring import (
    SubtreeInput,
    _component_rows,
    _local_component_likelihoods,
)
from src.results.tls import plot_state_coloring as state_coloring_plotter
from src.tree_utils import (
    cut_tree,
    get_datastructure_for_likelihood_calculation,
    prepare_nx_tree,
)


_EVENT_BY_LABEL = {
    f"{STATE_NAMES[source]}->{STATE_NAMES[target]}": (source, target)
    for source, target in EPOCH_EDGES
}
_EPOCH_PATTERN = re.compile(r"epoch_(\d+)\s+([A-Z]+->[A-Z]+)")
_TRANSITION_PATTERN = re.compile(r"([A-Z]+)->([A-Z]+)\s+([0-9.eE+-]+)")


def _cohort_from_inference_path(path: Path) -> str:
    for parent in path.parents:
        if parent.name.startswith(("S10", "S15")):
            return parent.name[:3]
    raise ValueError(f"Could not determine SGOOTR cohort from {path}.")


def _read_result(path: Path) -> tuple[
    float,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[float],
    tuple[tuple[int, int], ...],
    torch.Tensor,
]:
    lines = path.read_text().splitlines()
    likelihood = _read_log_likelihood(path)
    vectors: dict[str, torch.Tensor] = {}
    for parameter in ("birth", "death", "sampling"):
        matching = [line for line in lines if line.startswith(f"{parameter}:")]
        if len(matching) != 1:
            raise ValueError(f"Expected one {parameter} vector in {path}.")
        vectors[parameter] = torch.tensor(
            [float(value) for value in matching[0].partition(":")[2].split()],
            dtype=torch.float64,
        )
    if any(values.numel() != len(STATE_NAMES) for values in vectors.values()):
        raise ValueError(f"Expected three state values for each parameter in {path}.")

    epoch_edges: dict[int, tuple[int, int]] = {}
    matrices_by_epoch: dict[int, torch.Tensor] = {}
    current_edge: tuple[int, int] | None = None
    current_epoch: int | None = None
    for line in lines:
        epoch_match = _EPOCH_PATTERN.fullmatch(line)
        if epoch_match:
            epoch_index = int(epoch_match.group(1))
            label = epoch_match.group(2)
            if label not in _EVENT_BY_LABEL:
                raise ValueError(f"Unexpected epoch transition {label!r} in {path}.")
            if epoch_index in epoch_edges:
                raise ValueError(f"Duplicate epoch {epoch_index} in {path}.")
            current_edge = _EVENT_BY_LABEL[label]
            current_epoch = epoch_index
            epoch_edges[current_epoch] = current_edge
            matrices_by_epoch[current_epoch] = torch.zeros(
                (len(STATE_NAMES), len(STATE_NAMES)),
                dtype=torch.float64,
            )
            continue
        transition_match = _TRANSITION_PATTERN.fullmatch(line)
        if transition_match and current_edge is not None and current_epoch is not None:
            source_name, target_name, probability = transition_match.groups()
            source = STATE_NAMES.index(source_name)
            target = STATE_NAMES.index(target_name)
            matrices_by_epoch[current_epoch][source, target] = float(probability)
        elif line == "transition_times_by_edge:":
            current_edge = None
            current_epoch = None

    transition_time_edges: list[tuple[int, int]] = []
    transition_times: list[float] = []
    in_times = False
    for line in lines:
        if line == "transition_times_by_edge:":
            in_times = True
            continue
        if not in_times or not line:
            continue
        match = _TRANSITION_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"Malformed edge-specific transition time in {path}: {line}")
        source_name, target_name, value = match.groups()
        label = f"{source_name}->{target_name}"
        if label not in _EVENT_BY_LABEL:
            raise ValueError(f"Unexpected transition time edge {label!r} in {path}.")
        transition_time_edges.append(_EVENT_BY_LABEL[label])
        transition_times.append(float(value))

    expected_epochs = set(range(len(epoch_edges)))
    if set(epoch_edges) != expected_epochs or set(matrices_by_epoch) != expected_epochs:
        raise ValueError(f"{path} must contain contiguous, indexed epoch matrices.")
    if transition_time_edges != [
        epoch_edges[index] for index in range(len(epoch_edges))
    ]:
        raise ValueError(f"{path} must list transition times in epoch order.")
    if len(transition_times) != len(epoch_edges):
        raise ValueError(f"{path} must contain one transition time per epoch.")
    for epoch_index, edge in epoch_edges.items():
        matrix = matrices_by_epoch[epoch_index]
        if not torch.allclose(
            matrix.sum(dim=1),
            torch.ones(len(STATE_NAMES), dtype=torch.float64),
            rtol=1e-8,
            atol=1e-8,
        ):
            raise ValueError(f"Transition matrix for {edge} does not sum to one.")
        permitted = {(state, state) for state in range(len(STATE_NAMES))}
        permitted.add(edge)
        if any(
            matrix[source, target] != 0.0
            for source in range(len(STATE_NAMES))
            for target in range(len(STATE_NAMES))
            if (source, target) not in permitted
        ):
            raise ValueError(
                f"Transition matrix for {edge} enables a disallowed state change."
            )

    ordered_events = sorted(
        (
            time_value,
            epoch_edges[epoch_index],
            matrices_by_epoch[epoch_index],
        )
        for epoch_index, time_value in enumerate(transition_times)
    )
    return (
        likelihood,
        vectors["birth"],
        vectors["death"],
        vectors["sampling"],
        [time_value for time_value, _, _ in ordered_events],
        tuple(edge for _, edge, _ in ordered_events),
        torch.stack([matrix for _, _, matrix in ordered_events]),
    )


def _specific_state_posteriors(
    datastructure: list[dict],
    local_likelihoods: list[torch.Tensor],
    transition_probabilities: torch.Tensor,
) -> tuple[list[torch.Tensor], float]:
    num_levels = len(datastructure)
    num_epochs = transition_probabilities.shape[0]
    log_probabilities = torch.log(
        transition_probabilities.clamp_min(torch.finfo(torch.float64).tiny)
    )

    upward: list[torch.Tensor | None] = [None] * num_levels
    child_messages: list[torch.Tensor | None] = [None] * (num_levels - 1)
    upward[-1] = local_likelihoods[-1]
    for level_index in range(num_levels - 2, -1, -1):
        child_upward = upward[level_index + 1]
        assert child_upward is not None
        epoch_index = num_epochs - 1 - level_index
        messages = torch.logsumexp(
            log_probabilities[epoch_index][None, :, :]
            + child_upward[:, None, :],
            dim=2,
        )
        child_messages[level_index] = messages
        adjacency = datastructure[level_index]["transition_matrix"].to(
            dtype=torch.float64
        )
        connected = adjacency > 0
        contributions = torch.where(
            connected[:, :, None],
            messages[None, :, :],
            torch.zeros_like(messages[None, :, :]),
        )
        upward[level_index] = local_likelihoods[level_index] + (
            contributions * adjacency[:, :, None]
        ).sum(dim=1)

    root_upward = upward[0]
    assert root_upward is not None
    if root_upward.shape[0] != 1:
        raise ValueError("The CRC11 tree must have exactly one root component.")
    root_log_likelihood = float(root_upward[0, 0])

    downward: list[torch.Tensor | None] = [None] * num_levels
    downward[0] = torch.full_like(root_upward, -torch.inf)
    downward[0][0, 0] = 0.0
    for level_index in range(num_levels - 1):
        parent_downward = downward[level_index]
        assert parent_downward is not None
        adjacency = datastructure[level_index]["transition_matrix"].to(
            dtype=torch.float64
        )
        messages = child_messages[level_index]
        assert messages is not None
        epoch_index = num_epochs - 1 - level_index
        child_downward = torch.full(
            (messages.shape[0], len(STATE_NAMES)),
            -torch.inf,
            dtype=torch.float64,
        )
        assigned_children: set[int] = set()
        for parent_index in range(adjacency.shape[0]):
            children = (
                torch.nonzero(adjacency[parent_index], as_tuple=False)
                .flatten()
                .tolist()
            )
            for child_index in children:
                outside = (
                    parent_downward[parent_index]
                    + local_likelihoods[level_index][parent_index]
                )
                for sibling_index in children:
                    if sibling_index != child_index:
                        outside = outside + messages[sibling_index]
                child_downward[child_index] = _safe_logsumexp(
                    outside[:, None] + log_probabilities[epoch_index],
                    dim=0,
                )
                if child_index in assigned_children:
                    raise ValueError("A component has more than one parent.")
                assigned_children.add(child_index)
        if len(assigned_children) != messages.shape[0]:
            raise ValueError("A component is disconnected from the root.")
        downward[level_index + 1] = child_downward

    posteriors = []
    for upward_level, downward_level in zip(upward, downward):
        assert upward_level is not None and downward_level is not None
        joint = upward_level + downward_level
        normalizer = torch.logsumexp(joint, dim=1, keepdim=True)
        if not torch.isfinite(normalizer).all():
            raise ValueError("Could not normalize an edge-segment state posterior.")
        posteriors.append(torch.exp(joint - normalizer))
    return posteriors, root_log_likelihood


def _evaluate_profile_likelihood(
    birth: torch.Tensor,
    death: torch.Tensor,
    sampling: torch.Tensor,
    matrices: torch.Tensor,
    times: list[float],
    datastructure: list[dict],
    root_time: float,
) -> float:
    u_result = calculate_Us(
        birth,
        death,
        sampling,
        matrices,
        torch.tensor(times, dtype=torch.float64),
        [root_time],
    )
    likelihood = calculate_log_likelihood(
        datastructure,
        birth,
        death,
        matrices,
        sampling,
        u_result.Cs,
        times,
    )[0, 0]
    return float(likelihood)


def _profile_candidate(
    profile_type: str,
    index: int,
    candidate: float,
    birth: torch.Tensor,
    death: torch.Tensor,
    sampling: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    birth = birth.clone()
    death = death.clone()
    sampling = sampling.clone()
    if profile_type == "growth":
        birth[index] = death[index] + candidate
    elif profile_type == "scaled_sampling":
        birth[index] = candidate / sampling[index]
    elif profile_type == "birth_ratio":
        pair_total = birth[0] + birth[index]
        birth[0] = pair_total / (1.0 + candidate)
        birth[index] = pair_total * candidate / (1.0 + candidate)
    elif profile_type == "sampling_ratio":
        pair_total = sampling[0] + sampling[index]
        sampling[0] = pair_total / (1.0 + candidate)
        sampling[index] = pair_total * candidate / (1.0 + candidate)
    else:
        raise ValueError(f"Unsupported state-specific profile type: {profile_type}")
    return birth, death, sampling


def _profile_outputs(
    inference_path: Path,
    output_dir: Path,
    cohort: str,
    birth: torch.Tensor,
    death: torch.Tensor,
    sampling: torch.Tensor,
    matrices: torch.Tensor,
    times: list[float],
    datastructure: list[dict],
    root_time: float,
    config,
) -> list[Path]:
    profile_context = SimpleNamespace(
        birth_rates=birth,
        death_rates=death,
        sampling_probabilities=sampling,
        transition_probabilities=matrices,
        tree_sampling_scope=None,
    )
    profiles = [
        *_growth_profiles(profile_context, config),
        *_scaled_sampling_profiles(profile_context, config),
        *_birth_ratio_profiles(profile_context, config),
        *_sampling_ratio_profiles(profile_context, config),
    ]
    baseline = _read_log_likelihood(inference_path)
    rows_by_parameter = []
    range_rows = []
    metadata_rows = []
    for profile in profiles:
        likelihoods = []
        for candidate in profile.values:
            candidate_birth, candidate_death, candidate_sampling = (
                _profile_candidate(
                    profile.parameter_type,
                    profile.index,
                    float(candidate),
                    birth,
                    death,
                    sampling,
                )
            )
            likelihoods.append(
                _evaluate_profile_likelihood(
                    candidate_birth,
                    candidate_death,
                    candidate_sampling,
                    matrices,
                    times,
                    datastructure,
                    root_time,
                )
            )
        profile_rows_data, ranges = profile_rows(
            f"combined_{cohort}_CRC11_state_specific_transition_times_only",
            profile,
            np.asarray(likelihoods),
            baseline,
            config.likelihood_fractions,
        )
        rows_by_parameter.extend(profile_rows_data)
        range_rows.extend(ranges)
        metadata_rows.append(
            {
                "dataset": f"combined_{cohort}_CRC11_state_specific_transition_times_only",
                "parameter": profile.name,
                "parameter_type": profile.parameter_type,
                "parameter_index": profile.index,
                "original_value": profile.original_value,
            }
        )

    dataset_name = f"combined_{cohort}_CRC11_state_specific_transition_times_only"
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    profile_frame = pd.DataFrame(rows_by_parameter)
    range_frame = pd.DataFrame(range_rows)
    metadata_frame = pd.DataFrame(metadata_rows)
    profile_frame.to_csv(data_dir / config.profile_filename, index=False)
    range_frame.to_csv(data_dir / config.ranges_filename, index=False)
    metadata_frame.to_csv(data_dir / config.metadata_filename, index=False)

    plot_config = _plot_config(
        config,
        output_dir,
        dataset_name,
        cohort,
        "inferred",
        ("CRC11",),
        include_rate_ratio_profiles=False,
        transition_times_only=True,
    )
    rate_plotting = dict(plot_config.plotting)
    rate_plotting.update(
        {
            "output_basename": "plot_1_net_growth_scaled_birth_relative_birth_profiles",
            "composite_panel_types": ["growth", "scaled_sampling", "birth_ratio"],
            "composite_panel_rows": [
                ["growth", "scaled_sampling", "birth_ratio"]
            ],
            "composite_panel_width_ratios": [1.0, 1.0, 1.0],
            "composite_panel_gaps": [0.22, 0.22],
            "composite_bottom": 0.18,
            "figsize": [16.0, 4.8],
        }
    )
    rate_plot_config = replace(
        plot_config,
        figure_root=output_dir.parent / "figures",
        plotting=rate_plotting,
    )
    ranges_by_dataset = {dataset_name: range_frame}
    metadata_by_dataset = {dataset_name: metadata_frame}
    outputs = _render_sensitivity(
        rate_plot_config,
        ranges_by_dataset,
        metadata_by_dataset,
    )
    outputs.extend(
        _plot_sampling_ratio_profile_intervals_by_tree(
            plot_config,
            ranges_by_dataset,
            metadata_by_dataset,
        )
    )
    return outputs


def plot_state_specific_inference(
    inference_path: Path,
    input_dir: Path,
    *,
    likelihood_drops: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0),
) -> list[Path]:
    (
        fitted_likelihood,
        birth,
        death,
        sampling,
        transition_times,
        epoch_edges,
        transition_matrices,
    ) = _read_result(inference_path)
    config = load_config(
        Path(__file__).with_name("config.yaml"),
        section="sgootr_profiles",
    )
    config = _config_with_likelihood_drops(config, likelihood_drops)

    tree_pairs = find_tree_file_pairs(input_dir, num_trees=1)
    node_path, edge_path = tree_pairs[0]
    with redirect_stdout(StringIO()):
        nodes = load_nodes(str(node_path))
        edges = load_edges(str(edge_path))
        tree = prepare_nx_tree(nodes, edges)
    root = tree.graph["root"]
    root_time = float(tree.nodes[root]["T"])
    if transition_times[-1] >= root_time:
        raise ValueError(
            f"Transition times in {inference_path} must precede root time "
            f"{root_time:.15g}."
        )
    levels, hierarchy = cut_tree(tree, transition_times)
    with redirect_stdout(StringIO()):
        datastructure = get_datastructure_for_likelihood_calculation(
            levels,
            hierarchy,
            transition_times,
            len(STATE_NAMES),
        )

    baseline = _evaluate_profile_likelihood(
        birth,
        death,
        sampling,
        transition_matrices,
        transition_times,
        datastructure,
        root_time,
    )
    if not math.isclose(baseline, fitted_likelihood, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError(
            "The state-specific plot baseline does not match the fitted result: "
            f"{baseline:.15g} != {fitted_likelihood:.15g}."
        )

    u_result = calculate_Us(
        birth,
        death,
        sampling,
        transition_matrices,
        torch.tensor(transition_times, dtype=torch.float64),
        [root_time],
    )
    local_likelihoods = _local_component_likelihoods(
        datastructure,
        birth,
        death,
        sampling,
        u_result.Cs,
        transition_times,
    )
    posteriors, root_log_likelihood = _specific_state_posteriors(
        datastructure,
        local_likelihoods,
        transition_matrices,
    )
    subtree = SubtreeInput(
        name="CRC11",
        root_time=root_time,
        levels=levels,
        datastructure=datastructure,
    )
    state_rows = _component_rows(
        "CRC11",
        subtree,
        posteriors,
        root_log_likelihood,
        STATE_NAMES,
    )
    state_segments = pd.DataFrame(state_rows)
    figures_dir = inference_path.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    state_segments.to_csv(figures_dir / "CRC11_state_segments.csv", index=False)
    state_coloring_plotter.CELL_STATE_COLORS.update(
        config.plotting["state_colors"]
    )
    state_plot_outputs = []
    for extension in ("pdf", "png"):
        output = figures_dir / f"CRC11_state_coloring.{extension}"
        state_coloring_plotter._plot_subtree(
            state_segments,
            output,
            f"{_cohort_from_inference_path(inference_path)} CRC11",
            "CRC11 (edge-segment posterior MAP states)",
            STATE_NAMES,
            180,
        )
        state_plot_outputs.append(output)

    tree_output = inference_path.parent / "tree_with_inferred_transition_times.png"
    node_colors = [
        (node, STATE_NAMES[int(data["state"])])
        for node, data in tree.nodes(data=True)
        if 0 <= int(data["state"]) < len(STATE_NAMES)
    ]
    transition_heights = sorted(root_time - time_value for time_value in transition_times)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
            category=UserWarning,
        )
        draw_ultrametric_tree(
            tree,
            root=root,
            add_labels=False,
            figsize=(20, 6),
            node_colors=node_colors,
            diff_times=transition_heights,
        )
    axis = plt.gca()
    axis.set_title("UPGMA ultrametric SGOOTR tree with inferred transition times")
    axis.figure.tight_layout()
    axis.figure.savefig(tree_output, dpi=180, bbox_inches="tight")
    plt.close(axis.figure)

    profile_output_dir = inference_path.parent / "profiles"
    profile_outputs = _profile_outputs(
        inference_path,
        profile_output_dir,
        _cohort_from_inference_path(inference_path),
        birth,
        death,
        sampling,
        transition_matrices,
        transition_times,
        datastructure,
        root_time,
        config,
    )
    outputs = [tree_output, *state_plot_outputs, *profile_outputs]
    print(f"Verified profile baseline: {baseline:.15g}")
    for output in outputs:
        print(f"Wrote {output}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot state-specific CRC11 transition inference results."
    )
    parser.add_argument("inference", type=Path)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--likelihood-drops",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 1.0],
    )
    args = parser.parse_args()
    plot_state_specific_inference(
        args.inference,
        args.input_dir,
        likelihood_drops=tuple(args.likelihood_drops),
    )


if __name__ == "__main__":
    main()
