from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
import json
import math
from pathlib import Path
import re
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch

from src.closed_form_given_transitions import calculate_Us, calculate_log_likelihood
from src.dataprocessor import load_edges, load_nodes
from src.main2_given_transitions import find_tree_file_pairs
from src.results.tls.config import PipelineConfig, load_config
from src.results.tls.data import (
    DatasetContext,
    SubtreeLikelihoodInput,
    load_inference_parameters,
)
from src.results.tls.likelihood import evaluate_total_log_likelihood
from src.tree_utils import (
    cut_tree,
    get_datastructure_for_likelihood_calculation,
    prepare_nx_tree,
)


LIKELIHOOD_LINE = re.compile(r"(?:ITER\s+\d+\s+)?LL=(\S+)")
OUTPUT_FILENAME = "edge_segment_state_likelihoods.csv"


@dataclass(frozen=True)
class SubtreeInput:
    name: str
    root_time: float
    levels: list[list[nx.DiGraph]]
    datastructure: list[dict[str, Any]]


def _read_log_likelihood(path: Path) -> float:
    matches = [
        match.group(1)
        for line in path.read_text().splitlines()
        if (match := LIKELIHOOD_LINE.fullmatch(line))
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


def _load_inputs(
    input_dir: Path,
    inference_path: Path,
    config: PipelineConfig,
) -> tuple[DatasetContext, list[SubtreeInput]]:
    parameters = load_inference_parameters(inference_path, config.state_names)
    if (
        parameters.tree_sampling_probabilities is not None
        or parameters.tree_transition_probabilities is not None
    ):
        raise ValueError("State coloring currently requires shared sampling and transitions.")

    subtree_inputs = []
    sensitivity_inputs = []
    tree_file_pairs = find_tree_file_pairs(input_dir)
    for node_path, edge_path in tree_file_pairs:
        with redirect_stdout(StringIO()):
            nodes = load_nodes(str(node_path))
            edges = load_edges(str(edge_path))
            tree = prepare_nx_tree(nodes, edges)

        invalid_leaf_states = [
            node.state
            for node in nodes
            if node.is_leaf and not 0 <= node.state < len(config.state_names)
        ]
        if invalid_leaf_states:
            raise ValueError(
                f"Invalid leaf states in {node_path}: "
                f"{sorted(set(invalid_leaf_states))}"
            )

        root = tree.graph["root"]
        root_time = float(tree.nodes[root]["T"])
        local_transition_times = [
            time for time in parameters.transition_times if time < root_time
        ]
        levels, hierarchy = cut_tree(tree, local_transition_times)
        with redirect_stdout(StringIO()):
            datastructure = get_datastructure_for_likelihood_calculation(
                levels,
                hierarchy,
                parameters.transition_times,
                len(config.state_names),
            )

        subtree_name = node_path.parent.relative_to(input_dir).as_posix()
        subtree_inputs.append(
            SubtreeInput(subtree_name, root_time, levels, datastructure)
        )
        sensitivity_inputs.append(
            SubtreeLikelihoodInput(
                root_time,
                parameters.transition_times,
                datastructure,
            )
        )

    if not subtree_inputs:
        raise FileNotFoundError(f"No subtree node/edge pairs found in {input_dir}")
    context = DatasetContext(
        birth_rates=parameters.birth_rates,
        death_rates=parameters.death_rates,
        sampling_probabilities=parameters.sampling_probabilities,
        transition_probabilities=parameters.transition_probabilities,
        transition_times=parameters.transition_times,
        sensitivity_inputs=sensitivity_inputs,
    )
    return context, subtree_inputs


def _local_component_likelihoods(
    datastructure: list[dict[str, Any]],
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    sampling_probabilities: torch.Tensor,
    Cs: torch.Tensor,
    transition_times: list[float],
) -> list[torch.Tensor]:
    dtype = birth_rates.dtype
    device = birth_rates.device
    rates = birth_rates - death_rates
    log_birth = torch.log(birth_rates)
    log_sampling = torch.log(
        sampling_probabilities.clamp_min(torch.finfo(dtype).tiny)
    )
    epoch_times = torch.as_tensor(
        [0.0, *transition_times],
        dtype=dtype,
        device=device,
    )
    local_likelihoods: list[torch.Tensor | None] = [None] * len(datastructure)

    for reverse_index, level in enumerate(reversed(datastructure)):
        level_index = len(datastructure) - reverse_index - 1
        num_births = level["num_births"].to(device=device, dtype=dtype)
        sampling_counts = level["num_sampling_by_state"].to(
            device=device,
            dtype=dtype,
        )
        sample_totals = sampling_counts.sum(dim=1)
        sampling_log_likelihood = sampling_counts @ log_sampling
        local = (
            num_births[:, None] * log_birth[None, :]
            + sampling_log_likelihood[:, None]
        )
        state_matches_samples = sampling_counts == sample_totals[:, None]
        local = torch.where(
            state_matches_samples,
            local,
            torch.full_like(local, -torch.inf),
        )

        start_times = level["start_times"].to(device=device, dtype=dtype)
        end_times = level["end_times"].to(device=device, dtype=dtype)
        edge_matrix = level["edge_matrix"].to(device=device, dtype=dtype)
        epoch_start = epoch_times[reverse_index]
        child_elapsed = end_times[:, None] - epoch_start
        parent_elapsed = start_times[:, None] - epoch_start
        edge_likelihoods = -rates[None, :] * (
            start_times[:, None] - end_times[:, None]
        )
        edge_likelihoods += 2.0 * (
            torch.log(
                1.0
                - Cs[reverse_index, :]
                * torch.exp(-rates[None, :] * child_elapsed)
            )
            - torch.log(
                1.0
                - Cs[reverse_index, :]
                * torch.exp(-rates[None, :] * parent_elapsed)
            )
        )
        local += edge_matrix @ edge_likelihoods
        local_likelihoods[level_index] = local

    return [value for value in local_likelihoods if value is not None]


def _component_state_posteriors(
    datastructure: list[dict[str, Any]],
    birth_rates: torch.Tensor,
    death_rates: torch.Tensor,
    sampling_probabilities: torch.Tensor,
    transition_probabilities: torch.Tensor,
    Cs: torch.Tensor,
    transition_times: list[float],
) -> tuple[list[torch.Tensor], float]:
    local = _local_component_likelihoods(
        datastructure,
        birth_rates,
        death_rates,
        sampling_probabilities,
        Cs,
        transition_times,
    )
    log_transitions = torch.log(
        transition_probabilities.clamp_min(
            torch.finfo(transition_probabilities.dtype).tiny
        )
    )

    upward: list[torch.Tensor | None] = [None] * len(datastructure)
    child_messages: list[torch.Tensor | None] = [None] * max(
        0, len(datastructure) - 1
    )
    upward[-1] = local[-1]
    for level_index in range(len(datastructure) - 2, -1, -1):
        child_upward = upward[level_index + 1]
        assert child_upward is not None
        messages = torch.logsumexp(
            log_transitions[None, :, :] + child_upward[:, None, :],
            dim=2,
        )
        child_messages[level_index] = messages
        adjacency = datastructure[level_index]["transition_matrix"].to(
            device=birth_rates.device,
            dtype=birth_rates.dtype,
        )
        upward[level_index] = local[level_index] + adjacency @ messages

    root_upward = upward[0]
    assert root_upward is not None
    if root_upward.shape[0] != 1:
        raise ValueError("Each subtree must have exactly one root component.")
    root_log_likelihood = float(root_upward[0, 0])

    downward: list[torch.Tensor | None] = [None] * len(datastructure)
    downward[0] = torch.full_like(root_upward, -torch.inf)
    downward[0][0, 0] = 0.0
    for level_index in range(len(datastructure) - 1):
        parent_downward = downward[level_index]
        assert parent_downward is not None
        adjacency = datastructure[level_index]["transition_matrix"]
        messages = child_messages[level_index]
        assert messages is not None
        child_count = messages.shape[0]
        child_downward = torch.full(
            (child_count, len(birth_rates)),
            -torch.inf,
            dtype=birth_rates.dtype,
            device=birth_rates.device,
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
                    + local[level_index][parent_index]
                )
                for sibling_index in children:
                    if sibling_index != child_index:
                        outside = outside + messages[sibling_index]
                child_downward[child_index] = torch.logsumexp(
                    outside[:, None] + log_transitions,
                    dim=0,
                )
                if child_index in assigned_children:
                    raise ValueError("A subtree component has multiple parent components.")
                assigned_children.add(child_index)
        if len(assigned_children) != child_count:
            raise ValueError("A subtree component is disconnected from its root.")
        downward[level_index + 1] = child_downward

    posteriors = []
    for upward_level, downward_level in zip(upward, downward):
        assert upward_level is not None and downward_level is not None
        joint = upward_level + downward_level
        log_normalizer = torch.logsumexp(joint, dim=1, keepdim=True)
        if not torch.isfinite(log_normalizer).all():
            raise ValueError("Could not normalize a component's state likelihoods.")
        posteriors.append(torch.exp(joint - log_normalizer))
    return posteriors, root_log_likelihood


def _component_rows(
    dataset: str,
    subtree: SubtreeInput,
    posterior_by_level: list[torch.Tensor],
    tree_log_likelihood: float,
    state_names: tuple[str, ...],
) -> list[dict[str, object]]:
    rows = []
    for level_index, components in enumerate(subtree.levels):
        posterior = posterior_by_level[level_index].detach().cpu().numpy()
        for component_index, component in enumerate(components):
            component_root = component.graph["root"]
            component_root_name = str(component_root)
            probabilities = posterior[component_index]
            best_state = int(np.argmax(probabilities))
            best_probability = float(probabilities[best_state])
            for segment_index, (parent, child) in enumerate(component.edges()):
                parent_data = component.nodes[parent]
                child_data = component.nodes[child]
                original_edge = (
                    parent_data.get("cut_edge")
                    or child_data.get("cut_edge")
                    or (parent, child)
                )
                for state_index, probability in enumerate(probabilities):
                    state_probability = float(probability)
                    log_joint = (
                        tree_log_likelihood + math.log(state_probability)
                        if state_probability > 0.0
                        else -math.inf
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "tree": subtree.name.split("/")[0],
                            "subtree": subtree.name,
                            "segment_id": (
                                f"{subtree.name}:level_{level_index}:"
                                f"component_{component_index}:segment_{segment_index}"
                            ),
                            "level": level_index,
                            "component_index": component_index,
                            "component_root": component_root_name,
                            "original_edge": (
                                f"{original_edge[0]}->{original_edge[1]}"
                            ),
                            "segment_parent_node": str(parent),
                            "segment_child_node": str(child),
                            "segment_start_time": float(parent_data["T"]),
                            "segment_end_time": float(child_data["T"]),
                            "state_index": state_index,
                            "state": state_names[state_index],
                            "log_likelihood_given_state": log_joint,
                            "state_probability": state_probability,
                            "is_most_likely_state": state_index == best_state,
                            "most_likely_state_index": best_state,
                            "most_likely_state": state_names[best_state],
                            "most_likely_state_probability": best_probability,
                        }
                    )
    return rows


def process_dataset(
    dataset: str,
    inference_path: Path,
    input_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
) -> Path:
    context, subtree_inputs = _load_inputs(input_dir, inference_path, config)
    baseline = evaluate_total_log_likelihood(
        context,
        context.birth_rates,
        context.death_rates,
        context.sampling_probabilities,
        context.transition_probabilities,
        normalize_root_likelihood=True,
    )
    fitted_log_likelihood = _read_log_likelihood(inference_path)
    if not np.isclose(baseline, fitted_log_likelihood, rtol=1e-8, atol=1e-8):
        raise ValueError(
            "State-coloring likelihood does not match the inference result: "
            f"{baseline:.15g} != {fitted_log_likelihood:.15g}."
        )

    u_result = calculate_Us(
        context.birth_rates,
        context.death_rates,
        context.sampling_probabilities,
        context.transition_probabilities,
        torch.tensor(context.transition_times, dtype=torch.float64),
        root_times=context.root_times,
    )
    rows = []
    for tree_index, subtree in enumerate(subtree_inputs):
        posterior_by_level, root_log_likelihood = _component_state_posteriors(
            subtree.datastructure,
            context.birth_rates,
            context.death_rates,
            context.sampling_probabilities,
            context.transition_probabilities,
            u_result.Cs,
            context.transition_times,
        )
        likelihood_by_root_state = calculate_log_likelihood(
            subtree.datastructure,
            context.birth_rates,
            context.death_rates,
            context.transition_probabilities,
            context.sampling_probabilities,
            u_result.Cs,
            context.transition_times,
        )
        if not np.isclose(
            root_log_likelihood,
            float(likelihood_by_root_state[0, 0]),
            rtol=1e-10,
            atol=1e-10,
        ):
            raise ValueError(
                f"Component likelihood recursion disagrees with inference for "
                f"{subtree.name}."
            )
        normalized_tree_likelihood = root_log_likelihood - float(
            torch.log1p(-u_result.root_U[tree_index, 0])
        )
        rows.extend(
            _component_rows(
                dataset,
                subtree,
                posterior_by_level,
                normalized_tree_likelihood,
                config.state_names,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    pd.DataFrame(rows).to_csv(output_path, index=False)
    metadata = {
        "dataset": dataset,
        "inference_file": str(inference_path.resolve()),
        "input_directory": str(input_dir.resolve()),
        "subtree_count": len(subtree_inputs),
        "edge_segment_count": len({row["segment_id"] for row in rows}),
        "state_names": list(config.state_names),
        "transition_time_count": len(context.transition_times),
        "log_likelihood_normalized_at_root": baseline,
        "birth_rates": context.birth_rates.tolist(),
        "death_rates": context.death_rates.tolist(),
        "sampling_probabilities": context.sampling_probabilities.tolist(),
        "transition_probabilities": context.transition_probabilities.tolist(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        f"{dataset}: processed {len(subtree_inputs)} subtrees and "
        f"{metadata['edge_segment_count']} edge segments; wrote {output_path}"
    )
    return output_path


def _default_inference(dataset: str) -> Path:
    run_dir = Path("data/troupe") / dataset / "fixed_sampling_0.1"
    for suffix in (".out", ".tmp.out"):
        path = run_dir / f"inference_transition_times_100{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No completed or checkpoint 100-transition fit found under {run_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-segment state likelihoods and posterior state "
            "probabilities for all TLS or TLSC subtrees."
        )
    )
    parser.add_argument("--dataset", choices=("TLS", "TLSC"), required=True)
    parser.add_argument("--inference", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config_fixed_sampling_0.1.yaml"),
    )
    arguments = parser.parse_args()
    inference_path = arguments.inference or _default_inference(arguments.dataset)
    input_dir = arguments.input_dir or (
        Path("data/troupe") / arguments.dataset / "subtrees_kept_states"
    )
    output_dir = arguments.output_dir or (
        Path("outputs/tls/state_coloring") / arguments.dataset
    )
    process_dataset(
        arguments.dataset,
        inference_path,
        input_dir,
        output_dir,
        load_config(arguments.config),
    )


if __name__ == "__main__":
    main()
